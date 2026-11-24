#!/usr/bin/env python3
"""Summarise and plot the frozen original-to-extra comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW = [
    "raw_smooth_normalized",
    "raw_features_or_disk_normalized",
    "raw_star_or_artifact_normalized",
]


def vote_descriptives(frame: pd.DataFrame) -> dict:
    """Compute class proportions and vote-distribution summaries."""
    probability = frame[RAW].to_numpy(dtype=float)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1) / np.log(3.0)
    sorted_probability = np.sort(probability, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    return {
        "n": int(len(frame)),
        "hard_class_counts": frame["hard_class"].value_counts().sort_index().astype(int).to_dict(),
        "hard_class_proportions": frame["hard_class"].value_counts(normalize=True).sort_index().to_dict(),
        "normalized_vote_entropy": {"mean": float(entropy.mean()), "median": float(np.median(entropy)), "p10": float(np.quantile(entropy, 0.1)), "p90": float(np.quantile(entropy, 0.9))},
        "top_two_vote_margin": {"mean": float(margin.mean()), "median": float(np.median(margin)), "p10": float(np.quantile(margin, 0.1)), "p90": float(np.quantile(margin, 0.9))},
    }


def main() -> None:
    """Read saved original and extra results, then write comparison outputs."""
    script_root = Path(__file__).resolve().parents[2]
    if (script_root / "round2_resnet").exists():
        root = script_root
        result_dir = root / "round2_resnet/results"
    else:
        root = Path(__file__).resolve().parents[1]
        result_dir = root / "results"
    extra = json.loads((result_dir / "extra_cross_sample_results.json").read_text())
    extra_split = pd.read_csv(result_dir / "extra_cross_sample_split.csv")
    original_split = pd.read_csv(result_dir / "new_eval_with_images.csv")
    original_split = original_split[original_split["pool_role"].eq("new_natural_evaluation")].copy()
    original_eval = json.loads((result_dir / "formal_evaluation.json").read_text())
    rows = []
    for record in original_eval["records"]:
        if record["train_size"] != "large_10000":
            continue
        natural = record["natural_evaluation"]
        rows.append({"pool": "original natural", "target": record["target"], "seed": int(record["seed"]), "ce": natural["soft_cross_entropy"], "brier": natural["soft_brier"], "accuracy": natural["hard_accuracy"]})
    for record in extra["records"]:
        rows.append({"pool": "extra", "target": record["target"], "seed": int(record["seed"]), "ce": record["soft_cross_entropy"], "brier": record["soft_brier"], "accuracy": record["hard_accuracy"]})
    metrics = pd.DataFrame(rows).sort_values(["target", "seed", "pool"])
    metrics.to_csv(result_dir / "extra_cross_sample_metrics.csv", index=False)
    descriptors = {
        "comparison_label": "original-to-extra cross-sample generalisation under selection shift",
        "original_natural": vote_descriptives(original_split),
        "extra_selected": vote_descriptives(extra_split),
        "note": "Extra is a GZ2 sample with a documented selection-history difference; sample name alone is not evidence of OOD or a different survey.",
    }
    (result_dir / "extra_cross_sample_pool_comparison.json").write_text(json.dumps(descriptors, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    colors = {"hard": "#3568a8", "soft": "#d66a3d"}
    pools = ["original natural", "extra"]
    labels = ["original\nnatural", "extra"]
    for class_id, label in enumerate(["smooth", "features/disk", "star/artifact"]):
        values = [descriptors["original_natural"]["hard_class_proportions"].get(class_id, 0.0), descriptors["extra_selected"]["hard_class_proportions"].get(class_id, 0.0)]
        axes[0].bar(np.arange(2) + (class_id - 1) * 0.24, values, width=0.23, label=label)
    axes[0].set_xticks(np.arange(2), labels)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("hard-class proportion")
    axes[0].set_title("Pool composition")
    axes[0].legend(frameon=False, loc="upper right")

    entropy_values = [
        -(original_split[RAW].to_numpy(float) * np.log(np.clip(original_split[RAW].to_numpy(float), 1e-12, 1.0))).sum(axis=1) / np.log(3.0),
        -(extra_split[RAW].to_numpy(float) * np.log(np.clip(extra_split[RAW].to_numpy(float), 1e-12, 1.0))).sum(axis=1) / np.log(3.0),
    ]
    margin_values = []
    for frame in (original_split, extra_split):
        probability = frame[RAW].to_numpy(float)
        sorted_probability = np.sort(probability, axis=1)
        margin_values.append(sorted_probability[:, -1] - sorted_probability[:, -2])
    axes[1].boxplot([entropy_values[0], margin_values[0], entropy_values[1], margin_values[1]], tick_labels=["orig H", "orig M", "extra H", "extra M"], patch_artist=True, boxprops={"facecolor": "#d9e5f3"}, medianprops={"color": "#1f3d5a"})
    axes[1].set_ylabel("normalised value")
    axes[1].set_title("Vote entropy (H) / margin (M)")
    axes[1].set_ylim(0, 1.02)

    ax = axes[2]
    for target in ["hard", "soft"]:
        subset = metrics[metrics["target"].eq(target)]
        for seed in sorted(subset["seed"].unique()):
            line = subset[subset["seed"].eq(seed)].set_index("pool").reindex(pools)
            ax.plot(labels, line["ce"], marker="o", color=colors[target], alpha=0.55, linewidth=1)
        means = subset.groupby("pool")["ce"].mean().reindex(pools)
        ax.plot(labels, means, marker="o", color=colors[target], linewidth=2, label=target)
    ax.set_title("Soft-target CE across seeds")
    ax.set_ylabel("CE")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Frozen original-to-extra cross-sample comparison", fontsize=12)
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "extra_cross_sample_generalisation.png", dpi=220)
    fig.savefig(figure_dir / "extra_cross_sample_generalisation.svg")
    print(json.dumps({"metrics": str(result_dir / 'extra_cross_sample_metrics.csv'), "figure": str(figure_dir / 'extra_cross_sample_generalisation.svg')}, indent=2))


if __name__ == "__main__":
    main()
