#!/usr/bin/env python3
"""Plot the saved DIS ResNet feasibility and evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


COLORS = {"small_3100": "#3366CC", "large_10000": "#CC6633"}
MARKERS = {"hard": "s", "soft": "o"}


def save_both(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def workflow(directory: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 3.1))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.2, "GZ2 raw votes\nmapping + image audit"),
        (2.6, "Object/image dedup\nhistorical pools excluded"),
        (5.0, "Frozen development\n3100 / 10000 train + 750 val"),
        (7.4, "Paired ResNet-18\nhard argmax vs soft votes"),
        (9.8, "New natural pool\n10,000 objects"),
    ]
    for x, label in boxes:
        patch = FancyBboxPatch((x, 1.05), 1.8, 0.9, boxstyle="round,pad=0.04", facecolor="#E8EEF7", edgecolor="#315A8A", linewidth=1.2)
        axis.add_patch(patch)
        axis.text(x + 0.9, 1.5, label, ha="center", va="center", fontsize=9)
    for left in [2.0, 4.4, 6.8, 9.2]:
        axis.annotate("", xy=(left + 0.55, 1.5), xytext=(left, 1.5), arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2})
    axis.annotate("class-2 diagnostic: 5 unseen objects; recall only", xy=(10.7, 1.05), xytext=(10.7, 0.35), ha="center", va="center", fontsize=8.5, arrowprops={"arrowstyle": "->", "color": "#777777"})
    axis.text(0.2, 2.55, "DIS round-2 protocol: vote distributions remain targets; the historical pilot and historical diagnostic pools are not reused", fontsize=10, weight="bold")
    save_both(fig, directory, "workflow")


def resnet_framework(directory: Path) -> None:
    fig, axis = plt.subplots(figsize=(11.4, 2.8))
    axis.set_xlim(0, 11.7)
    axis.set_ylim(0, 2.8)
    axis.axis("off")
    boxes = [
        (0.10, "ImageNet-pretrained\nResNet-18", "#E8EEF7"),
        (2.45, "Replace classifier\nThree outputs", "#E8EEF7"),
        (4.80, "Fine-tune\nlayer4 + classifier", "#FCE8D5"),
        (7.15, "Paired training\nHard vs soft targets", "#E8EEF7"),
        (9.50, "Select checkpoint\nLowest validation\nsoft CE", "#E8EEF7"),
    ]
    box_width = 2.05
    box_height = 1.15
    center_y = 1.45
    for x, label, color in boxes:
        patch = FancyBboxPatch((x, center_y - box_height / 2), box_width, box_height, boxstyle="round,pad=0.04", facecolor=color, edgecolor="#315A8A", linewidth=1.2)
        axis.add_patch(patch)
        axis.text(x + box_width / 2, center_y, label, ha="center", va="center", fontsize=8.6, linespacing=1.15)
    for left, right in zip(boxes[:-1], boxes[1:]):
        start = left[0] + box_width + 0.03
        end = right[0] - 0.03
        axis.annotate("", xy=(end, center_y), xytext=(start, center_y), arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2, "shrinkA": 0, "shrinkB": 0})
    axis.text(0.10, 2.50, "ResNet transfer-learning protocol", fontsize=11, weight="bold")
    axis.text(5.85, 0.25, "hard = one-hot argmax of raw votes     soft = normalized raw vote fractions", fontsize=8.8, ha="center")
    save_both(fig, directory, "resnet_framework")


def results(directory: Path, summary: dict, evaluation: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    records = summary["records"]
    for size in ["small_3100", "large_10000"]:
        for target in ["hard", "soft"]:
            rows = [row for row in records if row["train_size"] == size and row["target"] == target]
            x = [row["natural_evaluation"]["soft_cross_entropy"] for row in rows]
            y = [row["natural_evaluation"]["hard_accuracy"] for row in rows]
            axes[0].scatter(x, y, s=58, marker=MARKERS[target], color=COLORS[size], edgecolor="white", linewidth=0.6, label=f"{size.replace('_', ' ')} {target}")
    for name, base in summary["baselines"].items():
        metrics = base["natural_evaluation"]
        axes[0].scatter(metrics["soft_cross_entropy"], metrics["hard_accuracy"], marker="X", s=80, color="#333333", label=name.replace("training_", "baseline: ").replace("_", " "))
    axes[0].set_xlabel("Natural-pool soft cross-entropy (lower is better)")
    axes[0].set_ylabel("Natural-pool hard accuracy")
    axes[0].set_title("Probability fit versus recognition")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="lower left")

    eval_lookup = {(row["train_size"], row["target"], row["seed"]): row for row in evaluation["records"]}
    for index, size in enumerate(["small_3100", "large_10000"]):
        for offset, target in [(-0.16, "hard"), (0.16, "soft")]:
            rows = [eval_lookup[(size, target, seed)] for seed in [20261119, 20261120, 20261121]]
            values = [row["natural_evaluation"]["predicted_class_counts"][2] for row in rows]
            x = np.full(len(values), index + 1 + offset) + np.linspace(-0.035, 0.035, len(values))
            axes[1].scatter(x, values, color=COLORS[size], marker=MARKERS[target], s=55, edgecolor="white", linewidth=0.6)
            axes[1].plot([index + 1 + offset - 0.07, index + 1 + offset + 0.07], [np.mean(values), np.mean(values)], color=COLORS[size], linewidth=2)
    axes[1].set_xticks([1, 2], ["3,100 train", "10,000 train"])
    axes[1].set_ylabel("False positives predicted as class-2")
    axes[1].set_title("Natural pool: class-2 false positives")
    axes[1].text(0.02, 0.88, "10,000 objects; true class-2 count = 0\nline = mean over three seeds", transform=axes[1].transAxes, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    for index, size in enumerate(["small_3100", "large_10000"]):
        for offset, target in [(-0.16, "hard"), (0.16, "soft")]:
            rows = [row for row in records if row["train_size"] == size and row["target"] == target]
            values = [row["class2_diagnostic_recall_only"]["recall"] for row in rows]
            x = np.full(len(values), index + 1 + offset) + np.linspace(-0.035, 0.035, len(values))
            axes[2].scatter(x, values, color=COLORS[size], marker=MARKERS[target], s=60, edgecolor="white", linewidth=0.6)
            axes[2].plot([index + 1 + offset - 0.07, index + 1 + offset + 0.07], [np.mean(values), np.mean(values)], color=COLORS[size], linewidth=2)
    axes[2].axhline(0.0, color="#333333", linestyle=":", linewidth=1)
    axes[2].set_xticks([1, 2], ["3,100 train", "10,000 train"])
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].set_ylabel("Recall on 5 class-2 objects")
    axes[2].set_title("Diagnostic recall")
    axes[2].text(0.02, 0.77, "Hard squares; soft circles\npoint = one seed; line = seed mean", transform=axes[2].transAxes, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    save_both(fig, directory, "probability_vs_recall")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "round2_resnet/results"
    summary = json.loads((directory / "formal_summary.json").read_text())
    evaluation = json.loads((directory / "formal_evaluation.json").read_text())
    workflow(directory)
    resnet_framework(directory)
    results(directory, summary, evaluation)
    print(json.dumps({"output_dir": str(directory), "figures": ["workflow", "resnet_framework", "probability_vs_recall"]}, indent=2))


if __name__ == "__main__":
    main()
