#!/usr/bin/env python3
"""Create traceable baseline, paired-difference, and bootstrap summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW = ["raw_smooth_normalized", "raw_features_or_disk_normalized", "raw_star_or_artifact_normalized"]


def metrics(probability: np.ndarray, target: np.ndarray, hard: np.ndarray) -> dict:
    """Compute soft CE/Brier and hard-label counts for one object pool."""
    log_probability = np.log(np.clip(probability, 1e-12, 1.0))
    prediction = probability.argmax(axis=1)
    return {
        "n": int(len(hard)),
        "soft_cross_entropy": float(np.mean(-np.sum(target * log_probability, axis=1))),
        "soft_brier": float(np.mean(np.sum((probability - target) ** 2, axis=1))),
        "hard_accuracy": float(np.mean(prediction == hard)),
        "target_class_counts": np.bincount(hard, minlength=3).astype(int).tolist(),
        "predicted_class_counts": np.bincount(prediction, minlength=3).astype(int).tolist(),
    }


def bootstrap_difference(soft_prob: np.ndarray, hard_prob: np.ndarray, target: np.ndarray, hard: np.ndarray, seed: int, reps: int = 2000) -> dict:
    """Bootstrap paired soft-minus-hard differences over fixed objects."""
    rng = np.random.default_rng(seed)
    n = len(hard)
    soft_ce = -np.sum(target * np.log(np.clip(soft_prob, 1e-12, 1.0)), axis=1)
    hard_ce = -np.sum(target * np.log(np.clip(hard_prob, 1e-12, 1.0)), axis=1)
    soft_brier = np.sum((soft_prob - target) ** 2, axis=1)
    hard_brier = np.sum((hard_prob - target) ** 2, axis=1)
    soft_acc = (soft_prob.argmax(axis=1) == hard).astype(float)
    hard_acc = (hard_prob.argmax(axis=1) == hard).astype(float)
    result = {}
    for name, values in {
        "soft_minus_hard_ce": soft_ce - hard_ce,
        "soft_minus_hard_brier": soft_brier - hard_brier,
        "soft_minus_hard_accuracy": soft_acc - hard_acc,
    }.items():
        observed = float(np.mean(values))
        boot = np.empty(reps, dtype=float)
        for index in range(reps):
            sample = rng.integers(0, n, size=n)
            boot[index] = float(np.mean(values[sample]))
        result[name] = {"estimate": observed, "bootstrap_95_percent_interval": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "resamples": reps, "unit": "objects from the fixed natural evaluation pool"}
    return result


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a Wilson interval for a binomial proportion."""
    if total == 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def main() -> None:
    """Rebuild the formal summary from the frozen prediction JSON."""
    root = Path(__file__).resolve().parents[2]
    split = pd.read_csv(root / "round2_resnet/results/resnet_split.csv", dtype={"pool_role": "string", "image_path": "string"})
    train = split[split["large_train"]].copy()
    eval_data = json.loads((root / "round2_resnet/results/formal_evaluation.json").read_text())
    eval_split = pd.read_csv(root / "round2_resnet/results/new_eval_with_images.csv", dtype={"pool_role": "string"})
    natural = eval_split["pool_role"].eq("new_natural_evaluation").to_numpy()
    diagnostic = eval_split["pool_role"].eq("new_class2_diagnostic_recall_only").to_numpy()
    target = eval_split[RAW].to_numpy(dtype=float)
    hard = eval_split["hard_class"].to_numpy(dtype=int)
    soft_baseline = train[RAW].to_numpy(dtype=float).mean(axis=0)
    hard_baseline = np.bincount(train["hard_class"].to_numpy(dtype=int), minlength=3).astype(float)
    hard_baseline /= hard_baseline.sum()
    baselines = {
        "training_mean_soft_label": {"training_rows": int(len(train)), "probability": soft_baseline.tolist(), "natural_evaluation": metrics(np.tile(soft_baseline, (natural.sum(), 1)), target[natural], hard[natural])},
        "training_hard_class_frequency": {"training_rows": int(len(train)), "probability": hard_baseline.tolist(), "natural_evaluation": metrics(np.tile(hard_baseline, (natural.sum(), 1)), target[natural], hard[natural])},
    }
    records = []
    lookup = {}
    for record in eval_data["records"]:
        probs = np.asarray(record["natural_evaluation"]["probabilities"], dtype=float)
        lookup[(record["train_size"], record["seed"], record["target"])] = probs
        natural_metrics = metrics(probs, target[natural], hard[natural])
        diag_probs = np.asarray(record["class2_diagnostic_recall_only"]["probabilities"], dtype=float)
        diag_hard = hard[diagnostic]
        successes = int(np.sum(diag_probs.argmax(axis=1) == diag_hard))
        records.append({
            "train_size": record["train_size"],
            "target": record["target"],
            "seed": record["seed"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "natural_evaluation": natural_metrics,
            "natural_class2_false_positive": {
                "count": int(natural_metrics["predicted_class_counts"][2]),
                "rate": float(natural_metrics["predicted_class_counts"][2] / natural_metrics["n"]),
                "denominator_definition": "all 10,000 natural-pool objects; because the pool contains zero class-2 targets, this is also the rate among known non-class-2 objects",
            },
            "class2_diagnostic_recall_only": {"successes": successes, "total": int(len(diag_hard)), "recall": successes / len(diag_hard), "wilson_95_percent_interval": wilson(successes, len(diag_hard))},
        })
    paired = []
    for train_size in ["small_3100", "large_10000"]:
        for seed in [20261119, 20261120, 20261121]:
            soft = lookup[(train_size, seed, "soft")]
            hard_prob = lookup[(train_size, seed, "hard")]
            paired.append({"train_size": train_size, "seed": seed, "difference": bootstrap_difference(soft, hard_prob, target[natural], hard[natural], seed + (0 if train_size == "small_3100" else 100), reps=2000)})
    means_sd = []
    for train_size in ["small_3100", "large_10000"]:
        for metric_name in ["soft_minus_hard_ce", "soft_minus_hard_brier", "soft_minus_hard_accuracy"]:
            values = [row["difference"][metric_name]["estimate"] for row in paired if row["train_size"] == train_size]
            means_sd.append({"train_size": train_size, "metric": metric_name, "mean": float(np.mean(values)), "sample_sd_across_three_seeds": float(np.std(values, ddof=1)), "seed_count": len(values)})
    output = {
        "status": "formal_evaluation_summary_complete",
        "evaluation_policy": eval_data["policy"],
        "natural_evaluation_rows": int(natural.sum()),
        "diagnostic_rows": int(diagnostic.sum()),
        "baselines": baselines,
        "records": records,
        "false_positive_summary": [
            {
                "train_size": train_size,
                "target": target_name,
                "mean_count_across_three_seeds": float(np.mean([row["natural_class2_false_positive"]["count"] for row in records if row["train_size"] == train_size and row["target"] == target_name])),
                "mean_rate_across_three_seeds": float(np.mean([row["natural_class2_false_positive"]["rate"] for row in records if row["train_size"] == train_size and row["target"] == target_name])),
            }
            for train_size in ["small_3100", "large_10000"]
            for target_name in ["hard", "soft"]
        ],
        "paired_soft_minus_hard": paired,
        "paired_mean_and_sample_sd": means_sd,
        "interval_note": "Bootstrap intervals resample fixed natural-evaluation objects and do not measure seed variation. Seed sample SD is reported separately and is not a 95% confidence interval.",
    }
    output_path = root / "round2_resnet/results/formal_summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(means_sd).to_csv(root / "round2_resnet/results/paired_seed_summary.csv", index=False)
    print(json.dumps({"output": str(output_path), "paired_rows": len(paired), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
