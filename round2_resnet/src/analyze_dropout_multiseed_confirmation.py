"""Summarize paired soft-target dropout results across the three seeds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def validation_value(row: dict, name: str) -> float:
    """Read a validation metric from old or new result-field names."""

    value = row["validation"]
    aliases = {
        "soft_ce": ("soft_cross_entropy", "soft_label_cross_entropy"),
        "brier": ("soft_brier", "brier"),
        "calibration": ("soft_calibration_discrepancy", "calibration_discrepancy"),
        "accuracy": ("hard_accuracy",),
    }
    for key in aliases[name]:
        if key in value:
            return float(value[key])
    raise KeyError(name)


def select_metrics(record: dict) -> dict:
    """Extract the pre-specified best epoch and last-ten soft-CE mean."""

    history = record["history"]
    best = min(history, key=lambda row: (validation_value(row, "soft_ce"), validation_value(row, "calibration"), row["epoch"]))
    return {
        "best_epoch": int(best["epoch"]),
        "best_soft_ce": validation_value(best, "soft_ce"),
        "best_brier": validation_value(best, "brier"),
        "best_accuracy": validation_value(best, "accuracy"),
        "last_ten_soft_ce_mean": sum(validation_value(row, "soft_ce") for row in history[-10:]) / 10.0,
    }


def sample_mean_sd(values: list[float]) -> dict[str, float | None]:
    """Return a mean and sample SD, leaving SD undefined for one value."""

    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def main() -> None:
    """Build the paired per-seed and aggregate summary JSON."""

    root = Path(__file__).resolve().parents[2]
    new_path = root / "round2_resnet/results/closeout/dropout_multiseed_confirmation/dropout_multiseed_confirmation_runs.json"
    original_path = root / "round2_resnet/results/closeout/training_budget_diagnostic_runs.json"
    dropout_path = root / "round2_resnet/results/closeout/regularization_diagnostic/regularization_diagnostic_runs.json"
    new_records = json.loads(new_path.read_text())["records"]
    original_records = json.loads(original_path.read_text())["records"]
    dropout_records = json.loads(dropout_path.read_text())["records"]
    original_20261119 = next(r for r in original_records if r["learning_rate"] == 0.0003 and r["target"] == "soft")
    dropout_20261119 = next(r for r in dropout_records if r["run_name"] == "dropout_0.2" and r["target"] == "soft")
    by_seed: dict[int, dict[str, dict]] = {
        20261119: {"original": original_20261119, "dropout": dropout_20261119},
    }
    for seed in [20261120, 20261121]:
        by_seed[seed] = {
            "original": next(r for r in new_records if r["seed"] == seed and r["dropout"] == 0.0),
            "dropout": next(r for r in new_records if r["seed"] == seed and r["dropout"] == 0.2),
        }

    rows = []
    for seed, records in by_seed.items():
        original = select_metrics(records["original"])
        dropout = select_metrics(records["dropout"])
        rows.append({
            "seed": seed,
            "seed_role": "screened_candidate" if seed == 20261119 else "new_confirmation",
            "original": original,
            "dropout": dropout,
            "dropout_minus_original": {
                "best_soft_ce": dropout["best_soft_ce"] - original["best_soft_ce"],
                "best_brier": dropout["best_brier"] - original["best_brier"],
                "best_accuracy": dropout["best_accuracy"] - original["best_accuracy"],
                "last_ten_soft_ce_mean": dropout["last_ten_soft_ce_mean"] - original["last_ten_soft_ce_mean"],
            },
        })

    def aggregate(seed_rows: list[dict]) -> dict:
        differences = [row["dropout_minus_original"] for row in seed_rows]
        return {
            "seed_count": len(seed_rows),
            "seeds": [row["seed"] for row in seed_rows],
            "dropout_minus_original": {
                name: sample_mean_sd([float(diff[name]) for diff in differences])
                for name in ["best_soft_ce", "best_brier", "best_accuracy", "last_ten_soft_ce_mean"]
            },
        }

    output = {
        "status": "complete",
        "policy": "seed 20261119 is reused as a screened candidate; 20261120 and 20261121 are new confirmation seeds",
        "selection_rule": "minimum validation soft CE, then calibration discrepancy, then earliest epoch",
        "last_ten_definition": "mean validation soft CE over epochs 11--20",
        "independent_objects_claim": False,
        "source_files": {
            "new_confirmation": str(new_path.relative_to(root)),
            "screened_original": str(original_path.relative_to(root)),
            "screened_dropout": str(dropout_path.relative_to(root)),
        },
        "per_seed": rows,
        "aggregate_new_confirmation_seeds": aggregate([row for row in rows if row["seed"] != 20261119]),
        "aggregate_all_three_descriptive": aggregate(rows),
    }
    output_path = root / "round2_resnet/results/closeout/dropout_multiseed_confirmation/dropout_multiseed_summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "seeds": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
