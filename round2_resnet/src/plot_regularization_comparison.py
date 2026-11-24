"""Plot regularization diagnostics against the saved lr=3e-4 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def validation_metric(history_row: dict, name: str) -> float:
    """Read a validation metric from either the reference or new schema."""

    value = history_row["validation"]
    if name == "soft_ce":
        return float(value["soft_cross_entropy"] if "soft_cross_entropy" in value else value["soft_label_cross_entropy"])
    if name == "brier":
        return float(value["soft_brier"] if "soft_brier" in value else value["brier"])
    if name == "accuracy":
        return float(value["hard_accuracy"])
    raise KeyError(name)


def load_reference(path: Path) -> list[dict]:
    """Load the two existing lr=3e-4 reference records."""

    records = json.loads(path.read_text())["records"]
    return [
        {
            **record,
            "label": f"reference wd=1e-4, dropout=0 ({record['target']})",
            "short_label": "reference",
        }
        for record in records
        if float(record["learning_rate"]) == 3e-4
    ]


def load_regularization(path: Path) -> list[dict]:
    """Load the four new regularization records."""

    records = json.loads(path.read_text())["records"]
    return [
        {
            **record,
            "label": f"{record['run_name']} ({record['target']})",
            "short_label": record["run_name"],
        }
        for record in records
    ]


def plot(records: list[dict], output: Path) -> None:
    """Write validation CE, Brier, and accuracy curves for both targets."""

    styles = {
        "reference": ("#1f77b4", "reference"),
        "weight_decay_1e-3": ("#d62728", "wd=1e-3"),
        "dropout_0.2": ("#2ca02c", "dropout=0.2"),
    }
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    metric_names = [("soft_ce", "validation soft CE"), ("brier", "validation Brier"), ("accuracy", "validation accuracy")]
    for row_index, target in enumerate(["hard", "soft"]):
        target_records = [record for record in records if record["target"] == target]
        for record in target_records:
            colour, short_label = styles[record["short_label"]]
            epochs = [row["epoch"] for row in record["history"]]
            for column, (name, title) in enumerate(metric_names):
                axes[row_index, column].plot(
                    epochs,
                    [validation_metric(row, name) for row in record["history"]],
                    color=colour,
                    label=short_label,
                )
        for column, (_, title) in enumerate(metric_names):
            axes[row_index, column].set_title(f"{target} target: {title}")
            axes[row_index, column].grid(alpha=0.25, linewidth=0.6)
            axes[row_index, column].set_xlim(1, 20)
            axes[row_index, column].set_xticks([1, 5, 10, 15, 20])
            axes[row_index, column].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("soft CE")
    for ax in axes[:, 1]:
        ax.set_ylabel("Brier")
    for ax in axes[:, 2]:
        ax.set_ylabel("accuracy")
    fig.suptitle("DIS regularization check: lr=3e-4, seed 20261119, small_3100", y=1.03, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    """Parse saved result paths and render the comparison figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--regularization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_reference(args.reference) + load_regularization(args.regularization)
    if len(records) != 6:
        raise ValueError(f"Expected six comparison records, got {len(records)}")
    plot(records, args.output)


if __name__ == "__main__":
    main()
