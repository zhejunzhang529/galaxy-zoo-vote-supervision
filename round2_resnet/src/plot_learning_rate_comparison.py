"""Plot the four-learning-rate validation curves from saved JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_runs(path: Path, source: str) -> list[dict]:
    """Load diagnostic records and annotate their source file."""

    data = json.loads(path.read_text())
    return [{**record, "source": source} for record in data["records"]]


def metric(row: dict, name: str) -> float:
    """Read a common metric name across old and endpoint result schemas."""

    validation = row["validation"]
    if name == "soft_ce":
        value = validation["soft_cross_entropy"] if "soft_cross_entropy" in validation else validation["soft_label_cross_entropy"]
        return float(value)
    if name == "accuracy":
        return float(validation["hard_accuracy"])
    if name == "brier":
        value = validation["soft_brier"] if "soft_brier" in validation else validation["brier"]
        return float(value)
    raise KeyError(name)


def plot(records: list[dict], output: Path) -> None:
    """Write validation soft-CE and accuracy curves for hard and soft targets."""

    colors = {1e-5: "#9467bd", 1e-4: "#1f77b4", 3e-4: "#2ca02c", 1e-3: "#d62728"}
    labels = {1e-5: "1e-5", 1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row_index, target in enumerate(["hard", "soft"]):
        target_records = sorted((r for r in records if r["target"] == target), key=lambda r: r["learning_rate"])
        for record in target_records:
            epochs = [h["epoch"] for h in record["history"]]
            colour = colors[float(record["learning_rate"])]
            label = labels[float(record["learning_rate"])]
            axes[row_index, 0].plot(epochs, [metric(h, "soft_ce") for h in record["history"]], color=colour, label=label)
            axes[row_index, 1].plot(epochs, [metric(h, "accuracy") for h in record["history"]], color=colour, label=label)
        axes[row_index, 0].set_title(f"{target} target: validation soft CE")
        axes[row_index, 1].set_title(f"{target} target: validation accuracy")
        for column in range(2):
            axes[row_index, column].grid(alpha=0.25, linewidth=0.6)
            axes[row_index, column].set_xlim(1, 20)
            axes[row_index, column].set_xticks([1, 5, 10, 15, 20])
            axes[row_index, column].legend(title="learning rate", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation soft CE")
    for ax in axes[:, 1]:
        ax.set_ylabel("accuracy")
    fig.suptitle("DIS learning-rate endpoint check: seed 20261119, small_3100", y=1.03, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    """Parse result paths and save the comparison figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_runs(args.old, "existing") + load_runs(args.endpoints, "endpoint")
    if len(records) != 8:
        raise ValueError(f"Expected eight combined records, got {len(records)}")
    rates = {float(record["learning_rate"]) for record in records}
    if rates != {1e-5, 1e-4, 3e-4, 1e-3}:
        raise ValueError(f"Unexpected learning rates: {rates}")
    plot(records, args.output)


if __name__ == "__main__":
    main()
