"""Plot the saved 20-epoch DIS training-budget diagnostic curves.

The script reads the completed diagnostic JSON and does not load images,
checkpoints, or rerun model inference.  Each panel compares the target-specific
training loss with eval-mode soft cross-entropy on the training and validation
sets, making the difference between optimization and validation behaviour
visible for one fixed seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_records(path: Path) -> list[dict]:
    """Load and validate the completed diagnostic result file.

    Parameters
    ----------
    path:
        JSON file written by ``train_budget_diagnostic.py``.

    Returns
    -------
    list of dict
        Four completed run records, in the order stored in the result file.
    """

    data = json.loads(path.read_text())
    if data.get("status") != "complete":
        raise ValueError(f"Diagnostic status is {data.get('status')!r}, not complete")
    records = data.get("records", [])
    if len(records) != 4 or any(len(record.get("history", [])) != 20 for record in records):
        raise ValueError("Expected four records with 20 epochs each")
    return records


def make_plot(records: list[dict], output: Path) -> None:
    """Render the four-panel loss diagnostic to PNG and SVG.

    Parameters
    ----------
    records:
        Completed records returned by :func:`load_records`.
    output:
        PNG destination; an SVG with the same stem is written alongside it.
    """

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, record in zip(axes.flat, records):
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        train_target = [row["train_loss_mean"] for row in history]
        train_eval = [row["train_eval_soft_cross_entropy"] for row in history]
        validation = [row["validation"]["soft_cross_entropy"] for row in history]
        ax.plot(epochs, train_target, color="#1f77b4", label="training target loss")
        ax.plot(epochs, train_eval, color="#2ca02c", label="train eval soft CE")
        ax.plot(epochs, validation, color="#d62728", label="validation soft CE")
        ax.axvline(
            record["selected_epoch"], color="#555555", linestyle="--", linewidth=0.9,
            label="_nolegend_",
        )
        ax.set_title(
            f"lr={record['learning_rate']:.0e}, {record['target']} "
            f"(selected e{record['selected_epoch']})"
        )
        ax.set_ylabel("cross-entropy / loss")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 5, 10, 15, 20])
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "DIS extended training diagnostic: seed 20261119, small_3100, validation soft CE",
        y=1.08,
        fontsize=12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    """Parse paths and plot saved diagnostic curves."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_plot(load_records(args.input), args.output)


if __name__ == "__main__":
    main()
