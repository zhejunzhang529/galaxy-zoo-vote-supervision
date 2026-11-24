"""Plot paired soft-target validation curves for the three available seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def soft_ce(row: dict) -> float:
    """Read validation soft CE from either historical result schema."""

    value = row["validation"]
    return float(value["soft_cross_entropy"] if "soft_cross_entropy" in value else value["soft_label_cross_entropy"])


def get_histories(root: Path, new_path: Path, original_path: Path, dropout_path: Path) -> dict[int, dict[str, list[dict]]]:
    """Collect original/dropout histories for seed 20261119 and two new seeds."""

    old = json.loads((root / original_path).read_text())["records"]
    reg = json.loads((root / dropout_path).read_text())["records"]
    new = json.loads((root / new_path).read_text())["records"]
    histories = {
        20261119: {
            "original": next(r["history"] for r in old if r["learning_rate"] == 0.0003 and r["target"] == "soft"),
            "dropout": next(r["history"] for r in reg if r["run_name"] == "dropout_0.2" and r["target"] == "soft"),
        }
    }
    for seed in [20261120, 20261121]:
        histories[seed] = {
            "original": next(r["history"] for r in new if r["seed"] == seed and r["dropout"] == 0.0),
            "dropout": next(r["history"] for r in new if r["seed"] == seed and r["dropout"] == 0.2),
        }
    return histories


def plot(histories: dict[int, dict[str, list[dict]]], output: Path) -> None:
    """Render per-seed paired curves and dropout-minus-original differences."""

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for index, seed in enumerate([20261119, 20261120, 20261121]):
        ax = axes.flat[index]
        original = histories[seed]["original"]
        dropout = histories[seed]["dropout"]
        epochs = [row["epoch"] for row in original]
        ax.plot(epochs, [soft_ce(row) for row in original], color="#1f77b4", label="dropout=0")
        ax.plot(epochs, [soft_ce(row) for row in dropout], color="#d62728", label="dropout=0.2")
        ax.axvspan(11, 20, color="#777777", alpha=0.08, linewidth=0)
        role = "screened seed" if seed == 20261119 else "new confirmation seed"
        ax.set_title(f"seed {seed} ({role})")
        ax.set_ylabel("validation soft CE")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(frameon=False)
    ax = axes.flat[3]
    for seed, colour in [(20261119, "#7f7f7f"), (20261120, "#2ca02c"), (20261121, "#9467bd")]:
        original = histories[seed]["original"]
        dropout = histories[seed]["dropout"]
        delta = [soft_ce(d) - soft_ce(o) for o, d in zip(original, dropout)]
        ax.plot([row["epoch"] for row in original], delta, color=colour, label=str(seed))
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvspan(11, 20, color="#777777", alpha=0.08, linewidth=0)
    ax.set_title("dropout − original validation soft CE")
    ax.set_ylabel("CE difference")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(title="seed", frameon=False)
    for ax in axes.flat[:3]:
        ax.set_xlabel("epoch")
    fig.suptitle("DIS soft-target dropout confirmation at lr=3e-4", y=1.03, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    """Parse result paths and save the confirmation curves."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--dropout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    histories = get_histories(root, args.new, args.original, args.dropout)
    plot(histories, root / args.output)


if __name__ == "__main__":
    main()
