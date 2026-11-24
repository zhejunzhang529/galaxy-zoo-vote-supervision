"""Draw the compact training diagnostic figure used in the DIS report.

The figure is derived only from the saved 20-epoch diagnostic JSON. It does
not train a model, load images, or score an evaluation pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_runs(path: Path) -> dict[tuple[float, str], dict]:
    """Return saved diagnostic records keyed by learning rate and target."""

    payload = json.loads(path.read_text())
    return {(float(record["learning_rate"]), record["target"]): record for record in payload["records"]}


def draw(input_path: Path, output_stem: Path) -> None:
    """Plot saved hard and soft 3e-4 trajectories with shared metric labels."""

    runs = load_runs(input_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), dpi=260, sharex=True)
    colors = {"target": "#2f6ea5", "train": "#2b9348", "validation": "#c83e3e"}

    for ax, target in zip(axes, ("hard", "soft")):
        record = runs[(3e-4, target)]
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        target_loss = [row["train_loss_mean"] for row in history]
        train_soft = [row["train_eval_soft_cross_entropy"] for row in history]
        validation_soft = [row["validation"]["soft_cross_entropy"] for row in history]
        selected = record["selected_epoch"]
        expected_definition = (
            "hard one-hot cross-entropy"
            if target == "hard"
            else "soft normalized vote cross-entropy"
        )
        definitions = {row["train_loss_definition"] for row in history}
        if definitions != {expected_definition}:
            raise ValueError(
                f"Unexpected {target} target-loss definition: {sorted(definitions)}"
            )

        ax.plot(epochs, target_loss, color=colors["target"], linestyle="-", linewidth=1.45, label="training target loss")
        ax.plot(epochs, train_soft, color=colors["train"], linestyle="--", linewidth=1.45, label="train-evaluation soft CE")
        ax.plot(epochs, validation_soft, color=colors["validation"], linestyle="-.", linewidth=1.55, label="validation soft CE")
        ax.axvline(selected, color="#59636e", linestyle=":", linewidth=1.05, zorder=0)
        if selected > 15:
            ax.text(selected - 0.35, 0.97, f"selected e{selected}", transform=ax.get_xaxis_transform(), fontsize=6.3, color="#59636e", va="top", ha="right")
        else:
            ax.text(selected + 0.35, 0.97, f"selected e{selected}", transform=ax.get_xaxis_transform(), fontsize=6.3, color="#59636e", va="top", ha="left")
        panel = "(a) Hard targets" if target == "hard" else "(b) Soft targets"
        ax.set_title(panel, fontsize=8.2, pad=4)
        ax.set_xlabel("epoch", fontsize=7.2)
        ax.set_xlim(0.5, 20.5)
        ax.grid(True, color="#d9dee3", linewidth=0.45, alpha=0.8)
        ax.tick_params(labelsize=6.5, length=2.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    axes[0].set_ylabel("cross-entropy", fontsize=7.2)
    axes[1].set_ylabel("cross-entropy", fontsize=7.2)
    axes[0].set_xticks([1, 5, 10, 15, 20])
    axes[1].set_xticks([1, 5, 10, 15, 20])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.23, top=0.76, wspace=0.22)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    """Parse paths and write PNG/SVG report figures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    draw(args.input, args.output_stem)


if __name__ == "__main__":
    main()
