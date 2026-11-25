"""Draw a compact ResNet-18 diagram using the supplied reference layout.

The diagram is intentionally a new independent review asset. It expands the
main path into narrow convolution bars and draws one local shortcut per
BasicBlock; it does not modify the published report figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parent
MEASUREMENTS = ROOT / "architecture_measurements.json"
OUTPUT = ROOT / "dis_resnet18_reference_layout_review"

INK = "#263746"
EDGE = "#536f82"
MUTED = "#718392"
FROZEN = "#b9cedb"
TRAINABLE = "#dfb58c"
PROJECTION = "#b56b25"
STAGE_COLOURS = ["#dbe9f1", "#d9e6ef", "#e8e7f1", "#f1e6c8"]


def main_arrow(ax, x0: float, x1: float, y: float, *, color: str = INK) -> None:
    """Draw one main-path arrow between adjacent operators."""

    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=0.65,
                                 color=color, shrinkA=0, shrinkB=0))


def shortcut(ax, x0: float, x1: float, y: float, *, projection: bool,
             label: str | None = None) -> None:
    """Draw a single curved shortcut from a block input to its plus node."""

    colour = PROJECTION if projection else EDGE
    linestyle = "--" if projection else "-"
    curve_y = y + 1.70
    path = MplPath(
        [(x0, y), (x0 + 0.15, curve_y), (x1 - 0.15, curve_y), (x1, y)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, fill=False, linewidth=0.75,
                           linestyle=linestyle, edgecolor=colour))
    # A small arrowhead sits on the final part of the curve, pointing into ⊕.
    ax.add_patch(FancyArrowPatch((x1 - 0.25, y + 0.13), (x1 - 0.02, y + 0.02),
                                 arrowstyle="-|>", mutation_scale=5.5,
                                 linewidth=0.7, linestyle=linestyle,
                                 color=colour, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, curve_y + 0.10, label, ha="center", va="bottom",
                fontsize=4.8, color=colour)


def conv_bar(ax, x: float, y: float, label: str, colour: str,
             *, height: float = 1.42) -> None:
    """Draw a narrow vertical convolution bar with rotated notation."""

    ax.add_patch(Rectangle((x, y - height / 2), 0.42, height,
                           facecolor=colour, edgecolor=EDGE, linewidth=0.55))
    ax.text(x + 0.21, y, label, ha="center", va="center", rotation=90,
            fontsize=4.75, color=INK)


def plus_node(ax, x: float, y: float) -> None:
    """Draw the addition node where the local shortcut terminates."""

    ax.add_patch(Circle((x, y), radius=0.13, facecolor="white",
                        edgecolor=INK, linewidth=0.75))
    ax.text(x, y - 0.01, "⊕", ha="center", va="center", fontsize=7.4,
            weight="bold", color=INK)


def stage(ax, x: float, y: float, name: str, channels: int, spatial: int,
          colour: str, projection_first: bool) -> tuple[float, float]:
    """Draw two local BasicBlocks and the bottom stage grouping mark."""

    block_width = 2.05
    conv_offsets = (0.43, 1.06)
    block_starts = (x, x + 2.35)
    for index, start in enumerate(block_starts):
        first_stride = "3×3, s=2" if projection_first and index == 0 else "3×3, s=1"
        conv_bar(ax, start + conv_offsets[0], y, first_stride, colour)
        conv_bar(ax, start + conv_offsets[1], y, "3×3, s=1", colour)
        input_x = start + 0.18
        plus_x = start + 1.75
        main_arrow(ax, input_x, start + conv_offsets[0] - 0.04, y)
        main_arrow(ax, start + conv_offsets[0] + 0.46,
                   start + conv_offsets[1] - 0.04, y)
        main_arrow(ax, start + conv_offsets[1] + 0.46, plus_x - 0.14, y)
        plus_node(ax, plus_x, y)
        shortcut(ax, input_x, plus_x, y,
                 projection=projection_first and index == 0,
                 label="1×1, s=2" if projection_first and index == 0 else None)
        if index == 0:
            ax.text(start + 0.93, y - 1.03, "identity" if not projection_first else "projection",
                    ha="center", va="top", fontsize=4.4,
                    color=EDGE if not projection_first else PROJECTION)
    # A small arrow joins the two blocks, without enclosing them in a card.
    main_arrow(ax, block_starts[0] + block_width - 0.02,
               block_starts[1] + 0.10, y)
    group_left = x - 0.12
    group_right = x + 4.18
    bracket_y = 2.00
    ax.plot([group_left, group_left, group_right, group_right],
            [bracket_y + 0.16, bracket_y, bracket_y, bracket_y + 0.16],
            color=colour, linewidth=0.8)
    ax.text((group_left + group_right) / 2, bracket_y - 0.16,
            f"{name}\n2 BasicBlocks\n{channels}×{spatial}×{spatial}",
            ha="center", va="top", fontsize=4.75, color=INK,
            linespacing=0.92)
    return group_left, group_right


def draw(measurements: dict, output: Path) -> None:
    """Validate saved shapes and render the reference-layout architecture."""

    if measurements.get("dummy_input") != [1, 3, 128, 128]:
        raise ValueError("unexpected dummy input")
    expected = {
        "layer1": (64, 32, 1), "layer2": (128, 16, 2),
        "layer3": (256, 8, 2), "layer4": (512, 4, 2),
    }
    for name, (channels, spatial, stride) in expected.items():
        spec = measurements["residual_stages"][name]
        if spec["output_channels"] != channels or spec["spatial"][0] != spatial:
            raise ValueError(f"unexpected output shape for {name}")
        if spec.get("first_block_stride", 1) != stride:
            raise ValueError(f"unexpected first-block stride for {name}")

    fig, ax = plt.subplots(figsize=(7.09, 3.55), dpi=300)
    ax.set_xlim(0, 26.25)
    ax.set_ylim(0.55, 7.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = 4.35
    # Input and stem are operator bars, not information cards.
    conv_bar(ax, 0.70, y, "RGB\n3×128×128", "#eef1f2", height=1.42)
    ax.text(0.91, y - 1.00, "Input\n3×128×128", ha="center", va="top",
            fontsize=4.9, color=INK, linespacing=0.9)
    main_arrow(ax, 1.18, 1.52, y)
    conv_bar(ax, 1.58, y, "7×7, s=2", FROZEN)
    main_arrow(ax, 2.08, 2.33, y)
    conv_bar(ax, 2.38, y, "maxpool\n3×3, s=2", FROZEN, height=1.42)
    ax.text(2.59, y - 1.00, "Stem\n64×32×32", ha="center", va="top",
            fontsize=4.9, color=INK, linespacing=0.9)
    main_arrow(ax, 2.90, 3.35, y)

    positions = [3.45, 7.95, 12.45, 16.95]
    specs = [
        ("layer1", 64, 32, STAGE_COLOURS[0], False),
        ("layer2", 128, 16, STAGE_COLOURS[1], True),
        ("layer3", 256, 8, STAGE_COLOURS[2], True),
        ("layer4", 512, 4, STAGE_COLOURS[3], True),
    ]
    groups = []
    for x, spec in zip(positions, specs):
        groups.append(stage(ax, x, y, *spec))
    for left, right in zip(groups[:-1], groups[1:]):
        main_arrow(ax, left[1] + 0.02, right[0] - 0.02, y)

    # GAP and classifier are kept as simple terminal operators.
    main_arrow(ax, groups[-1][1] + 0.02, 21.62, y)
    conv_bar(ax, 21.68, y, "GAP\n512×1×1", "#eef1f2", height=1.42)
    main_arrow(ax, 22.18, 22.72, y)
    conv_bar(ax, 22.78, y, "flatten\n512", "#eef1f2", height=1.42)
    main_arrow(ax, 23.28, 23.84, y)
    conv_bar(ax, 23.90, y, "Linear\n512→3", TRAINABLE, height=1.42)
    ax.text(24.11, y - 1.00, "3 logits", ha="center", va="top", fontsize=5.5,
            color=INK)

    # Freeze/tune ranges are compact brackets above the network.
    ax.plot([1.35, 16.72], [6.72, 6.72], color=FROZEN, linewidth=1.0)
    ax.text(1.35, 6.86, "frozen: Stem + layers 1–3", ha="left", va="bottom",
            fontsize=5.8, color=EDGE)
    ax.plot([17.18, 24.35], [6.72, 6.72], color=TRAINABLE, linewidth=1.0)
    ax.text(17.18, 6.86, "trainable: layer4 + Linear", ha="left", va="bottom",
            fontsize=5.8, color=PROJECTION)

    # A small line-style key replaces repeated explanatory prose.
    ax.plot([0.65, 1.28], [0.70, 0.70], color=EDGE, linewidth=0.8)
    ax.text(1.40, 0.70, "identity shortcut", ha="left", va="center",
            fontsize=5.4, color=INK)
    ax.plot([4.35, 4.98], [0.70, 0.70], color=PROJECTION, linewidth=0.8,
            linestyle="--")
    ax.text(5.10, 0.70, "projection shortcut: 1×1, stride 2", ha="left",
            va="center", fontsize=5.4, color=INK)
    ax.text(15.0, 0.70,
            "BN/ReLU omitted · hard and soft use the same network",
            ha="center", va="center", fontsize=5.4, color=MUTED)

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.04, top=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), facecolor="white",
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    """Read the saved structural measurements and write the new review asset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    draw(json.loads(args.measurements.read_text()), args.output)


if __name__ == "__main__":
    main()
