"""Draw an expanded, paper-style ResNet-18 architecture review figure.

The figure expands every retained BasicBlock into two convolution boxes and
draws its own shortcut to its own addition node. It uses only the saved
architecture measurements; no images, weights or training are loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
DEFAULT_MEASUREMENTS = ROOT / "architecture_measurements.json"
DEFAULT_OUTPUT = ROOT / "dis_resnet18_expanded_review"


FROZEN = "#dceaf4"
TRAINABLE = "#f8dfc8"
EDGE = "#49677d"
INK = "#253746"
MUTED = "#657786"
PROJECTION = "#b46b22"


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *,
          color: str = INK, linestyle: str = "-", linewidth: float = 0.7,
          mutation_scale: float = 5.5) -> None:
    """Draw a compact directed connection between two diagram coordinates."""

    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=mutation_scale,
                                 linewidth=linewidth, linestyle=linestyle,
                                 color=color, shrinkA=0, shrinkB=0))


def block(ax, x: float, y: float, title: str, projection: bool,
          *, width: float = 2.15) -> None:
    """Draw one BasicBlock with a separate shortcut ending at its plus node."""

    main_y = y + 0.38
    shortcut_y = y + 1.16
    conv_w = 0.62
    conv_h = 0.36
    input_x = x + 0.10
    conv1_x = x + 0.36
    conv2_x = x + 1.12
    plus_x = x + width - 0.10
    shortcut_style = "--" if projection else "-"
    shortcut_color = PROJECTION if projection else EDGE
    first_conv = "3×3\ns=2" if projection else "3×3\ns=1"

    ax.text(x + width / 2, y + 1.63, title, ha="center", va="center",
            fontsize=5.5, weight="bold", color=INK)
    ax.add_patch(Rectangle((conv1_x, main_y - conv_h / 2), conv_w, conv_h,
                           facecolor="#ffffff", edgecolor=EDGE, linewidth=0.55))
    ax.add_patch(Rectangle((conv2_x, main_y - conv_h / 2), conv_w, conv_h,
                           facecolor="#ffffff", edgecolor=EDGE, linewidth=0.55))
    ax.text(conv1_x + conv_w / 2, main_y, first_conv, ha="center", va="center",
            fontsize=4.6, color=INK, linespacing=0.85)
    ax.text(conv2_x + conv_w / 2, main_y, "3×3\ns=1", ha="center", va="center",
            fontsize=4.6, color=INK, linespacing=0.85)

    arrow(ax, (input_x, main_y), (conv1_x - 0.03, main_y))
    arrow(ax, (conv1_x + conv_w + 0.03, main_y), (conv2_x - 0.03, main_y))
    arrow(ax, (conv2_x + conv_w + 0.03, main_y), (plus_x - 0.16, main_y))
    ax.text(plus_x, main_y, "⊕", ha="center", va="center", fontsize=9,
            weight="bold", color=INK)

    # Each shortcut is local to this block and lands at this block's plus node.
    ax.plot([input_x, input_x, x + 0.72],
            [main_y, shortcut_y, shortcut_y], color=shortcut_color,
            linewidth=0.65, linestyle=shortcut_style)
    label = "1×1\ns=2" if projection else "identity"
    label_w = 0.76 if projection else 0.64
    label_x = x + 0.73
    ax.add_patch(Rectangle((label_x, shortcut_y - 0.17), label_w, 0.34,
                           facecolor="#fff8f1" if projection else "#f7f9fa",
                           edgecolor=shortcut_color, linewidth=0.55,
                           linestyle=shortcut_style))
    ax.text(label_x + label_w / 2, shortcut_y, label, ha="center", va="center",
            fontsize=4.25, color=shortcut_color, linespacing=0.85)
    ax.plot([label_x + label_w, plus_x, plus_x],
            [shortcut_y, shortcut_y, main_y + 0.12], color=shortcut_color,
            linewidth=0.65, linestyle=shortcut_style)
    arrow(ax, (label_x + label_w + 0.04, shortcut_y),
          (plus_x - 0.14, shortcut_y), color=shortcut_color,
          linestyle=shortcut_style, linewidth=0.65, mutation_scale=5.2)


def stage(ax, x: float, y: float, name: str, channels: int, spatial: int,
          face: str, first_projection: bool) -> float:
    """Draw a stage bracket, two BasicBlocks and its output shape."""

    width = 4.48
    ax.add_patch(FancyBboxPatch((x, y - 0.08), width, 2.04,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor=face, edgecolor=EDGE, linewidth=0.7,
                                alpha=0.8))
    ax.text(x + 0.12, y + 1.84, name, ha="left", va="center", fontsize=7.0,
            weight="bold", color=INK)
    ax.text(x + width - 0.12, y + 1.84, "BasicBlock ×2", ha="right",
            va="center", fontsize=5.5, color=MUTED)
    ax.text(x + width / 2, y - 0.33, f"{channels}×{spatial}×{spatial}",
            ha="center", va="top", fontsize=5.6, color=INK)
    block(ax, x + 0.10, y, "B1", first_projection)
    block(ax, x + 2.22, y, "B2", False)
    arrow(ax, (x + 2.05, y + 0.38), (x + 2.20, y + 0.38),
          color=EDGE, linewidth=0.55, mutation_scale=4.7)
    return x + width


def draw(measurements: dict, output: Path) -> None:
    """Render the expanded architecture figure to PNG, SVG and PDF."""

    if measurements.get("dummy_input") != [1, 3, 128, 128]:
        raise ValueError("saved measurements do not describe the 128-pixel input")
    stages = measurements["residual_stages"]
    expected = {
        "layer1": (64, 32, False), "layer2": (128, 16, True),
        "layer3": (256, 8, True), "layer4": (512, 4, True),
    }
    for name, (channels, spatial, projection) in expected.items():
        spec = stages[name]
        if (spec["output_channels"], spec["spatial"][0],
                spec.get("first_block_stride", 1)) != (channels, spatial, 2 if projection else 1):
            raise ValueError(f"unexpected saved measurements for {name}")

    fig, ax = plt.subplots(figsize=(7.09, 3.75), dpi=300)
    ax.set_xlim(0, 26.8)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = 4.35
    # Input and stem are deliberately plain modules: the main visual is the
    # expanded two-convolution residual stages.
    ax.add_patch(FancyBboxPatch((0.12, y + 0.08), 1.42, 1.42,
                                boxstyle="round,pad=0.04,rounding_size=0.06",
                                facecolor="#f7f8f9", edgecolor="#8493a0",
                                linewidth=0.75))
    ax.text(0.83, y + 1.16, "Input", ha="center", va="center", fontsize=7.1,
            weight="bold", color=INK)
    ax.text(0.83, y + 0.62, "RGB input", ha="center", va="center", fontsize=5.2,
            color=MUTED)
    ax.text(0.83, y - 0.23, "3×128×128", ha="center", va="top", fontsize=5.7,
            color=INK)
    arrow(ax, (1.60, y + 0.78), (1.88, y + 0.78))

    ax.add_patch(FancyBboxPatch((1.93, y + 0.08), 2.02, 1.42,
                                boxstyle="round,pad=0.04,rounding_size=0.06",
                                facecolor=FROZEN, edgecolor=EDGE, linewidth=0.75))
    ax.text(2.94, y + 1.16, "Stem", ha="center", va="center", fontsize=7.1,
            weight="bold", color=INK)
    ax.text(2.94, y + 0.62, "7×7, s=2\nmaxpool 3×3, s=2",
            ha="center", va="center", fontsize=4.45, color=INK, linespacing=0.9)
    ax.text(2.94, y - 0.23, "64×32×32", ha="center", va="top", fontsize=5.7,
            color=INK)
    arrow(ax, (4.04, y + 0.78), (4.25, y + 0.78))

    stage_x = [4.30, 8.92, 13.54, 18.16]
    stage_specs = [
        ("layer1", 64, 32, FROZEN, False),
        ("layer2", 128, 16, FROZEN, True),
        ("layer3", 256, 8, FROZEN, True),
        ("layer4", 512, 4, TRAINABLE, True),
    ]
    ends = []
    for x, spec in zip(stage_x, stage_specs):
        ends.append(stage(ax, x, y, *spec))
    for left, right in zip(ends[:-1], stage_x[1:]):
        arrow(ax, (left + 0.03, y + 0.78), (right - 0.04, y + 0.78))

    # Compact classifier at the right, with the non-parameter operations kept
    # in one region rather than drawn as narrow standalone boxes.
    cx = 22.82
    ax.add_patch(FancyBboxPatch((cx, y + 0.02), 3.48, 1.55,
                                boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor="#fbfcfd", edgecolor=EDGE, linewidth=0.75))
    ax.text(cx + 0.16, y + 1.29, "Classifier", ha="left", va="center",
            fontsize=7.0, weight="bold", color=INK)
    ax.text(cx + 0.16, y + 0.80, "GAP 512×1×1  →  flatten → 512",
            ha="left", va="center", fontsize=4.85, color=INK)
    ax.text(cx + 0.16, y + 0.35, "Linear 512→3 logits", ha="left",
            va="center", fontsize=5.6, color=INK)
    arrow(ax, (ends[-1] + 0.03, y + 0.78), (cx - 0.06, y + 0.78))

    # Freeze/train bars mirror the actual requires_grad policy and do not cross
    # any labels.
    ax.plot([0.25, 17.95], [8.43, 8.43], color="#9fb6c8", linewidth=1.2)
    ax.text(0.25, 8.62, "frozen: stem + layers 1–3", ha="left", va="bottom",
            fontsize=6.1, color="#496a83")
    ax.plot([18.25, 26.25], [8.43, 8.43], color="#e7b488", linewidth=1.2)
    ax.text(18.25, 8.62, "trainable: layer4 + Linear", ha="left", va="bottom",
            fontsize=6.1, color="#a85b16")

    # The short key makes the line-style convention explicit without adding a
    # second residual drawing below the main network.
    ax.plot([0.35, 1.15], [1.45, 1.45], color=EDGE, linewidth=0.8)
    ax.text(1.28, 1.45, "identity shortcut", ha="left", va="center",
            fontsize=5.7, color=INK)
    ax.plot([4.05, 4.85], [1.45, 1.45], color=PROJECTION, linewidth=0.8,
            linestyle="--")
    ax.text(4.98, 1.45, "projection shortcut (1×1, s=2)", ha="left",
            va="center", fontsize=5.7, color=INK)
    ax.text(11.90, 1.45,
            "layers 2–4 B1: projection; all other shortcuts: identity",
            ha="left", va="center", fontsize=5.45, color=MUTED)
    ax.text(13.45, 0.72,
            "BN/ReLU are omitted from the schematic. Hard and soft supervision use the same network.",
            ha="center", va="center", fontsize=5.7, color=MUTED)

    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.03, top=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), facecolor="white",
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    """Read saved measurements and write the independent review figure."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import json
    measurements = json.loads(args.measurements.read_text())
    draw(measurements, args.output)


if __name__ == "__main__":
    main()
