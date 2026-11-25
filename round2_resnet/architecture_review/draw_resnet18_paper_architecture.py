"""Draw a compact paper-style schematic of the retained DIS ResNet-18.

The diagram is generated from the saved architecture measurement JSON. It does
not construct a model, load images, train, or run inference. Feature-map stacks
show the changing spatial scale; a separate residual key keeps the shortcut
mechanism explicit without turning every stage into a flowchart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent


def load_measurements(path: Path) -> dict:
    """Load the previously measured architecture specification."""

    payload = json.loads(path.read_text())
    required = {"dummy_input", "residual_stages", "logit_shape", "output_classes"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"architecture measurement is missing: {sorted(missing)}")
    return payload


def arrow(ax, x1: float, x2: float, y: float, color: str = "#34495e") -> None:
    """Draw one restrained horizontal connection between network components."""

    ax.add_patch(
        FancyArrowPatch(
            (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=color, shrinkA=2, shrinkB=2,
        )
    )


def feature_stack(
    ax, x: float, y: float, width: float, height: float, label: str,
    subtitle: str, face: str, edge: str, depth: int = 4,
) -> None:
    """Draw a stacked feature-map glyph with a label below it."""

    for index in range(depth - 1, -1, -1):
        dx = index * 0.075
        dy = index * 0.075
        ax.add_patch(
            Rectangle(
                (x + dx, y + dy), width, height,
                facecolor=face, edgecolor=edge, linewidth=0.9,
                zorder=2 + index,
            )
        )
    ax.text(x + width / 2 + 0.11, y + height + 0.27, label,
            ha="center", va="bottom", fontsize=8.0, weight="bold", color="#1f2933")
    ax.text(x + width / 2 + 0.11, y - 0.24, subtitle,
            ha="center", va="top", fontsize=6.7, color="#34495e", linespacing=1.05)


def classifier_region(
    ax, x: float, y: float, frozen: str, tuned: str, compact: bool = False
) -> None:
    """Draw either the report classifier summary or the full classifier path."""

    if compact:
        width, height = 3.95, 1.62
    else:
        width, height = 4.10, 1.70
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#f7f8f9", edgecolor="#8493a0", linewidth=1.0,
    ))
    ax.text(x + 0.20, y + height - 0.28,
            "classifier - 3 logits" if not compact else "classifier",
            ha="left", va="center",
            fontsize=7.5, weight="bold", color="#1f2933")
    if compact:
        labels = [("GAP", "512×1×1"), ("Linear", "3 logits")]
        positions = [x + 0.24, x + 2.54]
        widths = [1.02, 1.16]
        for (title, sub), px, box_width in zip(labels, positions, widths):
            face = tuned if title == "Linear" else "#eef1f3"
            ax.add_patch(Rectangle((px, y + 0.36), box_width, 0.66,
                                   facecolor=face, edgecolor="#697987", linewidth=0.8))
            ax.text(px + box_width / 2, y + 0.77, title, ha="center", va="center",
                    fontsize=6.6, weight="bold", color="#1f2933")
            ax.text(px + box_width / 2, y + 0.53, sub, ha="center", va="center",
                    fontsize=5.8, color="#34495e")
        arrow(ax, x + 1.33, x + 2.47, y + 0.69, "#697987")
        ax.text(x + 1.90, y + 1.20, "flatten → 512-dimensional vector",
                ha="center", va="center", fontsize=5.5, color="#52606d")
        return

    labels = [("GAP", "512×1×1"), ("flatten", "512"), ("Linear", "512 → 3")]
    positions = [x + 0.24, x + 1.55, x + 2.86]
    for (title, sub), px in zip(labels, positions):
        face = tuned if title == "Linear" else "#eef1f3"
        ax.add_patch(Rectangle((px, y + 0.39), 0.88, 0.66,
                               facecolor=face, edgecolor="#697987", linewidth=0.8))
        ax.text(px + 0.44, y + 0.79, title, ha="center", va="center",
                fontsize=6.6, weight="bold", color="#1f2933")
        ax.text(px + 0.44, y + 0.54, sub, ha="center", va="center",
                fontsize=5.8, color="#34495e")
    arrow(ax, x + 1.18, x + 1.50, y + 0.72, "#697987")
    arrow(ax, x + 2.49, x + 2.81, y + 0.72, "#697987")
    ax.text(x + width / 2, y + 0.12, "softmax at evaluation → 3 probabilities",
            ha="center", va="center", fontsize=6.1, color="#52606d")


def residual_block_example(ax, x: float, y: float, title: str, projection: bool) -> None:
    """Draw one residual block with a fully connected shortcut path."""

    ink = "#52606d"
    accent = "#a85b16" if projection else ink
    plus_x = x + 3.95
    input_x = x + 0.25
    main_y = y + 0.78
    shortcut_y = y + 1.72
    ax.text(x, y + 2.22, title, ha="left", va="center",
            fontsize=6.9, weight="bold", color="#1f2933")

    # Main branch: the downsampling case has stride 2 only on its first 3x3.
    ax.plot([input_x, input_x + 0.35], [main_y, main_y], color=ink, linewidth=0.9)
    first_label = "3×3, s=2" if projection else "3×3, s=1"
    second_label = "3×3, s=1"
    for offset, label in ((0.43, first_label), (2.00, second_label)):
        ax.add_patch(Rectangle((x + offset, main_y - 0.22), 1.02, 0.44,
                               facecolor="#ffffff", edgecolor=ink, linewidth=0.7))
        ax.text(x + offset + 0.51, main_y, label, ha="center", va="center",
                fontsize=5.3, color="#1f2933")
    arrow(ax, x + 1.48, x + 1.95, main_y, ink)
    arrow(ax, x + 3.05, plus_x - 0.16, main_y, ink)
    ax.text(plus_x, main_y, "⊕", ha="center", va="center", fontsize=12,
            weight="bold", color="#1f2933")

    # A separate shortcut path starts at the same input and ends at this block's
    # own addition node; it is never drawn as a second path into another key.
    shortcut_style = "--" if projection else "-"
    ax.plot([input_x, input_x, x + 1.42], [main_y + 0.03, shortcut_y, shortcut_y],
            color=accent, linewidth=0.9, linestyle=shortcut_style)
    if projection:
        ax.add_patch(Rectangle((x + 1.43, shortcut_y - 0.22), 1.35, 0.44,
                               facecolor="#fff8f1", edgecolor=accent, linewidth=0.7))
        ax.text(x + 2.105, shortcut_y, "1×1, s=2", ha="center", va="center",
                fontsize=5.4, color="#8b4b12")
        ax.plot([x + 2.78, plus_x, plus_x], [shortcut_y, shortcut_y, main_y + 0.17],
                color=accent, linewidth=0.9, linestyle=shortcut_style)
        ax.add_patch(FancyArrowPatch((x + 2.84, shortcut_y), (plus_x - 0.12, shortcut_y),
                                     arrowstyle="-|>", mutation_scale=7, linewidth=0.9,
                                     linestyle=shortcut_style, color=accent))
        ax.text(x + 0.12, y + 1.98, "projection shortcut", ha="left", va="center",
                fontsize=5.55, color="#8b4b12")
    else:
        ax.add_patch(Rectangle((x + 1.43, shortcut_y - 0.22), 1.08, 0.44,
                               facecolor="#f7f9fa", edgecolor=accent, linewidth=0.7))
        ax.text(x + 1.97, shortcut_y, "identity", ha="center", va="center",
                fontsize=5.4, color=ink)
        ax.plot([x + 2.51, plus_x, plus_x], [shortcut_y, shortcut_y, main_y + 0.17],
                color=accent, linewidth=0.9, linestyle=shortcut_style)
        ax.add_patch(FancyArrowPatch((x + 2.57, shortcut_y), (plus_x - 0.12, shortcut_y),
                                     arrowstyle="-|>", mutation_scale=7, linewidth=0.9,
                                     linestyle=shortcut_style, color=accent))
        ax.text(x + 0.12, y + 1.98, "identity shortcut", ha="left", va="center",
                fontsize=5.55, color=ink)


def residual_key(ax, x: float, y: float) -> None:
    """Draw two independent residual-block examples with separate sums."""

    ax.text(x, y + 2.75, "Residual block key", ha="left", va="center",
            fontsize=7.4, weight="bold", color="#1f2933")
    residual_block_example(ax, x, y, "(a) identity shortcut", projection=False)
    residual_block_example(ax, x + 8.75, y, "(b) downsampling shortcut", projection=True)


def compact_residual_block(ax, x: float, y: float, title: str, projection: bool) -> None:
    """Draw one small, independent block with its shortcut ending at ⊕."""

    ink = "#52606d"
    accent = "#a85b16" if projection else ink
    style = "--" if projection else "-"
    main_y = y + 0.42
    shortcut_y = y + 1.15
    plus_x = x + 5.35
    box_w = 1.55
    box_h = 0.30
    ax.text(x, y + 1.58, title, ha="left", va="center", fontsize=5.8,
            weight="bold", color="#1f2933")
    first = "3×3, s=2" if projection else "3×3, s=1"
    for offset, label in ((0.55, first), (2.65, "3×3, s=1")):
        ax.add_patch(Rectangle((x + offset, main_y - box_h / 2), box_w, box_h,
                               facecolor="#ffffff", edgecolor=ink, linewidth=0.65))
        ax.text(x + offset + box_w / 2, main_y, label, ha="center", va="center",
                fontsize=4.8, color="#1f2933")
    for start, end in ((0.18, 0.52), (2.12, 2.62), (4.22, plus_x - x - 0.14)):
        ax.add_patch(FancyArrowPatch((x + start, main_y), (x + end, main_y),
                                     arrowstyle="-|>", mutation_scale=5,
                                     linewidth=0.65, color=ink))
    ax.text(plus_x, main_y, "⊕", ha="center", va="center", fontsize=9,
            weight="bold", color="#1f2933")
    ax.plot([x + 0.18, x + 0.18, x + 1.25], [main_y, shortcut_y, shortcut_y],
            color=accent, linewidth=0.75, linestyle=style)
    shortcut_label = "1×1, stride 2" if projection else "identity"
    shortcut_w = 1.58 if projection else 1.05
    shortcut_x = x + 1.28
    ax.add_patch(Rectangle((shortcut_x, shortcut_y - box_h / 2), shortcut_w, box_h,
                           facecolor="#fff8f1" if projection else "#f7f9fa",
                           edgecolor=accent, linewidth=0.65))
    ax.text(shortcut_x + shortcut_w / 2, shortcut_y, shortcut_label, ha="center",
            va="center", fontsize=4.8, color=accent)
    ax.plot([shortcut_x + shortcut_w, plus_x, plus_x],
            [shortcut_y, shortcut_y, main_y + 0.10], color=accent,
            linewidth=0.75, linestyle=style)
    ax.add_patch(FancyArrowPatch((shortcut_x + shortcut_w + 0.04, shortcut_y),
                                 (plus_x - 0.14, shortcut_y), arrowstyle="-|>",
                                 mutation_scale=5, linewidth=0.75,
                                 linestyle=style, color=accent))


def compact_residual_key(ax, x: float, y: float) -> None:
    """Draw the two independent shortcut examples used in the report figure."""

    ax.text(x + 9.45, y + 1.92, "Representative residual blocks",
            ha="center", va="center", fontsize=6.2, weight="bold", color="#1f2933")
    compact_residual_block(ax, x, y, "(a) identity shortcut", projection=False)
    compact_residual_block(ax, x + 9.0, y, "(b) downsampling shortcut", projection=True)
    ax.text(x + 9.45, y - 0.02,
            "Identity: solid shortcut   |   projection: dashed shortcut; BN/ReLU omitted",
            ha="center", va="center", fontsize=4.9, color="#52606d")


def draw(
    measurements: dict,
    output_stem: Path,
    include_residual_key: bool = True,
    report_residual_key: bool = False,
    tight: bool = False,
) -> None:
    """Draw the architecture schematic as publication-style SVG and PNG."""

    stages = measurements["residual_stages"]
    if measurements["dummy_input"] != [1, 3, 128, 128]:
        raise ValueError("unexpected reference input shape")
    if measurements["logit_shape"] != [1, 3]:
        raise ValueError("unexpected classifier output shape")

    fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=240)
    ax.set_xlim(0, 19.0)
    ax.set_ylim(0, 8.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    frozen = "#dceaf4"
    tuned = "#f8dfc8"
    edge = "#49677d"
    ax.text(0.18, 7.88, "ResNet-18", ha="left", va="center",
            fontsize=9.5, weight="bold", color="#1f2933")
    ax.text(0.18, 7.50, "frozen feature extractor", ha="left", va="center",
            fontsize=6.8, color="#496a83")
    ax.plot([1.55, 9.65], [7.24, 7.24], color="#9fb6c8", linewidth=1.2)
    ax.text(9.98, 7.50, "trainable: layer4 + Linear", ha="left", va="center",
            fontsize=6.8, color="#a85b16")
    ax.plot([9.85, 18.65], [7.24, 7.24], color="#e7b488", linewidth=1.2)

    base_y = 4.95
    feature_stack(ax, 0.35, base_y, 0.72, 1.25, "input", "3×128×128", "#f7f8f9", "#8493a0")
    arrow(ax, 1.22, 1.55, base_y + 0.62)
    feature_stack(ax, 1.58, base_y + 0.05, 0.78, 1.15, "stem", "64×32×32", frozen, edge)
    ax.text(1.97, 4.35, "7×7,s2 + maxpool", ha="center", va="top", fontsize=5.8, color="#34495e")

    stage_x = [3.20, 5.55, 7.90, 10.25]
    stage_info = [
        ("layer1", stages["layer1"], 1.25, frozen),
        ("layer2", stages["layer2"], 1.08, frozen),
        ("layer3", stages["layer3"], 0.91, frozen),
        ("layer4", stages["layer4"], 0.74, tuned),
    ]
    previous = 2.42
    for x, (name, spec, scale, face) in zip(stage_x, stage_info):
        arrow(ax, previous, x - 0.12, base_y + 0.62)
        spatial = spec["spatial"][0]
        channels = spec["output_channels"]
        feature_stack(ax, x, base_y + (1.25 - scale) * 0.25, 0.82, scale,
                      name, f"BasicBlock×2\n{channels}×{spatial}×{spatial}", face, edge)
        previous = x + 1.04

    classifier_region(ax, 13.05, 4.70, frozen, tuned, compact=not include_residual_key)
    arrow(ax, previous, 13.00, base_y + 0.76)

    if include_residual_key:
        residual_key(ax, 0.45, 1.05)
        ax.text(9.50, 0.34, "The first block of layers 2–4 uses a stride-2 projection shortcut; all other shortcuts are identities.",
                ha="center", va="center", fontsize=5.9, color="#52606d")
    elif report_residual_key:
        compact_residual_key(ax, 0.45, 2.15)
    else:
        ax.text(9.50, 3.62, "First block of layers 2–4: stride-2 projection shortcut; other shortcuts: identity.",
                ha="center", va="center", fontsize=6.3, color="#52606d")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(output_stem.with_suffix(".png"), facecolor="white", bbox_inches=bbox, pad_inches=0.08 if tight else 0)
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white", bbox_inches=bbox, pad_inches=0.08 if tight else 0)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white", bbox_inches=bbox, pad_inches=0.08 if tight else 0)
    plt.close(fig)


def crop_report_raster(output_stem: Path) -> None:
    """Crop the simplified report PNG and write a tightly framed PDF copy."""

    png_path = output_stem.with_suffix(".png")
    image = Image.open(png_path).convert("RGB")
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    bbox = difference.getbbox()
    if bbox is None:
        raise ValueError("report architecture raster is blank")
    pad = 18
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(image.width, bbox[2] + pad)
    bottom = min(image.height, bbox[3] + pad)
    cropped = image.crop((left, top, right, bottom))
    cropped.save(png_path)

    fig, ax = plt.subplots(figsize=(cropped.width / 240, cropped.height / 240), dpi=240)
    ax.imshow(cropped)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white", pad_inches=0)
    plt.close(fig)


def main() -> None:
    """Read the saved architecture measurements and write the diagram."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path,
                        default=ROOT / "architecture_measurements.json")
    parser.add_argument("--output-stem", type=Path,
                        default=ROOT / "dis_resnet18_paper_architecture")
    parser.add_argument("--report-output-stem", type=Path, default=None,
                        help="Optional simplified, tightly cropped report asset stem.")
    args = parser.parse_args()
    measurements = load_measurements(args.measurements)
    draw(measurements, args.output_stem, include_residual_key=True, tight=False)
    if args.report_output_stem is not None:
        draw(measurements, args.report_output_stem, include_residual_key=False,
             report_residual_key=True, tight=True)
        crop_report_raster(args.report_output_stem)


if __name__ == "__main__":
    main()
