"""Measure and draw the retained DIS ResNet-18 reference architecture.

The script uses a dummy tensor only. It does not load image data, train a model,
or access any evaluation pool. The reference network is the existing
``make_model`` implementation with dropout=0.
"""

from __future__ import annotations

import copy
import json
import argparse
from pathlib import Path

import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


ROOT = Path(__file__).resolve().parent


def build_reference_model():
    """Construct the reference three-class model without pretrained downloads."""

    import torch
    from torch import nn
    from torchvision.models import resnet18
    from train_formal import make_model

    base = resnet18(weights=None)
    base.fc = nn.Linear(base.fc.in_features, 3)
    initial_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    return make_model(copy.deepcopy(initial_state))


def measure_model(model) -> dict:
    """Measure tensor shapes and train/eval/freeze behaviour with a dummy input."""

    import torch
    from train_formal import set_train_mode

    names = ["conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    shapes: dict[str, list[int]] = {}
    hooks = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            shapes[name] = list(tensor.shape)

        return hook

    for name in names:
        hooks.append(getattr(model, name).register_forward_hook(capture(name)))
    dummy = torch.zeros(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        logits = model(dummy)
    for hook in hooks:
        hook.remove()

    set_train_mode(model)
    trainable = {
        "stem_conv1": any(parameter.requires_grad for parameter in model.conv1.parameters()),
        "stem_bn1": any(parameter.requires_grad for parameter in model.bn1.parameters()),
        "layer1": any(parameter.requires_grad for parameter in model.layer1.parameters()),
        "layer2": any(parameter.requires_grad for parameter in model.layer2.parameters()),
        "layer3": any(parameter.requires_grad for parameter in model.layer3.parameters()),
        "layer4": any(parameter.requires_grad for parameter in model.layer4.parameters()),
        "classifier": any(parameter.requires_grad for parameter in model.fc.parameters()),
    }
    bn_modes = {
        "bn1_training": bool(model.bn1.training),
        "layer1_block1_bn_training": bool(model.layer1[0].bn1.training),
        "layer4_block1_bn_training": bool(model.layer4[0].bn1.training),
    }
    measurements = {
        "dummy_input": list(dummy.shape),
        "module_output_shapes": shapes,
        "logit_shape": list(logits.shape),
        "residual_stages": {
            "layer1": {"block_type": "BasicBlock", "blocks": 2, "output_channels": 64, "spatial": [32, 32]},
            "layer2": {"block_type": "BasicBlock", "blocks": 2, "output_channels": 128, "spatial": [16, 16], "first_block_stride": 2},
            "layer3": {"block_type": "BasicBlock", "blocks": 2, "output_channels": 256, "spatial": [8, 8], "first_block_stride": 2},
            "layer4": {"block_type": "BasicBlock", "blocks": 2, "output_channels": 512, "spatial": [4, 4], "first_block_stride": 2},
        },
        "trainable_parameters": trainable,
        "batchnorm_modes_after_set_train_mode": bn_modes,
        "reference_dropout": 0.0,
        "output_classes": ["smooth", "features or disk", "star or artifact"],
    }
    return measurements


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    face: str,
    edge: str = "#34495e",
    subtitle_size: float = 6.4,
) -> None:
    """Draw one compact architecture block."""

    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2, facecolor=face, edgecolor=edge)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", fontsize=7.5, weight="bold", color="#1f2933")
    ax.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center", fontsize=subtitle_size, color="#34495e", linespacing=1.15)


def add_arrow(ax, x1: float, x2: float, y: float) -> None:
    """Draw a horizontal directed connection."""

    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#34495e"))


def add_stage(ax, x: float, title: str, channels: int, spatial: int, face: str) -> None:
    """Draw a stage summary without duplicating the residual-block diagram."""

    from matplotlib.patches import FancyBboxPatch

    y, width, height = 4.35, 1.40, 1.80
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2, facecolor=face, edgecolor="#34495e")
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 1.28, title, ha="center", va="center", fontsize=7.4, weight="bold", color="#1f2933")
    ax.text(x + width / 2, y + 0.43, f"BasicBlock\n×2\n{channels}×{spatial}×{spatial}", ha="center", va="center", fontsize=5.7, color="#34495e", linespacing=1.1)


def add_residual_example(ax, x: float, y: float, width: float, height: float, title: str, projection: bool) -> None:
    """Draw an isolated identity or downsampling BasicBlock schematic."""

    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    panel = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=0.9, facecolor="#fbfcfd", edgecolor="#9fb0bb")
    ax.add_patch(panel)
    ax.text(x + width / 2, y + height - 0.27, title, ha="center", va="center", fontsize=6.8, weight="bold", color="#1f2933")

    main_y = y + 1.05
    shortcut_y = y + 1.70
    first = x + 0.38
    conv1_x, conv2_x, plus_x = x + 1.15, x + 2.38, x + 3.82
    box_w, box_h = 0.82, 0.50
    first_stride = 2 if projection else 1
    ax.add_patch(FancyArrowPatch((first, main_y), (conv1_x, main_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color="#34495e"))
    ax.add_patch(Rectangle((conv1_x, main_y - box_h / 2), box_w, box_h, facecolor="#ffffff", edgecolor="#34495e", linewidth=0.7))
    ax.text(conv1_x + box_w / 2, main_y, f"3×3\ns={first_stride}", ha="center", va="center", fontsize=5.1, linespacing=1.0)
    ax.add_patch(FancyArrowPatch((conv1_x + box_w, main_y), (conv2_x, main_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color="#34495e"))
    ax.add_patch(Rectangle((conv2_x, main_y - box_h / 2), box_w, box_h, facecolor="#ffffff", edgecolor="#34495e", linewidth=0.7))
    ax.text(conv2_x + box_w / 2, main_y, "3×3\ns=1", ha="center", va="center", fontsize=5.1, linespacing=1.0)
    ax.add_patch(FancyArrowPatch((conv2_x + box_w, main_y), (plus_x - 0.14, main_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color="#34495e"))
    ax.add_patch(Circle((plus_x, main_y), 0.12, facecolor="#ffffff", edgecolor="#34495e", linewidth=0.8))
    ax.text(plus_x, main_y, "+", ha="center", va="center", fontsize=8, weight="bold")
    ax.add_patch(FancyArrowPatch((plus_x + 0.14, main_y), (x + width - 0.35, main_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color="#34495e"))
    ax.text(x + width - 0.27, main_y, "out", ha="center", va="center", fontsize=5.5, color="#52606d")

    ax.plot([first, first, x + 0.72], [main_y + 0.12, shortcut_y, shortcut_y], color="#7f8c8d", linewidth=1.0)
    if projection:
        proj_x = x + 1.00
        ax.add_patch(FancyArrowPatch((x + 0.72, shortcut_y), (proj_x, shortcut_y), arrowstyle="-|>", mutation_scale=6, linewidth=0.8, color="#7f8c8d"))
        projection_w = 1.15
        ax.add_patch(Rectangle((proj_x, shortcut_y - 0.27), projection_w, 0.54, facecolor="#ffffff", edgecolor="#7f8c8d", linewidth=0.7))
        ax.text(proj_x + projection_w / 2, shortcut_y, "1×1 projection\ns=2", ha="center", va="center", fontsize=4.9, color="#52606d", linespacing=1.0)
        ax.add_patch(FancyArrowPatch((proj_x + projection_w, shortcut_y), (plus_x, shortcut_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color="#7f8c8d"))
        ax.add_patch(FancyArrowPatch((plus_x, shortcut_y), (plus_x, main_y + 0.17), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#7f8c8d"))
        ax.text(x + width / 2, y + 0.28, "first block of layers 2–4", ha="center", va="center", fontsize=5.5, color="#52606d")
    else:
        ax.add_patch(FancyArrowPatch((x + 0.72, shortcut_y), (plus_x, shortcut_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color="#7f8c8d"))
        ax.add_patch(FancyArrowPatch((plus_x, shortcut_y), (plus_x, main_y + 0.17), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color="#7f8c8d"))
        ax.text(x + width / 2, shortcut_y + 0.10, "identity shortcut", ha="center", va="bottom", fontsize=5.5, color="#52606d")
        ax.text(x + width / 2, y + 0.28, "remaining blocks", ha="center", va="center", fontsize=5.5, color="#52606d")


def draw(measurements: dict) -> None:
    """Create the publication-style architecture diagram as SVG and PNG."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    # Designed for approximately 17--18 cm placement in an A4 report.
    fig, ax = plt.subplots(figsize=(7.1, 4.0), dpi=240)
    ax.set_xlim(0, 16.3)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    frozen = "#e8f0f6"
    tuned = "#fce8d5"
    neutral = "#f4f5f6"
    ax.text(8.1, 7.95, "ResNet-18 architecture", ha="center", va="center", fontsize=11, weight="bold", color="#1f2933")
    ax.text(8.1, 7.67, "RGB preprocessing: resize to 128×128, scale by 1/255, then ImageNet normalization", ha="center", va="center", fontsize=6.1, color="#52606d")
    ax.text(5.05, 7.25, "FROZEN", ha="center", va="center", fontsize=7.1, weight="bold", color="#496a83")
    ax.text(10.65, 7.25, "FINE-TUNED", ha="center", va="center", fontsize=7.1, weight="bold", color="#a85b16")
    ax.plot([1.95, 8.25], [7.08, 7.08], color="#9fb6c8", linewidth=0.8)
    ax.plot([8.50, 14.05], [7.08, 7.08], color="#e7b488", linewidth=0.8)

    main_y = 5.25
    add_box(ax, 0.10, 4.35, 1.60, 1.80, "Input", "3×128×128", neutral, subtitle_size=5.9)
    add_box(ax, 1.95, 4.35, 1.45, 1.80, "Stem", "7×7 conv\nmax-pool\n64×32×32", frozen, subtitle_size=5.9)
    add_stage(ax, 3.65, "layer1", 64, 32, frozen)
    add_stage(ax, 5.30, "layer2", 128, 16, frozen)
    add_stage(ax, 6.95, "layer3", 256, 8, frozen)
    add_stage(ax, 8.60, "layer4", 512, 4, tuned)
    add_box(ax, 10.25, 4.35, 1.25, 1.80, "GAP", "512×1×1", neutral)
    add_box(ax, 11.75, 4.35, 1.20, 1.80, "Flatten", "512", neutral)
    add_box(ax, 13.20, 4.35, 1.30, 1.80, "Linear", "512 → 3", tuned)
    add_box(ax, 14.75, 4.35, 1.45, 1.80, "Softmax", "3\nprobabilities", neutral, subtitle_size=5.9)

    for x1, x2 in [(1.70, 1.95), (3.40, 3.65), (5.05, 5.30), (6.70, 6.95), (8.35, 8.60), (10.00, 10.25), (11.50, 11.75), (12.95, 13.20), (14.50, 14.75)]:
        add_arrow(ax, x1, x2, main_y)

    ax.text(8.10, 3.95, "Output classes: smooth  ·  features or disk  ·  star or artifact", ha="center", va="center", fontsize=6.6, color="#1f2933")
    ax.text(8.10, 3.66, "Evaluation applies softmax to the three logits to obtain probabilities.", ha="center", va="center", fontsize=6.0, color="#52606d")

    add_residual_example(ax, 0.25, 0.80, 6.15, 2.45, "Identity BasicBlock", projection=False)
    add_residual_example(ax, 6.65, 0.80, 6.15, 2.45, "Downsampling BasicBlock", projection=True)
    ax.text(8.10, 0.44, "The first block of layers 2–4 downsamples; all other shortcuts are identities.", ha="center", va="center", fontsize=5.9, color="#52606d")
    ax.text(8.10, 0.18, "BN/ReLU omitted. The same architecture is used for hard and soft targets; reference classifier dropout=0.", ha="center", va="center", fontsize=5.7, color="#52606d")

    output = ROOT / "dis_resnet18_architecture.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(ROOT / "dis_resnet18_architecture.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Run the dummy measurement, draw from saved measurements, or do both."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-only", action="store_true", help="run the dummy-input check and save JSON")
    parser.add_argument("--draw-only", action="store_true", help="draw from the saved JSON without importing PyTorch")
    args = parser.parse_args()
    if args.measure_only and args.draw_only:
        parser.error("--measure-only and --draw-only are mutually exclusive")

    measurement_path = ROOT / "architecture_measurements.json"
    if not args.draw_only:
        model = build_reference_model()
        measurements = measure_model(model)
        measurement_path.write_text(json.dumps(measurements, indent=2) + "\n")
        print(json.dumps(measurements, indent=2))
        if args.measure_only:
            return
    else:
        measurements = json.loads(measurement_path.read_text())
    draw(measurements)


if __name__ == "__main__":
    main()
