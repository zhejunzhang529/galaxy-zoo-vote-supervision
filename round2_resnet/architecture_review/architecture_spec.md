# DIS ResNet-18 architecture specification

This specification records the retained DIS reference model used for the
architecture diagram. It was checked from the implementation and with a
dummy tensor; no image, checkpoint, training run, or independent evaluation
pool was loaded.

## Reference implementation

- Model construction and freezing: `round2_resnet/src/train_formal.py:35-54`.
- Training-mode policy for frozen early BatchNorm layers:
  `round2_resnet/src/train_formal.py:57-64`.
- Image preprocessing: `round2_resnet/src/train_formal.py:66-78`.
- Reference input and normalization: `round2_resnet/configs/formal.json:2-16`.
- The retained model is the standard torchvision `resnet18`, with
  `fc = Linear(512, 3)` and no dropout. The training targets are the only
  difference between hard and soft supervision; the network is shared.

The preprocessing used by the reference protocol is RGB conversion, bilinear
resize to `128×128`, division by 255, and ImageNet channel normalization
with mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

## Verified tensor path

The script `draw_resnet18_architecture.py` registered forward hooks and ran a
zero-valued dummy tensor of shape `[1, 3, 128, 128]`. The measured output
shapes are saved in `architecture_measurements.json`:

| component | verified output | structure |
|---|---:|---|
| input | `3×128×128` | RGB image |
| stem | `64×32×32` | `7×7` convolution, stride 2, followed by `3×3` max-pool, stride 2 |
| `layer1` | `64×32×32` | 2 `BasicBlock`s |
| `layer2` | `128×16×16` | 2 `BasicBlock`s; first block downsamples with stride 2 |
| `layer3` | `256×8×8` | 2 `BasicBlock`s; first block downsamples with stride 2 |
| `layer4` | `512×4×4` | 2 `BasicBlock`s; first block downsamples with stride 2 |
| global average pool | `512×1×1` | adaptive average pooling |
| flatten | `512` | flattening the `512×1×1` tensor |
| classifier | 3 logits | `Linear(512, 3)` |
| probability output | 3 probabilities | evaluation applies softmax to the logits |

The main network row intentionally summarizes each stage only by its name,
the repeated `BasicBlock` count, and its output shape. The separate lower
schematics show the two-convolution residual structure. In `layer1`, the
first shortcut is an identity path. In `layer2`--`layer4`, the first shortcut
is a learned `1×1` projection with stride 2, required to match both channel
count and spatial size; it is not an untransformed identity. The remaining
blocks use identity shortcuts. The downsampling schematic labels its first
main-branch convolution as `3×3, stride 2` and its second as `3×3, stride 1`;
the identity schematic uses stride 1 for both. BatchNorm and ReLU are omitted
from the schematics, but remain in the implementation. The dimensions and
block counts are the measured values above.

All feature-map sizes in the diagram use channel-first `C×H×W` notation. In
particular, global average pooling produces `512×1×1`; flattening then gives a
512-dimensional vector before `Linear(512, 3)` produces the three logits.

## Frozen and fine-tuned parameters

`make_model` first sets every parameter to `requires_grad=False`, then
reenables gradients only for `layer4` and `fc`. The dummy check confirms:

- frozen: stem (`conv1` and `bn1`) and `layer1`, `layer2`, `layer3`;
- trainable: all of `layer4` and the final linear classifier;
- after `set_train_mode`, the stem BatchNorm and the BatchNorm layers in
  `layer1`--`layer3` remain in evaluation mode;
- `layer4` BatchNorm remains in training mode, as does the classifier (which
  has no BatchNorm).

The final handoff diagram `resnet18_final.svg`/`.png` uses blue for the frozen region and orange for the fine-tuned
region, with text labels so the distinction does not rely on colour alone.
The reference model has `dropout=0`; no tested alternative dropout or MLP
head is included.

`draw_resnet18_paper_architecture.py` is a presentation variant that reads the
same saved measurements and draws feature-map stacks, the classifier region and
two independent residual-block examples. The full diagram retains GAP,
flatten, Linear and evaluation-softmax as separate classifier steps. The
one-page report asset uses a compact `GAP → Linear` summary with the
512-dimensional flattened-vector transition written above the arrow; this is
only a layout reduction, not a model change. The identity and downsampling
shortcuts each terminate at their own addition node; the latter labels both
the `1×1, stride 2` projection shortcut and the first main-branch `3×3,
stride 2` convolution. It changes only the visual representation;
the dimensions, block counts, trainable region and output semantics above are
unchanged. Its full presentation assets are
`dis_resnet18_paper_architecture.svg` and `.png`.

## Reproduction and provenance

Run from the DIS repository root:

```bash
MPLCONFIGDIR=/private/tmp/dis_mpl_config \
  .venv/bin/python round2_resnet/architecture_review/draw_resnet18_architecture.py \
  --measure-only
MPLCONFIGDIR=/private/tmp/dis_mpl_config MPLBACKEND=Agg \
  python3 round2_resnet/architecture_review/draw_resnet18_architecture.py \
  --draw-only
MPLCONFIGDIR=/private/tmp/dis_mpl_config MPLBACKEND=Agg \
  python3 round2_resnet/architecture_review/draw_resnet18_paper_architecture.py
```

The first command performs the PyTorch dummy-input and mode/freeze check and
writes `architecture_measurements.json`. The second command reads only that
JSON and writes the original SVG and PNG diagram; the third draws the
paper-style presentation variant from the same JSON. `weights=None` is used
for structural instantiation, so no ImageNet weight download or checkpoint
access is needed.
