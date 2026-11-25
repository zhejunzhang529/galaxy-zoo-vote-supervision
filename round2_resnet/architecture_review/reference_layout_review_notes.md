# Reference-layout architecture review

This is an independent review asset. It does not replace the published
Figure 1, PDF, ZIP or any result file.

## What was checked

The diagram was drawn from the saved dummy-input measurements and the retained
implementation in `round2_resnet/src/train_formal.py:35-64`:

- input: RGB `3×128×128`;
- stem: `7×7` convolution with stride 2, then `3×3` max-pool with stride 2;
- layer1–4: two `BasicBlock`s each, with outputs
  `64×32×32`, `128×16×16`, `256×8×8`, and `512×4×4`;
- layer2–4 first blocks: main `3×3, stride 2` plus dashed `1×1, stride 2`
  projection shortcut;
- all other shortcuts: identity;
- GAP: `512×1×1`, flatten to 512, then `Linear(512, 3)` logits;
- frozen: Stem and layers 1–3; trainable: layer4 and Linear.

BN/ReLU are omitted from the schematic. Hard and soft supervision use the same
network. Projection shortcuts are shown as part of the residual blocks and do
not add extra backbone stages.

## Layout reference

The user-provided reference image was used only for its visual conventions:
horizontal operator bars, rotated convolution labels, curved local shortcuts,
and bracketed stage groups. Its input size, channel counts, classifier and
other task-specific values were not copied.

## Reproduction and preview

Run from the publication repository root:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl_reference \
  python3 round2_resnet/architecture_review/draw_resnet18_reference_layout_review.py \
  --measurements round2_resnet/architecture_review/architecture_measurements.json \
  --output round2_resnet/architecture_review/dis_resnet18_reference_layout_review
```

The source is vector-native Matplotlib and writes PNG, SVG and PDF. The
`*_18cm.png` preview is a 300-dpi raster at approximately 18 cm wide, created
from the same render. No image, checkpoint, training or inference asset is
read.
