# Expanded Figure 1 review

This independent figure is a new layout for review. It does not replace the
published report asset and does not change the model or any saved result.

## Structural checks

The retained implementation is `round2_resnet/src/train_formal.py:35-64`:
torchvision ResNet-18, with `fc = Linear(512, 3)`. The saved dummy-input
measurements in `architecture_measurements.json` confirm:

- input: RGB `3×128×128`;
- stem: `7×7, stride 2`, then `3×3` max-pool, stride 2;
- layer1–4: two `BasicBlock`s per stage;
- stage outputs: `64×32×32`, `128×16×16`, `256×8×8`, `512×4×4`;
- global average pool: `512×1×1`, then flatten to `512`;
- classifier: `Linear(512, 3)` logits.

The freeze policy is frozen stem and layers 1–3, with layer4 and the final
Linear classifier trainable. The diagram uses blue and orange regions plus
explicit labels. BatchNorm and ReLU are omitted from the schematic but remain
in the implementation.

## Reference-image design choices

The teacher reference was used for its structural conventions: a horizontal
network with each residual block expanded into two convolution operations, a
shortcut crossing those two operations, and different line styles for identity
and projection shortcuts. Its task-specific parameters were not copied.

In this project, the first block of layers 2–4 uses a dashed `1×1, s=2`
projection shortcut and a first main-branch `3×3, s=2` convolution. All other
shortcuts are solid identity paths. Every shortcut is local to its own block
and ends at that block's own `⊕` node; no shortcut is drawn across an entire
stage. The classifier is kept compact so the expanded residual structure
remains readable at approximately 18 cm width.

## Reproduction

Run from the publication repository root; this reads only the saved measurement
JSON and writes the independent PNG, SVG and PDF:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl_expanded \
  python3 round2_resnet/architecture_review/draw_resnet18_expanded_review.py \
  --measurements round2_resnet/architecture_review/architecture_measurements.json \
  --output round2_resnet/architecture_review/dis_resnet18_expanded_review
```

The `*_18cm.png` file is the same vector-derived raster preview at 300 dpi;
its width is 18.18 cm. No image, checkpoint, training or inference input is
used.
