# Project guide

This repository studies vote-distribution supervision for three-class Galaxy
Zoo morphology. It contains the ResNet-18 implementation, saved evidence and
the one-page report. Raw data and model weights are not distributed.

## Recommended reading

1. `README.md` — research question, representative results, data sources and
   limitations.
2. `round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf` — one-page report;
   `round2_resnet/reports/DIS_resnet_stage_report_zh.md` is the Chinese
   explanation.
3. `REPRODUCE.md` — commands separated by required assets.
4. `round2_resnet/README.md` — methods, diagnostics and result navigation.

## Main files

- `round2_resnet/src/` — preprocessing, training, evaluation, audit and plot
  scripts.
- `round2_resnet/configs/` — fixed split, formal and diagnostic settings.
- `round2_resnet/results/` — retained metrics, manifests, predictions and
  derived figures.
- `round2_resnet/reports/` — Markdown, LaTeX, PDF, Chinese explanation and
  report figures.
- `round2_resnet/architecture_review/` — editable architecture source,
  measurements and vector/raster diagrams.

## Scope and exclusions

Saved-result commands read the retained CSV/JSON evidence. Training,
checkpoint inference, label generation, image preparation and split
regeneration require the excluded official assets. The repository excludes
raw labels and mapping files, the image archive, extracted images, checkpoints,
virtual environments, compiled files, `.git`, credentials and secrets.

The fixed-budget evaluation and the 20-epoch train/validation diagnostic are
separate evidence sets; the diagnostic is not a new independent evaluation.
