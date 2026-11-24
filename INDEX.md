# DIS publication candidate

This package contains the current personal DIS ResNet-18 study and its
saved-result evidence. It is not a claim that the full training or inference
workflow is reproducible without the excluded data and checkpoint assets.

Published repository: <https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision>

## Recommended reading

1. `README.md` - research question, main results, limitations and navigation.
2. `round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf` - one-page English
   report; `round2_resnet/reports/DIS_resnet_stage_report_zh.md` provides the
   Chinese explanation.
3. `teacher_brief.md` - current teacher-facing summary.
4. `REPRODUCE.md` - commands separated into saved-result checks and commands
   requiring excluded assets.
5. `round2_resnet/results/` and `round2_resnet/requirements_evidence_matrix.md`
   - traceable evidence and limitations.

## Candidate provenance

- Source repository: local DIS repository, commit
  `8cdee09e9110f18f55340f0f7d06a6451eacc0f3` (`Clarify DIS reproduction boundaries`).
- Current study: `round2_resnet/`; historical TinyCNN work is not included.
- The full architecture SVG/PNG/PDF is under
  `round2_resnet/architecture_review/`. The report uses the compact classifier
  asset under `round2_resnet/reports/assets/` to preserve one-page layout.

## Contents and exclusions

Included are source scripts, fixed JSON configurations, Markdown/LaTeX report
sources, PDFs and previews, architecture sources, saved CSV/JSON metrics,
figures, manifests and the saved-result notebook. The package root also keeps
the approved `normalize_labels.py`, `audit_gz2.py` and `config.json` sources
needed to document the official-label audit boundary; they do not include the
data they read.

Excluded are raw labels and mapping files, the image ZIP, extracted images,
all model checkpoints, virtual environments, compiled/intermediate build
files, `.git`, teacher coursework, credentials, account information and
secrets. Commands requiring those assets are listed but are not claimed as
package-level reproduction.

Checkpoint hash checks therefore report `SKIPPED` in a clean extraction. The
local repository audit may report `PASS` when the original checkpoint files
are present.

The package-only Quick Start is the saved-result audit and redraw commands in
`REPRODUCE.md`. Training, checkpoint inference, data audit, label generation,
and split regeneration require the excluded assets and are listed separately.
The notebook source is included, but a full Jupyter execution still depends
on an installed notebook environment.
