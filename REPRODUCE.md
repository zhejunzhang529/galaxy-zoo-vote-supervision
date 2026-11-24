# DIS reproduction map

Run from the DIS repository root. The current ResNet-18 report is
`round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf`; the Markdown source,
Chinese explanation and architecture source are under
`round2_resnet/reports/` and `round2_resnet/architecture_review/`.

## Environment setup

The recorded dependency versions are in `round2_resnet/requirements.txt`.
They are a historical record and were not freshly installed and verified in
this closeout. A new environment recipe is:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r round2_resnet/requirements.txt
```

Saved-result checks need NumPy, pandas, Matplotlib and Pillow. Training and
checkpoint inference additionally need PyTorch and torchvision. The notebook
requires Jupyter/nbconvert and nbclient; PDF compilation requires a local
`latexmk`/pdfLaTeX installation. No command above is evidence that a new
environment has been installed successfully.

## Saved-result checks and redraws

These commands use retained CSV/JSON evidence only. They do not train, load a
checkpoint, or score an independent pool:

```bash
python3 round2_resnet/src/audit_resnet_package.py
python3 round2_resnet/src/closeout_analysis.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/src/plot_extra_cross_sample.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/reports/plot_dis_report_figures.py \
  --input round2_resnet/results/closeout/training_budget_diagnostic_runs.json \
  --output-stem round2_resnet/reports/assets/dis_training_curves_compact
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/architecture_review/draw_resnet18_paper_architecture.py \
  --measurements round2_resnet/architecture_review/architecture_measurements.json \
  --output-stem round2_resnet/architecture_review/dis_resnet18_paper_architecture \
  --report-output-stem round2_resnet/reports/assets/dis_resnet18_paper_architecture_report
```

`audit_resnet_package.py` returns non-zero when a required saved-evidence
check fails. Checkpoint binary hashes are reported as `SKIPPED` when the
checkpoint files are absent. `closeout_analysis.py` recomputes post-hoc
tables from saved predictions and labels; it does not rerun inference.

The saved-result notebook is
`notebooks/analysis_saved_results.ipynb`. A local Jupyter installation can
execute it with:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120 \
  --output /tmp/dis_saved_results_executed.ipynb \
  notebooks/analysis_saved_results.ipynb
```

This is an analysis check, not a training notebook. The review package
includes the notebook source but not its execution environment. In the current
machine check, the command was not executable because the `jupyter` command
was unavailable; a complete notebook execution is therefore `SKIPPED`. A
Python sequential execution of notebook cells would not by itself be reported
as a full Jupyter verification.

## Report build

The PDF build requires a local LaTeX installation in addition to the Python
packages recorded in `round2_resnet/requirements.txt`:

```bash
latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=round2_resnet/reports/pdf \
  round2_resnet/reports/pdf_source/DIS_resnet_stage_report.tex
```

The report uses the generated PDF assets under
`round2_resnet/reports/assets/`. The checked-in preview is
`round2_resnet/reports/pdf/DIS_resnet_stage_report_page1.png`.

## Commands requiring excluded assets

The following are provenance commands only. They require the official GZ2
label/mapping files and image archive, local processed images, and/or model
checkpoint binaries. They were not rerun for the publication candidate and
must not be used to tune the frozen protocol:

```bash
python3 round2_resnet/src/audit_data_intensive.py
./.venv/bin/python round2_resnet/src/train_formal.py \
  --config round2_resnet/configs/formal.json
./.venv/bin/python round2_resnet/src/prepare_eval_images.py
./.venv/bin/python round2_resnet/src/evaluate_formal.py
./.venv/bin/python round2_resnet/src/summarize_formal.py
```

The original-to-extra preparation and scoring commands are listed in
`round2_resnet/README.md`. They require the fixed image archive and six
excluded large-model checkpoints; their saved outputs are included only as
evidence, not as a claim of package-level rerun.

## Data acquisition and split boundary

The official source is the Galaxy Zoo data page, https://data.galaxyzoo.org/.
Expected paths, relative to the repository root, are:

```text
data/raw/gz2_hart16.csv.gz
data/raw/gz2_filename_mapping.csv
data/raw/images_gz2.zip
```

From the repository root, `normalize_labels.py` generates
`data/processed/normalized_t01_labels.csv.gz` and
`results/normalization_manifest.json` from the Hart16 table, preserving raw
and weighted votes and deriving normalized three-class fractions. The
read-only `audit_gz2.py` checks labels, mappings and archive coverage. The
approved current split source is
`round2_resnet/src/prepare_split.py`; its command is:

```bash
python3 round2_resnet/src/prepare_split.py \
  --config round2_resnet/configs/split.json
```

That command needs the excluded raw labels, mapping and image ZIP and writes
the current split/manifest paths in the config; it is not part of the
package-only Quick Start and must not overwrite retained formal results during
a documentation check. The retained frozen split is used directly by the
saved-result commands. Historical pilot, expanded and independent split
regeneration requires its original local inputs; missing-input steps are not
claimed reproducible from the review package.

## Environment and publication boundary

Use `round2_resnet/ENVIRONMENT.md` for the recorded Python, PyTorch,
torchvision, CPU/MPS and storage details. Raw data, extracted images,
checkpoints, virtual environments, compiled files, teacher coursework,
credentials and Git metadata are excluded from the publication candidate;
`.gitignore` covers common secret filenames and build products but does not
remove files already tracked by Git, so inspect `git ls-files` before release.
The publication snapshot is available at
<https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision>. The local
development repository remains separate; this snapshot does not claim to
reproduce its full history.
