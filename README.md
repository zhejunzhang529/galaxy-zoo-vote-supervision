# Vote-distribution supervision for Galaxy Zoo morphology

This project asks whether retaining the normalized Galaxy Zoo 2 (GZ2) vote
distribution improves probability fitting without an unacceptable loss of
recognition of the rare `star or artifact` screening class. Raw vote fractions
are volunteer responses for an image, not physical truth. The experiment uses
static SDSS/GZ2 data and does not claim universal soft-label superiority or
cross-survey OOD ability.

Published repository: <https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision>

## Report and representative result

The one-page report is
[`round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf`](round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf).
The fixed-budget evaluation means on the natural pool are:

| target | CE | Brier | accuracy |
|---|---:|---:|---:|
| hard | 0.8805 | 0.0851 | 0.8892 |
| soft | 0.6053 | 0.0384 | 0.9034 |

These are means over three paired seeds under the fixed three-epoch protocol,
with 10,000 training objects and a fixed 750-object validation split. The
separate 20-epoch learning and regularisation diagnostics use train/validation
only; they are not a new independent evaluation.

## Quick Start: saved evidence only

Run from this repository root. The following commands do not train, load a
checkpoint or score an independent pool:

```bash
python3 round2_resnet/src/audit_resnet_package.py
python3 round2_resnet/src/closeout_analysis.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/reports/plot_dis_report_figures.py \
  --input round2_resnet/results/closeout/training_budget_diagnostic_runs.json \
  --output-stem round2_resnet/reports/assets/dis_training_curves_compact
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl_handoff \
  python3 round2_resnet/architecture_review/draw_resnet18.py
```

The audit reports checkpoint hashes as `SKIPPED` when checkpoint binaries are
absent. The saved-result plotting commands regenerate derived files, so use a
separate working copy if the retained previews must remain unchanged.

## Environment

The recorded versions are in
[`round2_resnet/requirements.txt`](round2_resnet/requirements.txt). They are a
historical record, not evidence of a fresh environment installation. A new
environment recipe is:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r round2_resnet/requirements.txt
```

Saved-result checks require NumPy, pandas, Matplotlib and Pillow. Training and
checkpoint inference additionally require PyTorch and torchvision. Notebook
execution requires Jupyter/nbconvert and nbclient; PDF compilation requires a
local `latexmk`/pdfLaTeX installation. These optional environments were not
freshly installed for this publication.

## Data acquisition and preprocessing

The official GZ2 data page is <https://data.galaxyzoo.org/>. The label table
used here is the Hart16 catalogue, available from the official page as
[`gz2_hart16.csv.gz`](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz).
The official GZ2 image archive is linked from that page through the Zenodo
record <https://zenodo.org/records/3565489>; its archive download record is
<https://zenodo.org/records/3565489/files/images_gz2.zip?download=1>.

Place the inputs at these paths relative to the repository root:

```text
data/raw/gz2_hart16.csv.gz
data/raw/gz2_filename_mapping.csv
data/raw/images_gz2.zip
```

`gz2_filename_mapping.csv` is the object/sample-to-archive-asset join table
provided with the GZ2 image record. The verified public file is
[`gz2_filename_mapping.csv`](https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1).
From the repository root, its expected acquisition command is:

```bash
curl --fail --location -o data/raw/gz2_filename_mapping.csv \
  'https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1'
```

The image archive and mapping are used together to resolve object IDs to
archive members; neither input is distributed in this repository.

From the repository root, the approved normalisation command is:

```bash
python3 normalize_labels.py \
  --input data/raw/gz2_hart16.csv.gz \
  --output data/processed/normalized_t01_labels.csv.gz \
  --manifest results/normalization_manifest.json
```

It preserves raw and weighted vote fields, normalises the three-class vector
by its positive row sum and derives the raw-vote argmax class. The read-only
`audit_gz2.py` checks labels, mapping and archive coverage. These inputs are
not distributed, and the data audit, image preparation, training and inference
commands are not package-only Quick Start checks.

The retained split is
[`round2_resnet/results/resnet_split.csv`](round2_resnet/results/resnet_split.csv)
with manifest
[`round2_resnet/results/split_manifest.json`](round2_resnet/results/split_manifest.json).
Regenerating it with
`python3 round2_resnet/src/prepare_split.py --config round2_resnet/configs/split.json`
requires the excluded inputs and is separate from using the frozen files.

## Diagnostics and limits

The eight soft-only grid candidates, four learning-rate endpoints, finite
regularisation comparison and two additional dropout seeds are documented in
`round2_resnet/configs/`, `round2_resnet/src/` and
`round2_resnet/results/closeout/`. The additional dropout seeds did not show a
consistent validation-soft-CE gain, so dropout=0 remains the reference.
Selection was by validation soft CE, with calibration discrepancy and earliest
epoch as tie-breakers; accuracy is therefore not a separately accuracy-tuned
comparison.

The study is limited to one architecture, 128-pixel inputs, a short fixed
training budget and three paired seeds. Peak memory was not measured. The
extra sample is an original-to-extra cross-sample test under selection shift,
not cross-survey validation. Physical-truth calibration and a larger natural
rare-class evaluation remain open.

## Navigation

- [`INDEX.md`](INDEX.md) — file map and scope.
- [`REPRODUCE.md`](REPRODUCE.md) — commands, asset requirements and boundaries.
- [`round2_resnet/README.md`](round2_resnet/README.md) — method and diagnostic map.
- [`round2_resnet/src/`](round2_resnet/src/) — source scripts.
- [`round2_resnet/reports/`](round2_resnet/reports/) — report sources and figures.
- [`notebooks/analysis_saved_results.ipynb`](notebooks/analysis_saved_results.ipynb) — saved-result analysis.
