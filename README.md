# DIS: vote-distribution supervision for Galaxy Zoo 2 morphology

This personal project asks whether retaining three-class GZ2 raw vote fractions
improves probability-target fitting without an unacceptable recognition trade-off
relative to their argmax class. Raw votes are volunteer responses for an image,
not physical truth. The project uses static SDSS/GZ2 data and does not claim
universal soft-label superiority or general OOD ability.

Published repository: <https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision>
This is a cleaned publication snapshot sourced from local research commit
`8cdee09e9110f18f55340f0f7d06a6451eacc0f3`; it does not rewrite or replace the
private/local development history.

## Current round

`round2_resnet/` is the current ResNet-18 study. It contains the fixed
ImageNet-pretrained model, 128-pixel preprocessing, eight soft-only grid
records, 12 formal runs, the original natural-pool evaluation, and the frozen
original-to-extra cross-sample test. The latter uses six existing 10,000-object
checkpoints on a fixed 1,000-object GZ2-extra pool; it is a selection-shift
test within GZ2, not cross-survey validation.

The formal evaluation is the original three-epoch paired hard/soft protocol
on two training sizes and three seeds. Separately, the development diagnostics
contain fixed 20-epoch train/validation runs: the learning-rate endpoints are
at [`configs/learning_rate_endpoints.json`](round2_resnet/configs/learning_rate_endpoints.json)
with source [`src/train_learning_rate_endpoints.py`](round2_resnet/src/train_learning_rate_endpoints.py)
and results under
[`results/closeout/learning_rate_endpoints/`](round2_resnet/results/closeout/learning_rate_endpoints/);
the budget diagnostic is recorded in
[`configs/training_budget_diagnostic.json`](round2_resnet/configs/training_budget_diagnostic.json),
[`src/train_budget_diagnostic.py`](round2_resnet/src/train_budget_diagnostic.py) and
[`results/closeout/training_budget_diagnostic_runs.json`](round2_resnet/results/closeout/training_budget_diagnostic_runs.json);
and the finite regularisation and two-new-seed dropout checks are under
[`configs/regularization_diagnostic.json`](round2_resnet/configs/regularization_diagnostic.json),
[`configs/dropout_multiseed_confirmation.json`](round2_resnet/configs/dropout_multiseed_confirmation.json),
their corresponding `src/train_*.py` scripts, and
[`results/closeout/regularization_diagnostic/`](round2_resnet/results/closeout/regularization_diagnostic/)
and [`results/closeout/dropout_multiseed_confirmation/`](round2_resnet/results/closeout/dropout_multiseed_confirmation/).
These diagnostics use only the fixed training and validation split. The two
new dropout seeds did not show a consistent validation-soft-CE gain, so the
original dropout=0 reference configuration is retained; this does not turn the
diagnostics into a new formal multi-seed comparison.

The current one-page report is
[`round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf`](round2_resnet/reports/pdf/DIS_resnet_stage_report.pdf)
and is based on saved results. The safe saved-result checks are:

```bash
python3 round2_resnet/src/audit_resnet_package.py
python3 round2_resnet/src/closeout_analysis.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl python3 round2_resnet/src/plot_extra_cross_sample.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/reports/plot_dis_report_figures.py \
  --input round2_resnet/results/closeout/training_budget_diagnostic_runs.json \
  --output-stem round2_resnet/reports/assets/dis_training_curves_compact
```

`audit_resnet_package.py` and the plotting commands use retained results; the
data-intensive audit additionally reads the local official labels, mapping and
image archive. It is therefore not a package-only check. `closeout_analysis.py`
and the figure command use retained CSV/JSON only. The extra cross-sample
scoring commands are listed in `round2_resnet/README.md`; they require the
local official label/mapping/archive assets and, for inference, the excluded
checkpoint files. The fixed data acquisition source is the official Galaxy Zoo
data page: https://data.galaxyzoo.org/.

The data-intensive audit is separately documented and requires the local
archive:

```bash
python3 round2_resnet/src/audit_data_intensive.py
```

## Quick Start and data boundary

Run from this DIS repository root. A fresh environment can be created with
the following commands, but the recorded requirements have not been freshly
installed and verified in this closeout:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r round2_resnet/requirements.txt
```

For saved-result audits and plots, the core packages are NumPy, pandas,
Matplotlib and Pillow. Training and checkpoint inference additionally require
PyTorch and torchvision. The notebook needs Jupyter/nbconvert and nbclient;
LaTeX PDF compilation needs a local `latexmk`/pdfLaTeX installation. No
installation command above was executed here, so it is an environment recipe,
not a fresh-environment verification.

The official source is the Galaxy Zoo data page, https://data.galaxyzoo.org/.
The expected local assets are `data/raw/gz2_hart16.csv.gz` (Hart16 GZ2 label
table), `data/raw/gz2_filename_mapping.csv` (object/sample to asset mapping),
and `data/raw/images_gz2.zip` (official image archive). `normalize_labels.py`
reads the first file and writes
`data/processed/normalized_t01_labels.csv.gz` plus
`results/normalization_manifest.json`; it preserves raw and weighted values,
normalises each three-class vector by its positive row sum, and derives the
raw-vote argmax class. Run it from the repository root, for example:

```bash
python3 normalize_labels.py \
  --input data/raw/gz2_hart16.csv.gz \
  --output data/processed/normalized_t01_labels.csv.gz \
  --manifest results/normalization_manifest.json
```

The read-only `audit_gz2.py` checks the label table, mapping and archive
coverage. The retained frozen ResNet split is
`round2_resnet/results/resnet_split.csv` with
`round2_resnet/results/split_manifest.json`. Regenerating it is a separate,
asset-dependent operation using
`round2_resnet/src/prepare_split.py --config round2_resnet/configs/split.json`;
it requires the raw labels, mapping and image ZIP and is not part of the
package-only Quick Start. The historical pilot/expanded/independent split
scripts depend on their original local inputs and are not claimed to recreate
the historical draws from the review package alone. The approved normalization,
audit and current split-preparation sources are retained in the repository;
raw inputs, extracted images and weights are intentionally absent.

## Results and limitations

On the original natural pool, the 10,000-object model means are hard
CE/Brier/accuracy `0.8805/0.0851/0.8892` and soft
`0.6053/0.0384/0.9034`. On the extra pool they are hard
`0.9592/0.1169/0.7843` and soft `0.7417/0.0611/0.8117`. The extra pool has
four class-2 objects, so its rare-class metrics are descriptive. The study is
short-budget, one-architecture, 128-pixel and three-seed; peak memory,
physical-truth calibration, cross-survey/OOD ability and a larger rare-class
evaluation remain unestablished.

## Repository navigation

- `round2_resnet/src/`: split, training, evaluation, audit, resource and plot scripts.
- `round2_resnet/configs/`: fixed formal, grid, throughput and split configurations.
- `round2_resnet/results/`: retained metrics, manifests, hashes, predictions and figures.
- `round2_resnet/reports/`: one-page English report, Chinese explanation and teacher brief.
- `notebooks/analysis_saved_results.ipynb`: saved-result analysis without training.
- `round2_resnet/requirements.txt`: recorded Python package versions.
- `round2_resnet/ENVIRONMENT.md`, `REPRODUCE.md`: versions, hardware, asset boundaries and command map.

The historical TinyCNN work remains in the parent repository for provenance and
is not substituted for the current ResNet audit. Teacher coursework archives
are reference examples only and are not part of this repository.

## Public repository preparation

The `.gitignore` covers raw data, processed images, checkpoints, virtual
environments, common Python/LaTeX build products, and common credential
filenames. Ignore rules do not remove files already tracked by Git, so
`git ls-files` must still be reviewed before publication. Before publication,
review `git diff`, citations and data-access conditions. The public repository
is now available at
<https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision>. Full training,
data acquisition and checkpoint inference remain asset-dependent and are not
claimed as package-level reproduction.
