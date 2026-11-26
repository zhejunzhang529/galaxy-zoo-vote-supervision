# ResNet-18 vote-distribution study

This directory contains the ResNet-18 implementation, configurations, saved
evidence and report for the vote-distribution study. The report and result
files describe the fixed protocol and its limits.

## Research question

Under a fixed image model and training protocol, does retaining the normalized
raw GZ2 vote distribution improve probability fit without an unacceptable loss
of recognition of the rare `star or artifact` screening class? Raw vote
fractions are volunteer responses for an image, not physical truth.

## Scientific motivation and design choices

Galaxy morphology supports studies of galaxy evolution, structure, bars, spiral
features, and links between morphology and other physical properties. GZ2
retains multiple volunteer responses per image, so a vote distribution preserves
ambiguity that an argmax label discards. Ambiguity may reflect limited
resolution or signal-to-noise, a morphology between available answers, or
differences between classifiers; vote entropy alone cannot identify the cause.
The `star or artifact` branch is the root screening class, not a third galaxy
morphology. Raw fractions describe volunteer responses to one image, not
physical truth.

The ResNet-18 is a fixed design hypothesis: convolutional layers provide local
visual features for edges, bulges, bars and spiral-like structure, while
ImageNet pretraining supplies a reusable representation. Early layers are
frozen to limit cost and overfitting; layer4 plus the new head is fine-tuned to
allow limited adaptation to SDSS colour images. The 128-pixel input was chosen
for this feasibility budget because the recorded CPU throughput was 65.72
samples/s at 128 pixels versus 29.44 at 224 pixels. This is a resource choice,
not evidence that 128 pixels is optimal for astronomy, and ImageNet-to-SDSS
domain shift remains a limitation.

The eight short soft-target grid candidates vary update scale (`1e-4` versus
`3e-4` learning rate), no regularisation versus a small weight decay (`0` versus
`1e-4`), and adaptation cost (head-only versus layer4 plus head). All eight
were run before the common configuration was selected by soft-validation CE;
calibration discrepancy and earliest epoch were tie-breakers only. The hard/soft comparison is
conditional on that soft-only selection; it is not a separately tuned
hard-target search.

## Data-intensive evidence

The local Hart16 label table (Hart et al. 2016, MNRAS 461, 3663–3682,
doi:10.1093/mnras/stw1588) contains 239,695 labelled objects (209,294
`original`, 21,844 `extra`, and 8,557 `stripe82` rows). The mapping table has
355,990 rows and the official image archive is 3.35 GB compressed with 243,434
non-metadata JPG members. The formal large run uses 10,000 training objects,
750 validation objects, and 10,005 evaluation objects. The corresponding
extracted files occupy about 135.4 MB, 10.1 MB, and 135.2 MB. Archive storage,
extracted image bytes, decoded image memory, batch tensors and peak process
memory are different quantities; peak memory was not measured. In
`src/train_formal.py`, all distinct training/validation images are decoded,
resized and normalised before the first epoch and stored as tensors; batches
then stack cached tensors. The four 20-epoch diagnostic scripts use the same
cache through `train_formal.load_images`. In contrast, the historical short
grid `src/train_resnet.py` uses a Dataset that decodes in `__getitem__` with a
single-worker DataLoader. `src/evaluate_formal.py` and
`src/evaluate_extra_cross_sample.py` pre-cache their selected evaluation
images, whereas `src/benchmark_extra_inference.py` deliberately times decode,
resize, normalisation and CPU forward together. The 12 retained formal runs
took 2,145.3 CPU seconds in total (178.8 s mean). This is a static archive
experiment, not a real-time service; a live workflow would add matching,
cutout retrieval, decoding, storage, inference and quality control.

## Development diagnostics after the formal protocol

The fixed-budget evaluation is a three-epoch, two-size,
three-paired-seed experiment described in `configs/formal.json` and
`results/formal_runs.json`. It must not be conflated with the separate
train/validation-only diagnostics. The fixed 20-epoch budget diagnostic is
defined by `configs/training_budget_diagnostic.json`, implemented by
`src/train_budget_diagnostic.py`, and summarised in
`results/closeout/training_budget_diagnostic_runs.json` and
`results/closeout/training_budget_checkpoint_analysis.json`. The four learning
rate endpoint runs use `configs/learning_rate_endpoints.json` and
`src/train_learning_rate_endpoints.py`; their curves and analysis are in
`results/closeout/learning_rate_endpoints/`. The finite weight-decay/dropout
comparison uses `configs/regularization_diagnostic.json` and
`src/train_regularization_diagnostic.py`, with outputs in
`results/closeout/regularization_diagnostic/`. The two additional soft-target
dropout seeds use `configs/dropout_multiseed_confirmation.json` and
`src/train_dropout_multiseed_confirmation.py`, with outputs in
`results/closeout/dropout_multiseed_confirmation/`.

All of these runs used the fixed 3,100-object training subset and 750-object
validation split, 128px inputs, layer4 plus classifier fine-tuning and a
20-epoch budget, unless the individual JSON says otherwise; they did not load
the original, extra or other independent pools. Seed 20261119 supplied the
screening reference; seeds 20261120 and 20261121 were the additional dropout
confirmation runs. Their results did not show a consistent dropout=0.2 gain,
so the original dropout=0 reference was retained. This is a finite diagnostic,
not a separately tuned or statistically stable replacement for the formal
hard/soft evaluation.

## Reproduction scope

Run commands from the DIS repository root. The dependency record is
`round2_resnet/requirements.txt`; the recorded machine and device state is in
`round2_resnet/ENVIRONMENT.md`. The commands below are divided by required
assets. None of the saved-result commands trains, selects a checkpoint, or
scores an independent pool.

The environment recipe is:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r round2_resnet/requirements.txt
```

The requirements file records the historical versions used for the round; it
was not freshly installed and verified in this closeout. Saved-result audits
and plots need NumPy, pandas, Matplotlib and Pillow. Training/checkpoint
inference additionally need PyTorch and torchvision. The saved-result notebook
needs Jupyter/nbconvert and nbclient, and PDF compilation needs `latexmk` and
pdfLaTeX. The commands above are setup instructions, not evidence of a fresh
environment pass.

Saved-result audits and plots (no raw images or checkpoint binaries):

```bash
python3 round2_resnet/src/audit_resnet_package.py
python3 round2_resnet/src/closeout_analysis.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/src/plot_extra_cross_sample.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl \
  python3 round2_resnet/reports/plot_dis_report_figures.py \
  --input round2_resnet/results/closeout/training_budget_diagnostic_runs.json \
  --output-stem round2_resnet/reports/assets/dis_training_curves_compact
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl_handoff \
  python3 round2_resnet/architecture_review/draw_resnet18.py
```

The first four commands read retained CSV/JSON evidence and regenerate
derived audit or figure files. The architecture command reads the saved dummy
shape measurements; it does not construct a model or load weights. Building
the PDF additionally requires a local LaTeX installation:

```bash
latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=round2_resnet/reports/pdf \
  round2_resnet/reports/pdf_source/DIS_resnet_stage_report.tex
```

The formal feasibility run is pre-registered in `configs/formal.json`: three
paired seeds, hard/soft targets, 3,100 and 10,000 training objects, 128-pixel
inputs, layer4 plus head fine-tuning, and three epochs. Training, image
extraction, and inference require the excluded official image archive, local
processed images, and checkpoint binaries. They are documented for provenance
but are not claimed as package-level reproduction:

```bash
./.venv/bin/python round2_resnet/src/train_formal.py \
  --config round2_resnet/configs/formal.json
./.venv/bin/python round2_resnet/src/prepare_eval_images.py
./.venv/bin/python round2_resnet/src/evaluate_formal.py
./.venv/bin/python round2_resnet/src/summarize_formal.py
```

The labels, mapping table, image archive, processed images, weights,
checkpoints and virtual environment are excluded from publication. The
training and evaluation commands have not been rerun from this repository.

The official data source is https://data.galaxyzoo.org/. The Hart16 table is
linked directly at https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz and the
image archive is linked through https://zenodo.org/records/3565489 (recorded
file: https://zenodo.org/records/3565489/files/images_gz2.zip?download=1). The
mapping file is linked at
https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1.
Place the Hart16 GZ2 table at `data/raw/gz2_hart16.csv.gz`, the
object/sample-to-asset mapping at `data/raw/gz2_filename_mapping.csv`, and the
official image archive at
`data/raw/images_gz2.zip`. From the repository root, run:

```bash
python3 normalize_labels.py \
  --input data/raw/gz2_hart16.csv.gz \
  --output data/processed/normalized_t01_labels.csv.gz \
  --manifest results/normalization_manifest.json
```

Acquire the mapping file from the same official record with:

```bash
curl --fail --location -o data/raw/gz2_filename_mapping.csv \
  'https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1'
```

This creates the preserved raw and weighted columns, normalized vote fractions
and raw-vote argmax class. The
read-only `audit_gz2.py` checks these inputs and archive coverage. The current
frozen split can be regenerated only with those excluded inputs using
`python3 round2_resnet/src/prepare_split.py --config
round2_resnet/configs/split.json`; this is distinct from using the retained
`results/resnet_split.csv` and manifest. Historical pilot/expanded/independent
draws require their original local inputs and are not directly reproducible
without those historical inputs. The repository includes the current ResNet
source and configs, but not raw assets, processed images or weights.

## Evidence and limitations

`results/split_manifest.json` records ID/image isolation and sampling. The
selected-state checkpoints are listed in `results/formal_runs.json` with SHA256
hashes. `results/formal_summary.json` contains baselines, model metrics, paired
soft-minus-hard differences, object bootstrap intervals, and five-object
diagnostic recall intervals. The first checkpoint-saving error is retained in
`results/invalid_final_epoch_attempt/` and is not used for the report.

The new natural pool has no true class-2 object, so it cannot estimate natural
class-2 recall. It can still estimate false alarms: from the saved predictions,
the mean false-positive counts/rates are 41.7/0.417% (3,100 hard), 8.7/0.087%
(3,100 soft), 20.3/0.203% (10,000 hard), and 9.3/0.093% (10,000 soft). The
five-object diagnostic pool contains all remaining candidate class-2 objects
and is recall-only; detections by seeds 19/20/21 are 5/5/4 hard versus 2/2/2
soft at 3,100 objects, and 4/4/5 versus 3/5/5 at 10,000. Its Wilson intervals use an exchangeable-Bernoulli
description and are not population intervals. The three-epoch comparison is a
resource-limited feasibility study: the larger run has 3.23 times as many
optimizer updates, while class-2 remains 34 training objects and its fraction
falls from 1.10% to 0.34%. The 3,100 versus 10,000 comparison therefore changes
both update budget and composition.

All eight short grid runs used soft targets. The common configuration was
selected by soft-validation CE, with calibration discrepancy and earliest
epoch as tie-breakers only. Layer4+head had lower validation CE than head-only, `3e-4` was more
effective than `1e-4` in this short search, and weight-decay effects were small
and learning-rate dependent; these are not claims of numerical optimality.

The current report figures are
`reports/assets/dis_resnet18_paper_architecture_report.pdf` and
`reports/assets/dis_training_curves_compact.pdf`; the full architecture and
curve sources are retained in `architecture_review/` and `reports/`. The
historical summary figures are `results/resnet_framework.svg` and
`results/probability_vs_recall.svg`, with PNG previews. The latter includes
natural-pool false-positive counts and positive-only diagnostic detections and
was generated from saved predictions; it does not rerun scoring.

## Frozen original-to-extra cross-sample evaluation

The GZ2 `extra` sample is not a different survey. Willett et al. describe it as
objects added after the original query omitted galaxies carrying both BLENDED
and CHILD flags. It is therefore treated as `original-to-extra cross-sample
generalisation under selection shift`, not proven OOD or cross-survey
generalisation. The local labels, mapping, and image archive provide 21,836
eligible extra objects after checking archive coverage and excluding every
object ID, asset ID and image member in the retained development, validation,
historical and current evaluation manifests. A fixed seed without class quotas
selected 1,000 objects before inference. The six frozen large_10000 checkpoints
were scored with the existing 128-pixel transform and no selection or
calibration change.

The six per-checkpoint forward-only records sum to 28.20 seconds. The recorded
31.62-second surrounding loop includes loading and metric/bootstrap work; it is
not a pure forward time. The small-batch estimate remains 48.52 seconds.

The selected extra pool has hard-class proportions 0.613/0.383/0.004 for
smooth/features-or-disk/star-or-artifact, compared with 0.7586/0.2414/0 in the
original natural pool. Mean normalised vote entropy is 0.586 versus 0.487, and
the mean top-two vote margin is 0.481 versus 0.614. On extra, the three-seed
means are hard CE/Brier/accuracy `0.9592/0.1169/0.7843` and soft
`0.7417/0.0611/0.8117`; the corresponding original natural values for the
10,000-object models are hard `0.8805/0.0851/0.8892` and soft
`0.6053/0.0384/0.9034`. These descriptive shifts are consistent with changed
sample composition and vote ambiguity; they do not establish an OOD law,
physical-truth calibration, or universal soft-label superiority. Class-2 has
only four extra objects, so its classwise metrics are descriptive, not a stable
population estimate. Object bootstrap intervals and seed sample SD are kept
separate in `extra_cross_sample_results.json`.

The frozen cross-sample commands are:

```bash
python3 round2_resnet/src/prepare_extra_cross_sample.py
python3 round2_resnet/src/extract_extra_cross_sample.py
./.venv/bin/python round2_resnet/src/benchmark_extra_inference.py
./.venv/bin/python round2_resnet/src/evaluate_extra_cross_sample.py
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dis_mpl python3 round2_resnet/src/plot_extra_cross_sample.py
```

The first two commands require the local official labels, mapping and archive;
the benchmark and evaluation require the six excluded checkpoint files and
project environment. The last command reads saved CSV/JSON only.

## Closeout saved-result analysis

The current-round ResNet evidence audit is separate from the historical
TinyCNN audit. Run it first:

```bash
python3 round2_resnet/src/audit_resnet_package.py
```

It checks all 12 ResNet records, selected epochs, split hashes, evaluation
probabilities, recomputed metrics and paired summaries. Checkpoint binary hash
checks are reported as SKIPPED when the binaries are absent from a public checkout.

src/closeout_analysis.py writes vote entropy/top-two-margin groups, the five
remaining star or artifact cases, weighted-target sensitivity, scale-matched
constant baselines, and resource estimates from saved predictions. It does not
load images or checkpoints. src/run_traditional_baseline.py is one fixed
16-by-16 RGB multinomial-logistic supplement; standardization is fit on the
3,100-object training subset only. Its use of the 10,000-object natural pool
is post hoc, not a new blind evaluation.

    python3 round2_resnet/src/closeout_analysis.py
    python3 round2_resnet/src/run_traditional_baseline.py  # requires local images

The first command needs saved JSON/CSV only. The second needs excluded
processed images and SciPy/Pillow. The saved-result notebook is
notebooks/analysis_saved_results.ipynb.

The corrected baseline stores the actual optimizer status, message, iteration
count, objective and gradient-check result. The previous inconsistent-gradient
result and corrected-but-iteration-limited attempts are retained with
invalid_attempt or nonconverged filenames and are not cited as valid results.
The valid supplement uses the mean-weight L2 objective and converged L-BFGS-B
settings recorded in configs/traditional_baseline.json.
