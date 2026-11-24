# DIS training-budget diagnostic

Status: read-only review and planned extension; no training or evaluation-pool
scoring was run in this step.

## Evidence reviewed

- `configs/grid.json` and `results/grid_short.json`: the registered eight-way
  search, all with three epochs and a soft normalized vote target.
- `configs/formal.json` and `results/formal_runs.json`: twelve paired runs
  using seeds 20261119, 20261120 and 20261121, both targets, and the
  `small_3100` and `large_10000` training sizes.
- `src/train_resnet.py` and `src/train_formal.py`: actual loss, validation,
  data-order and checkpoint-selection implementations.
- `results/throughput.json` and the `training_seconds` fields in
  `results/formal_runs.json`: resource estimates only.

No original, extra or other independent evaluation pool was loaded for this
diagnostic.

## What the existing records establish

The grid training loss is soft-target cross-entropy because
`src/train_resnet.py` multiplies `log_softmax` by the normalized vote vector.
In the formal runs, `target="hard"` uses a one-hot argmax target while
`target="soft"` uses the vote vector in `src/train_formal.py`. The recorded
training losses therefore have different target definitions and should not be
compared as if they were one common loss. Validation soft CE is computed from
the same normalized vote target for both modes, together with soft Brier and
classwise soft calibration discrepancy; this is also the checkpoint-selection
key. Accuracy is not the selection objective.

All eight grid configurations selected an epoch within the three-epoch budget:
six selected epoch 3, one epoch 2, and one epoch 1. The short grid therefore
does not show whether the selected configuration would remain best later.

The grid curves show three distinct patterns:

- The two head-only settings at each learning rate have nearly identical
  curves for weight decay 0 versus `1e-4` in this short run. Their validation
  soft CE decreases through epoch 3.
- At `1e-4`, layer4+head validation soft CE decreases slightly through epoch 3
  (`0.6459 -> 0.6442 -> 0.6417` with zero decay; `0.6462 -> 0.6454 ->
  0.6442` with `1e-4` decay), while training loss also decreases.
- At `3e-4`, layer4+head validation soft CE is non-monotone: with zero decay
  it is `0.6342 -> 0.6330 -> 0.6447`, and with `1e-4` decay it is
  `0.6306 -> 0.6372 -> 0.6474`. Training loss continues to decrease. This
  is evidence of a short-run train/validation separation, not proof that the
  learning rate alone caused it.

The formal records strengthen the case for an extension but do not determine
the eventual budget. For hard targets, mean training loss falls while mean
validation soft CE rises from epoch 1 to 3 at both scales:

| scale | mean training loss, epochs 1–3 | mean validation soft CE, epochs 1–3 | selected epochs |
|---|---:|---:|---|
| 3,100 hard | 0.3981, 0.1983, 0.1346 | 0.8980, 1.0809, 1.3738 | 1, 1, 1 |
| 10,000 hard | 0.3147, 0.2176, 0.1757 | 0.8993, 1.0266, 1.1004 | 1, 1, 1 |

For soft targets, training loss falls in every run, while validation CE is
mostly flat or mildly non-monotone:

| scale | mean training loss, epochs 1–3 | mean validation soft CE, epochs 1–3 | selected epochs |
|---|---:|---:|---|
| 3,100 soft | 0.6876, 0.6075, 0.5890 | 0.6479, 0.6381, 0.6370 | 2, 3, 2 |
| 10,000 soft | 0.6406, 0.6024, 0.5924 | 0.6228, 0.6237, 0.6183 | 3, 3, 2 |

Thus some soft runs still improve at epoch 3, while hard runs have already
separated from their validation soft-CE objective by epoch 2. These observations
justify a finite convergence diagnostic. They do not prove that more epochs
will improve generalization, and they do not make hard and soft training-loss
values directly comparable.

The formal configuration was `lr=3e-4`, `weight_decay=1e-4`,
`layer4_plus_head`, Adam, 128 pixels and batch size 32. The three seeds use the
same initial state between hard and soft runs and the same epoch-dependent data
permutation, as implemented in `src/train_formal.py`. The current comparison
uses a common validation-soft-CE checkpoint rule, so it is a probability-fit
comparison under that rule rather than separate accuracy-optimised tuning.

## Candidate extended configurations

Only two candidates are retained; a third would add a fine-tuning or
regularisation dimension without sufficient evidence from the three-epoch
records.

1. **Current configuration:** `lr=3e-4`, `weight_decay=1e-4`,
   `layer4_plus_head`, Adam, 128 pixels, batch size 32. This tests whether the
   selected short-run configuration recovers after its observed epoch-1/2
   validation minima. It is the required continuation baseline.
2. **Lower-learning-rate control:** `lr=1e-4`, `weight_decay=1e-4`,
   `layer4_plus_head`, with all other settings unchanged. In the grid its
   validation soft CE was still decreasing at epoch 3, so it tests whether the
   lower update scale needs more epochs to reach a stable validation region.

The short search did not provide a clean basis for adding another weight-decay
value or head-only arm: the weight-decay contrast was almost invisible for
head-only and small, configuration-dependent for layer4+head, while layer4+head
had lower short-run validation CE than head-only. Those dimensions remain
unresolved rather than being silently treated as settled.

## Proposed finite diagnostic

Use `small_3100` first because its measured three-epoch runs take about
90.2–95.8 CPU seconds, versus about 263.0–268.3 seconds for `large_10000`.
Run both targets and the three existing paired seeds for both candidates: 12
runs in total. Keep the split, 128-pixel preprocessing, batch size, optimizer,
initialisation, data-order rule and validation set unchanged.

- Maximum budget: 8 epochs per run; validate and save the metrics after every
  epoch.
- Diagnostic stopping rule: do not stop before epoch 4; after that, stop after
  two consecutive validation epochs without an improvement of at least
  `0.001` in validation soft CE. Keep all completed histories and select the
  checkpoint with minimum validation soft CE, then minimum classwise soft
  calibration discrepancy, then earliest epoch.
- Interpretation: this is a convergence/overfitting diagnostic. Its early
  stopping times are not a fair final hard/soft budget comparison. If a later
  formal comparison is justified, hard and soft must be rerun under one
  pre-declared common budget and the same checkpoint rule.
- Estimated cost: using the recorded small-run rate, roughly 4.0–4.3 minutes
  per 8-epoch run and about 48–52 CPU minutes for 12 runs before early
  stopping. The observed machine used CPU (`mps_available=false`,
  `cuda_available=false`, six Torch threads); this is an estimate, not a
  guarantee.

No new architecture, dropout, batch size, data source, evaluation pool or
hyperparameter search is included. The large training scale can be considered
only after this small-scale diagnostic, using the same pre-declared rule.
