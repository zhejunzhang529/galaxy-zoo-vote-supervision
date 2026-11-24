# DIS learning-rate endpoint diagnostic

This is a finite endpoint check, not a complete learning-rate search. It adds
`1e-5` and `1e-3` to the existing `1e-4` and `3e-4` records. All runs use seed
`20261119`, `small_3100`, the fixed 750-object validation split, ResNet-18 at
128 px, batch size 32, Adam with weight decay `1e-4`, `layer4+head`, no
scheduler, no augmentation, and 20 epochs without early stopping. Each run
starts from the same ImageNet-pretrained state and uses the existing epoch-wise
data-order rule.

## Endpoint run summary

| learning rate | target | seconds | soft-CE best epoch | accuracy best epoch | validation soft CE at epoch 20 |
|---:|---|---:|---:|---:|---:|
| `1e-5` | hard | 955.38 | 2 | 7 | 1.1345 |
| `1e-5` | soft | 973.14 | 8 | 12 | 0.6491 |
| `1e-4` | hard | existing | 1 | 7 | 1.9561 |
| `1e-4` | soft | existing | 11 | 7 | 0.6288 |
| `3e-4` | hard | existing | 1 | 18 | 2.1351 |
| `3e-4` | soft | existing | 19 | 11 | 0.6290 |
| `1e-3` | hard | 980.19 | 1 | 15 | 2.0276 |
| `1e-3` | soft | 982.34 | 9 | 4 | 0.6343 |

The four new runs took 3,891.04 s (64.85 min) in total. The pre-run estimate
was 3,600--4,400 s. No run produced a non-finite loss or runtime failure.

## First three epochs versus the full budget

| target | lr | validation soft CE e1 | e2 | e3 | minimum over 20 epochs | e20 |
|---|---:|---:|---:|---:|---:|---:|
| hard | `1e-5` | 0.8157 | 0.7378 | 0.7512 | 0.7378 (e2) | 1.1345 |
| hard | `1e-4` | 0.8231 | 0.9403 | 1.1338 | 0.8231 (e1) | 1.9561 |
| hard | `3e-4` | 0.8676 | 1.1644 | 1.2398 | 0.8676 (e1) | 2.1351 |
| hard | `1e-3` | 1.0458 | 1.1711 | 1.2707 | 1.0458 (e1) | 2.0276 |
| soft | `1e-5` | 0.8029 | 0.7099 | 0.6834 | 0.6475 (e8) | 0.6491 |
| soft | `1e-4` | 0.6524 | 0.6393 | 0.6410 | 0.6288 (e11) | 0.6288 |
| soft | `3e-4` | 0.6453 | 0.6288 | 0.6409 | 0.6251 (e19) | 0.6290 |
| soft | `1e-3` | 0.6549 | 0.6572 | 0.6495 | 0.6238 (e9) | 0.6343 |

The plot is `learning_rate_comparison_curves.png` with an SVG counterpart.
It uses only the saved histories; the existing intermediate-rate histories do
not contain per-epoch hard CE, maximum probability, or entropy, so those
missing quantities were not interpolated or reconstructed.

## Interpretation

- For hard supervision, `1e-5` has the best validation soft CE within this
  budget and declines more slowly after its early minimum. The three higher
  rates select the first epoch by soft CE, although their hard-target training
  losses continue to fall. This is consistent with faster hard-label fitting
  being poorly aligned with the vote-distribution objective. It does not by
  itself prove that `1e-4`, `3e-4`, or `1e-3` is intrinsically too large.
- For soft supervision, `1e-5` improves substantially after epoch 3 but remains
  near `0.65` from about epoch 8 onward. Its target training loss still falls
  through epoch 20, so the current budget cannot establish its eventual
  asymptotic performance. The intermediate rates reach lower validation soft
  CE, while `1e-3` reaches the lowest observed value (`0.6238` at epoch 9)
  before worsening to `0.6343` at epoch 20. This is a finite-budget pattern,
  not proof of a globally optimal rate or a universal high-rate instability.
- The `1e-3` curves show larger epoch-to-epoch changes than `1e-4` and
  `3e-4`, especially for accuracy and hard-target CE, but remain finite. That
  supports describing the endpoint as more oscillatory under this protocol,
  not as an optimizer failure.
- New endpoint metrics show the target distinction directly. At epoch 20,
  `1e-5 hard` and `1e-3 hard` reach training hard accuracy 1.0000 with mean
  maximum probabilities 0.9944 and 0.9961, while validation accuracies are
  0.8627 and 0.8760. Their validation soft CE is 1.1345 and 2.0276. The
  corresponding soft runs have validation soft CE 0.6491 and 0.6343, mean
  maximum probabilities 0.7604 and 0.7570, and mean prediction entropies
  0.5990 and 0.5985. These gaps support sharper hard-target fitting and a
  different probability-fit behaviour, but do not identify one causal factor.

The comparison remains limited to one seed, one fixed split, one architecture,
one weight decay, one unmodified preprocessing pipeline, and a 20-epoch
budget. No independent pool was loaded or scored, and no endpoint result was
used to alter the frozen formal evaluation protocol.

## Reproduction and files

From `validation_round1/dis`, the new training command was:

```bash
./.venv/bin/python round2_resnet/src/train_learning_rate_endpoints.py
```

The endpoint result JSON is
`learning_rate_endpoints_runs.json`; it contains every epoch's training loss,
train/validation hard and soft CE, Brier, accuracy, mean maximum probability,
entropy, calibration discrepancy, timing, and the three checkpoint records per
run. The endpoint configuration is
`configs/learning_rate_endpoints.json`. The comparison plot was generated from
the endpoint JSON plus the existing
`results/closeout/training_budget_diagnostic_runs.json` using
`src/plot_learning_rate_comparison.py` and system Python/Matplotlib; no model
inference is needed to redraw it.
