# DIS soft-target dropout multi-seed confirmation

This analysis adds two new seeds to the fixed soft-target comparison at
`lr=3e-4`, `weight_decay=1e-4`, `small_3100`, and the existing 750-object
validation split. The model is ResNet-18 at 128 px with `layer4+classifier`
fine-tuning, batch size 32, Adam, no scheduler, no augmentation, and 20
epochs without early stopping. Within each new seed, dropout `0` and `0.2`
share the same initial state and epoch-wise data order.

Seed `20261119` is reused from the previously screened results and is not an
independent confirmation seed. Seeds `20261120` and `20261121` are the new
confirmation runs.

## Paired per-seed results

The main checkpoint rule is minimum validation soft CE, then calibration
discrepancy, then earliest epoch. “Last ten” means the mean validation soft CE
over epochs 11--20. Differences are dropout minus original; negative CE or
Brier differences favour dropout, while positive accuracy differences favour
dropout.

| seed / role | configuration | best epoch | best soft CE | best Brier | best accuracy | last-ten soft CE |
|---|---|---:|---:|---:|---:|---:|
| 20261119 screened | original | 19 | 0.6251 | 0.0426 | 0.8773 | 0.6290 |
| 20261119 screened | dropout 0.2 | 13 | 0.6227 | 0.0415 | 0.8853 | 0.6284 |
| 20261120 new | original | 12 | 0.6246 | 0.0431 | 0.8787 | 0.6281 |
| 20261120 new | dropout 0.2 | 12 | 0.6239 | 0.0430 | 0.8720 | 0.6268 |
| 20261121 new | original | 17 | 0.6236 | 0.0420 | 0.8813 | 0.6292 |
| 20261121 new | dropout 0.2 | 18 | 0.6252 | 0.0421 | 0.8787 | 0.6292 |

| seed | Δ best soft CE | Δ best Brier | Δ best accuracy | Δ last-ten soft CE |
|---:|---:|---:|---:|---:|
| 20261119 screened | -0.00238 | -0.00110 | +0.00800 | -0.00061 |
| 20261120 new | -0.00075 | -0.00005 | -0.00667 | -0.00127 |
| 20261121 new | +0.00157 | +0.00017 | -0.00267 | -0.00000 |

## Seed-level aggregation

For the two new confirmation seeds, the paired differences have the following
mean and sample SD:

| quantity | mean dropout − original | sample SD |
|---|---:|---:|
| best soft CE | +0.00041 | 0.00164 |
| best Brier | +0.00006 | 0.00015 |
| best accuracy | -0.00467 | 0.00283 |
| last-ten soft CE mean | -0.00064 | 0.00090 |

The three-seed summary, including the screened seed for descriptive context,
is mean `-0.00052` (sample SD `0.00199`) for best soft CE and mean `-0.00063`
(sample SD `0.00064`) for the last-ten soft-CE difference. These three-seed
values are descriptive, not an independent confirmation interval: seed
`20261119` participated in the original candidate screening. The ten epochs
within a seed are trajectories, not ten independent repetitions.

## Interpretation and boundary

The two new seeds do not give a consistent best-checkpoint improvement: dropout
is slightly better for seed `20261120` but worse for seed `20261121`, with a
two-seed mean close to zero relative to the sample SD. The last-ten trajectory
mean is lower with dropout for both new seeds, but the effect is small and is
essentially zero for seed `20261121`. Validation accuracy is lower with dropout
for both new seeds. On this evidence, the original configuration remains the
main reference; the result does not justify automatically replacing it or
adding more seeds.

This is only a soft-target dropout stability check. It is not a new multi-seed
hard/soft comparison, not an independent evaluation, and not evidence that
dropout is generally beneficial or harmful. No original, extra, pilot, or
other independent evaluation pool was loaded or scored.

## Reproducibility and cost

The new machine-readable run file is
`dropout_multiseed_confirmation_runs.json`; the paired summary is
`dropout_multiseed_summary.json`. The curve figure is
`dropout_multiseed_confirmation_curves.png` with an SVG counterpart. The
summary and figure use the saved new-run JSON plus the existing screened
records; no model is rerun for analysis.

The four new runs took 3,848.44 s (64.14 min), compared with the pre-run
estimate of 3,800--4,600 s. Each new run has 20 epoch records and three
checkpoint types (soft-CE-best, accuracy-best, and last), with state epoch,
configuration, initial-state hash, and checkpoint SHA-256.

From `validation_round1/dis`, the training command was:

```bash
./.venv/bin/python round2_resnet/src/train_dropout_multiseed_confirmation.py
```
