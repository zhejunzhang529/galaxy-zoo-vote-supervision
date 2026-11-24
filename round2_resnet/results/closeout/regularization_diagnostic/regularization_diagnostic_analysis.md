# DIS finite regularization comparison

This diagnostic compares two new configurations at the fixed learning rate
`3e-4` with the existing 20-epoch reference. All runs use seed `20261119`,
`small_3100`, the fixed 750-object validation split, ResNet-18 at 128 px,
batch size 32, Adam, `layer4+head`, no scheduler, no augmentation, and no early
stopping. The reference is reused from
`results/closeout/training_budget_diagnostic_runs.json`; it was not rerun.

The new configurations are:

- A: `weight_decay=1e-3`, `dropout=0`.
- B: `weight_decay=1e-4`, `dropout=0.2`, with dropout after flattening and
  before the original three-output Linear classifier.

## New run summary

| configuration | target | seconds | soft-CE best epoch | accuracy best epoch | validation soft CE at epoch 20 |
|---|---|---:|---:|---:|---:|
| A: `wd=1e-3` | hard | 949.79 | 1 | 12 | 1.9525 |
| A: `wd=1e-3` | soft | 979.33 | 13 | 13 | 0.6381 |
| B: `dropout=0.2` | hard | 966.55 | 1 | 11 | 2.1498 |
| B: `dropout=0.2` | soft | 964.59 | 13 | 19 | 0.6240 |

The four new runs took 3,860.26 s (64.34 min), within the pre-run estimate of
3,800--4,600 s. The first launch stopped before training because the script
missed an import; no epoch or checkpoint was produced in that attempt. The
corrected launch completed all four runs without non-finite loss or runtime
failure.

## Reference comparison

The table uses the original soft-CE-selected checkpoint and the epoch-20 last
checkpoint. Accuracy-selected epochs remain separate in the JSON and are not
used to replace the reference selection.

| target | configuration | soft-CE best: epoch / CE / Brier / accuracy | last: CE / Brier / accuracy |
|---|---|---:|---:|
| hard | reference `wd=1e-4, dropout=0` | 1 / 0.8676 / 0.0869 / 0.8653 | 2.1351 / 0.1551 / 0.8760 |
| hard | A `wd=1e-3` | 1 / 0.8754 / 0.0890 / 0.8680 | 1.9525 / 0.1655 / 0.8613 |
| hard | B `dropout=0.2` | 1 / 0.8769 / 0.0867 / 0.8693 | 2.1498 / 0.1570 / 0.8760 |
| soft | reference `wd=1e-4, dropout=0` | 19 / 0.6251 / 0.0426 / 0.8773 | 0.6290 / 0.0446 / 0.8773 |
| soft | A `wd=1e-3` | 13 / 0.6257 / 0.0442 / 0.8880 | 0.6381 / 0.0514 / 0.8760 |
| soft | B `dropout=0.2` | 13 / 0.6227 / 0.0415 / 0.8853 | 0.6240 / 0.0411 / 0.8840 |

## Interpretation

For hard supervision, A reduces training hard-label fitting relative to the
reference: at epoch 20 its train hard CE is `0.0260` and accuracy `0.9910`,
compared with `0.0018` and `0.9997` for the reference. Its validation hard CE
is `0.6424`, so the hard-label gap remains; validation soft CE is slightly
lower than the reference at epoch 20, but Brier is higher and accuracy is
lower. This is a trade-off, not evidence that stronger weight decay resolves
the probability-fit problem.

B does not materially prevent sharp hard-label fitting in this run. At epoch
20 its train hard CE is `0.0008`, accuracy `0.9997`, mean validation maximum
probability `0.9701`, and validation entropy `0.0715`; the reference values are
`0.0018`, `0.9997`, `0.9693`, and `0.0758`. Dropout therefore does not make the
hard predictions less concentrated under this single-seed protocol.

For soft supervision, A has a similar selected validation soft CE to the
reference but a worse epoch-20 CE and Brier. Its accuracy is higher at the
selected epoch, illustrating the intended probability-fit versus accuracy
trade-off. B gives the lowest new soft CE and Brier at its selected epoch and
remains slightly better than the reference at epoch 20. Its validation maximum
probability at epoch 20 is `0.7703` and entropy `0.5787`, compared with `0.7512`
and `0.6118` for the reference, so the improvement does not correspond to
uniformly less concentrated predictions.

The hard and soft training losses are different objectives. In addition,
dropout training loss is measured with dropout enabled whereas all train and
validation fit metrics are eval-mode measurements with dropout disabled. Their
difference must therefore not be interpreted as a direct generalisation gap.
The results support a finite, single-seed comparison of two regularization
choices at one learning rate; they do not establish a generally effective
regularizer or a fully tuned hard/soft comparison.

## Reproducibility and data boundary

The machine-readable output is
`regularization_diagnostic_runs.json`. It records every epoch's target loss,
eval-mode train/validation hard and soft CE, Brier, accuracy, mean maximum
probability, entropy, calibration discrepancy, timing, and the three
checkpoint types per new run. The reference's older JSON does not contain
per-epoch hard CE, maximum probability, or entropy; those quantities were not
interpolated or regenerated for the reference.

The comparison figure is `regularization_comparison_curves.png` with an SVG
counterpart. It was generated from the saved reference and new histories by
`src/plot_regularization_comparison.py`.

From `validation_round1/dis`, the new training command was:

```bash
./.venv/bin/python round2_resnet/src/train_regularization_diagnostic.py
```

No original, extra, pilot, or other independent evaluation pool was loaded or
scored. The formal evaluation protocol and all historical results remain
unchanged.
