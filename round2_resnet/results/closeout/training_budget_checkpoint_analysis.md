# Training-target and checkpoint-selection analysis

This is an inference-only analysis of the four existing 20-epoch diagnostic
runs. It evaluates their four saved best checkpoints and four saved last
checkpoints on the same `small_3100` training subset and fixed 750-object
validation set. No checkpoint was replaced, no selection rule was changed, and
no original, extra, pilot, or other independent evaluation pool was loaded.

## Selection-rule sensitivity

The original rule remains minimum validation soft CE, followed by calibration
discrepancy and earliest epoch. The accuracy rows below are descriptive only;
ties in maximum accuracy use the earliest epoch and do not create new
checkpoints.

| lr | target | original soft-CE epoch: CE / accuracy / Brier | earliest max-accuracy epoch: accuracy / CE / Brier |
|---:|---|---:|---:|
| `3e-4` | hard | e1: 0.8676 / 0.8653 / 0.0869 | e18: 0.8867 / 1.9591 / 0.1478 |
| `3e-4` | soft | e19: 0.6251 / 0.8773 / 0.0426 | e11: 0.8827 / 0.6266 / 0.0445 |
| `1e-4` | hard | e1: 0.8231 / 0.8640 / 0.0903 | e7: 0.8787 / 1.4564 / 0.1421 |
| `1e-4` | soft | e11: 0.6288 / 0.8800 / 0.0471 | e7: 0.8800 / 0.6321 / 0.0488 |

The largest disagreement is in the hard runs: later accuracy peaks occur with
substantially worse soft CE and Brier. In the soft runs the two criteria are
closer, but they still select different epochs. This confirms that the
reported best checkpoints answer a probability-fit question, not an
accuracy-maximisation question.

## Existing checkpoint metrics

All values below are recomputed in eval mode. `hard CE` uses a hard one-hot
target; `soft CE` uses the normalized raw vote vector. `max p` is the mean of
the largest predicted class probability, and `entropy` is the mean predictive
entropy `-sum_c p_c log p_c`.

| lr | target | checkpoint | split | hard CE | soft CE | accuracy | mean max p | mean entropy |
|---:|---|---|---|---:|---:|---:|---:|---:|
| `3e-4` | hard | best e1 | train | 0.1772 | 0.8262 | 0.9387 | 0.8912 | 0.2691 |
| `3e-4` | hard | best e1 | validation | 0.2994 | 0.8676 | 0.8653 | 0.8780 | 0.2911 |
| `3e-4` | hard | last e20 | train | 0.0018 | 2.4895 | 0.9997 | 0.9985 | 0.0068 |
| `3e-4` | hard | last e20 | validation | 0.6326 | 2.1351 | 0.8760 | 0.9693 | 0.0758 |
| `3e-4` | soft | best e19 | train | 0.2677 | 0.5461 | 0.9777 | 0.7802 | 0.5605 |
| `3e-4` | soft | best e19 | validation | 0.3654 | 0.6251 | 0.8773 | 0.7536 | 0.6077 |
| `3e-4` | soft | last e20 | train | 0.2673 | 0.5478 | 0.9732 | 0.7813 | 0.5628 |
| `3e-4` | soft | last e20 | validation | 0.3726 | 0.6290 | 0.8773 | 0.7512 | 0.6118 |
| `1e-4` | hard | best e1 | train | 0.1432 | 0.7931 | 0.9610 | 0.9042 | 0.2732 |
| `1e-4` | hard | best e1 | validation | 0.3199 | 0.8231 | 0.8640 | 0.8727 | 0.3236 |
| `1e-4` | hard | last e20 | train | 0.0008 | 2.4223 | 1.0000 | 0.9992 | 0.0040 |
| `1e-4` | hard | last e20 | validation | 0.6105 | 1.9561 | 0.8747 | 0.9699 | 0.0782 |
| `1e-4` | soft | best e11 | train | 0.2717 | 0.5492 | 0.9632 | 0.7798 | 0.5706 |
| `1e-4` | soft | best e11 | validation | 0.3861 | 0.6288 | 0.8800 | 0.7414 | 0.6291 |
| `1e-4` | soft | last e20 | train | 0.2571 | 0.5454 | 0.9713 | 0.7893 | 0.5462 |
| `1e-4` | soft | last e20 | validation | 0.3764 | 0.6288 | 0.8707 | 0.7516 | 0.6034 |

## Interpretation

The hard-target runs fit the hard argmax labels increasingly sharply: by epoch
20 their training hard CE is below `0.002`, accuracy is approximately `1.0`,
mean maximum probability is about `0.999`, and entropy is near zero. On
validation, hard CE is about `0.61--0.63`, while soft CE is `1.96--2.14` and
the predicted probabilities remain much sharper than the vote distributions.
Together with the training/validation hard-accuracy and hard-CE gaps, this is
consistent with overfitting to the hard supervision target. It does not by
itself identify whether the cause is learning rate, finite data, model
capacity, or the target definition.

The soft-target runs retain much higher predictive entropy and lower maximum
probabilities. Their train/validation soft-CE gaps are about `0.08` at the
best checkpoints and remain similar at epoch 20; validation soft CE is also
stable to within a few thousandths between the selected and last checkpoints.
This supports better fitting of the observed vote distribution under this
protocol, while hard accuracy still has a separate train/validation gap. It is
not evidence that the vote distribution is physical truth or that soft
supervision is universally superior.

The comparison also separates three quantities: hard-label fit, vote-
distribution fit, and generalisation between the fixed training and validation
rows. The original training losses are not directly comparable because hard
and soft runs optimize different targets. The checkpoint analysis uses common
eval-mode definitions for both targets.

## Reproducibility and cost

The machine-readable output is
`training_budget_checkpoint_analysis.json`. The inference script is
`src/analyze_training_budget_checkpoints.py`. From
`validation_round1/dis`, the command was:

```bash
./.venv/bin/python round2_resnet/src/analyze_training_budget_checkpoints.py
```

The reported checkpoint evaluation loop took 185.64 s on the local CPU
environment. The output records the device, split, checkpoint hashes, object
counts, and per-checkpoint timing. The source training JSON and all original
checkpoint files remain unchanged.
