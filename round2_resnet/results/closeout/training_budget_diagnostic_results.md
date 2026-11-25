# DIS extended training diagnostic: results

This is a single-seed diagnostic, not the final hard/soft performance
comparison. It uses the fixed `small_3100` training subset, the existing 750
object validation split, ResNet-18 at 128 px, batch size 32, Adam, weight decay
`1e-4`, `layer4+head`, no scheduler, and no early stopping. The four runs use
seed `20261119` and fresh copies of the same ImageNet-pretrained initial state.
The training order is identical across target and learning-rate settings.

## Run summary

| learning rate | target | seconds | selected epoch | validation soft CE at selected epoch | validation soft CE at epoch 20 |
|---:|---|---:|---:|---:|---:|
| `3e-4` | hard | 967.21 | 1 | 0.8676 | 2.1351 |
| `3e-4` | soft | 984.09 | 19 | 0.6251 | 0.6290 |
| `1e-4` | hard | 989.74 | 1 | 0.8231 | 1.9561 |
| `1e-4` | soft | 939.98 | 11 | 0.6288 | 0.6288 |

The four runs took 3,881.02 s in total (64.68 min). The pre-run estimate was
600--840 s per run, or 2,400--3,360 s in total; it underestimated the actual
CPU cost. The result file records the environment and the estimate separately.

Each run has 20 epoch records. The target-specific training loss is hard
one-hot cross-entropy for hard training and normalized vote cross-entropy for
soft training. For comparison, `train_eval_soft_cross_entropy` is computed in
eval mode after the epoch using the soft vote target; it is not the same loss
as the in-training hard-target loss. Validation soft CE, Brier score,
calibration discrepancy, and accuracy use the same fixed 750-object validation
set at every epoch.

## What the curves show

- Both hard runs reduce their in-training one-hot loss to approximately zero,
  but their eval-mode train soft CE and validation soft CE rise after epoch 1.
  Under this protocol, the hard-target runs therefore show an early validation
  optimum and subsequent separation between optimization loss and soft-target
  validation behaviour. Accuracy fluctuates rather than providing the selection
  criterion.
- Both soft runs continue to reduce their target loss and eval-mode train soft
  CE. Validation soft CE improves overall but fluctuates: the `3e-4` run reaches
  its minimum at epoch 19, while the `1e-4` run reaches its minimum at epoch 11
  and is almost unchanged by epoch 20. This is evidence that the three-epoch
  budget could miss later soft-target improvement, but it is not evidence that
  either learning rate is globally preferable.
- The early soft ranking favours `3e-4` (`0.6453` versus `0.6524` at epoch 1),
  whereas the epoch-20 values are essentially tied (`0.6290` versus `0.6288`).
  The lower learning rate changes the late trajectory slightly, but this
  single-seed difference is not a stable ranking claim.
- The hard and soft training losses have different targets and should not be
  compared by their absolute values. The eval-mode train soft CE provides the
  comparable train-side probability-fit diagnostic; even then, this remains a
  single fixed split and seed.

The full curves are in
`training_budget_curves.png` and `training_budget_curves.svg`. The machine-
readable source is
`training_budget_diagnostic_runs.json`; the configuration and implementation
are `configs/training_budget_diagnostic.json` and
`src/train_budget_diagnostic.py`.

## Boundary and remaining uncertainty

No original, extra, historical pilot, or other independent evaluation pool was
loaded or scored. The results do not establish statistical stability across
seeds, sufficient convergence beyond 20 epochs, or a final hard/soft ranking.
They also do not justify changing the frozen formal comparison protocol. Any
future extension should be specified separately after reviewing these curves.

## Reproduction

From `validation_round1/dis`, the training command was:

```bash
./.venv/bin/python round2_resnet/src/train_budget_diagnostic.py \
  --config round2_resnet/configs/training_budget_diagnostic.json
```

This command requires the local image cache, the split manifest, and the
ImageNet weight cache; it is not a no-data reproduction command. To redraw the
curves from the saved JSON only, use system Python with an existing
Matplotlib installation (the project venv has no Matplotlib):

```bash
MPLCONFIGDIR=/tmp/dis_mpl_config MPLBACKEND=Agg \
python3 round2_resnet/src/plot_training_budget_diagnostic.py \
  --input round2_resnet/results/closeout/training_budget_diagnostic_runs.json \
  --output round2_resnet/results/closeout/training_budget_curves.png
```
