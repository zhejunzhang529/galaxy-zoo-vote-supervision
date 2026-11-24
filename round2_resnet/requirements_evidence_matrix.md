# DIS round-2 evidence matrix

| Requirement | Evidence | Status | Limitation and effect |
|---|---|---|---|
| Scientific motivation from GZ2 and DECaLS | `reports/DIS_resnet_stage_report.md`, DOI links in the report | completed | Vote fractions remain response distributions, not physical truth |
| Scientific interpretation of disagreement | `README.md`, `reports/DIS_resnet_stage_report.md`, `reports/DIS_resnet_stage_report_zh.md` | completed | Resolution, intermediate morphology, and classifier variation are possible explanations; entropy alone is not causal evidence |
| Hardware and throughput check | `results/throughput.json`, `configs/throughput.json` | completed | CPU only; MPS was built but unavailable on this machine |
| ResNet-18, resolution, and preprocessing check | `src/benchmark_resnet.py`, `src/train_resnet.py`, `results/throughput.json` | completed | 128 pixels selected by measured CPU capacity; 224 pixels is a tested alternative, not a formal comparison |
| Registered 8-way short grid | `configs/grid.json`, `results/grid_short.json` | completed | Short budget only |
| Frozen object/image-isolated split | `configs/split.json`, `results/resnet_split.csv`, `results/split_manifest.json` | completed | Natural target is the unseen original-sample eligible pool, not all galaxies or another survey |
| Paired hard/soft formal feasibility | `configs/formal.json`, `results/formal_runs.json` | completed | Three seeds and three epochs; not a final production training campaign |
| Checkpoint selection | `results/formal_runs.json`, checkpoint SHA256 values | completed after repair | Initial final-epoch saving error is isolated in `results/invalid_final_epoch_attempt/` |
| Frozen independent evaluation | `results/new_eval_image_manifest.json`, `results/formal_evaluation.json` | completed | Evaluation was viewed after selection and cannot support later tuning |
| Baselines and uncertainty | `results/formal_summary.json`, `results/paired_seed_summary.csv` | completed | Bootstrap intervals measure object resampling; seed sample SD is separate and is not a CI |
| Data-intensive evidence | `src/audit_data_intensive.py`, `results/data_intensive_audit.json`, `results/throughput.json` | completed | Archive, label, image-byte, training-time and processing boundaries are recorded; peak memory remains unmeasured |
| Rare class analysis | `formal_summary.json`, `probability_vs_recall.svg` | partial | Only five unseen class-2 objects; recall only, no precision/F1 or natural class-2 estimate |
| GZ2 extra source audit | `src/prepare_extra_cross_sample.py`, `results/extra_cross_sample_manifest.json`, `results/extra_cross_sample_split.csv` | completed | 21,836 eligible extra objects; fixed 1,000-object no-quota sample; extra is a documented GZ2 selection difference, not proof of OOD |
| Original-to-extra cross-sample inference | `src/{benchmark_extra_inference.py,evaluate_extra_cross_sample.py}`, `results/extra_cross_sample_results.json`, `results/extra_cross_sample_predictions.csv` | completed with evidence limits | Six frozen large_10000 checkpoints, no retraining or selection; only four class-2 objects in extra, so rare-class metrics are descriptive |
| Cross-sample analysis and figure | `src/plot_extra_cross_sample.py`, `results/extra_cross_sample_pool_comparison.json`, `results/figures/extra_cross_sample_generalisation.svg` | completed | Compares composition, vote entropy/margin and frozen-seed CE; does not establish OOD, cross-survey generalization, or physical-truth calibration |
