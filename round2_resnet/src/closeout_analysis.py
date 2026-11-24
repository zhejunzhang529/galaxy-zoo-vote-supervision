#!/usr/bin/env python3
"""Post-hoc diagnostics from frozen DIS labels and predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW = ["raw_smooth_normalized","raw_features_or_disk_normalized","raw_star_or_artifact_normalized"]
WEIGHTED = ["weighted_smooth_normalized","weighted_features_or_disk_normalized","weighted_star_or_artifact_normalized"]

def multiclass_metrics(probability: np.ndarray, soft: np.ndarray, hard: np.ndarray) -> dict:
    """Calculate probability, hard-label, confusion, and per-class metrics."""
    prediction = probability.argmax(axis=1)
    confusion = np.zeros((3,3), dtype=int)
    for truth, predicted in zip(hard, prediction):
        confusion[int(truth), int(predicted)] += 1
    per_class = []
    for cls in range(3):
        tp = confusion[cls,cls]; fp = confusion[:,cls].sum()-tp; fn = confusion[cls,:].sum()-tp
        precision = float(tp/(tp+fp)) if tp+fp else None
        recall = float(tp/(tp+fn)) if tp+fn else None
        f1 = float(2*precision*recall/(precision+recall)) if precision is not None and recall is not None and precision+recall else None
        per_class.append({"class":cls,"precision":precision,"recall":recall,"f1":f1,"support":int(confusion[cls].sum()),
                          "note":"recall/F1 not estimable without true positives" if recall is None else ""})
    return {"n":int(len(hard)),
            "cross_entropy":float(np.mean(-np.sum(soft*np.log(np.clip(probability,1e-12,1.0)),axis=1))),
            "brier":float(np.mean(np.sum((probability-soft)**2,axis=1))),
            "accuracy":float(np.mean(prediction==hard)),
            "predicted_class_counts":np.bincount(prediction,minlength=3).astype(int).tolist(),
            "confusion_matrix_true_rows_predicted_columns":confusion.tolist(),"per_class":per_class}

def constant_baseline(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    """Evaluate train-only soft-mean and hard-frequency constant probabilities."""
    soft = train[RAW].to_numpy(float).mean(axis=0)
    hard = np.bincount(train.hard_class.to_numpy(int),minlength=3).astype(float); hard /= hard.sum()
    nat = evaluation.pool_role.eq("new_natural_evaluation").to_numpy()
    target = evaluation.loc[nat,RAW].to_numpy(float); labels = evaluation.loc[nat,"hard_class"].to_numpy(int)
    return {"training_rows":int(len(train)),"soft_mean_probability":soft.tolist(),"hard_frequency_probability":hard.tolist(),
            "soft_mean_natural_metrics":multiclass_metrics(np.tile(soft,(len(target),1)),target,labels),
            "hard_frequency_natural_metrics":multiclass_metrics(np.tile(hard,(len(target),1)),target,labels)}

def main() -> None:
    """Write saved-prediction diagnostics without images, checkpoints, or model inference."""
    root = Path(__file__).resolve().parents[2]; out = root/"round2_resnet/results/closeout"; out.mkdir(parents=True,exist_ok=True)
    split = pd.read_csv(root/"round2_resnet/results/resnet_split.csv",dtype={"dr7objid":str,"pool_role":str})
    evaluation = pd.read_csv(root/"round2_resnet/results/new_eval_with_images.csv",dtype={"dr7objid":str,"pool_role":str})
    saved = json.loads((root/"round2_resnet/results/formal_evaluation.json").read_text())
    natural = evaluation.pool_role.eq("new_natural_evaluation").to_numpy(); diagnostic = evaluation.pool_role.eq("new_class2_diagnostic_recall_only").to_numpy()
    raw = evaluation[RAW].to_numpy(float); weighted = evaluation[WEIGHTED].to_numpy(float); hard = evaluation.hard_class.to_numpy(int)
    baselines = {}
    for size, flag in [("small_3100","small_train"),("large_10000","large_train")]:
        baselines[size] = constant_baseline(split[split[flag]],evaluation)
    (out/"scale_matched_constant_baselines.json").write_text(json.dumps({"status":"post_hoc_saved_labels_only","baselines":baselines},indent=2)+"\n")

    nat_target,nat_hard = raw[natural],hard[natural]
    entropy = -np.sum(np.where(raw>0,raw*np.log(np.clip(raw,1e-12,1.0)),0.0),axis=1)
    ordered=np.sort(raw,axis=1); margin=ordered[:,-1]-ordered[:,-2]
    entropy_cut=float(np.median(entropy[natural])); margin_cut=float(np.median(margin[natural]))
    groups=np.full(len(evaluation),"outside_natural",dtype=object)
    groups[natural]=np.where(entropy[natural]<=entropy_cut,"entropy_low","entropy_high")+"+"+np.where(margin[natural]<=margin_cut,"margin_low","margin_high")
    rows=[]
    for record in saved["records"]:
        probability=np.asarray(record["natural_evaluation"]["probabilities"],float)
        for group in sorted(set(groups[natural])):
            mask=groups[natural]==group
            rows.append({"train_size":record["train_size"],"target":record["target"],"seed":record["seed"],"group":group,
                         "entropy_median_cut":entropy_cut,"margin_median_cut":margin_cut,**multiclass_metrics(probability[mask],nat_target[mask],nat_hard[mask])})
    pd.DataFrame(rows).to_csv(out/"vote_disagreement_group_metrics.csv",index=False)
    pd.DataFrame({"dr7objid":evaluation.dr7objid[natural],"entropy":entropy[natural],"top_two_vote_margin":margin[natural],"group":groups[natural]}).to_csv(out/"natural_vote_disagreement.csv",index=False)

    weighted_rows=[]
    for record in saved["records"]:
        probability=np.asarray(record["natural_evaluation"]["probabilities"],float)
        mr=multiclass_metrics(probability,raw[natural],hard[natural]); mw=multiclass_metrics(probability,weighted[natural],hard[natural])
        weighted_rows.append({"train_size":record["train_size"],"target":record["target"],"seed":record["seed"],
                              "raw_target_ce":mr["cross_entropy"],"weighted_target_ce":mw["cross_entropy"],
                              "raw_target_brier":mr["brier"],"weighted_target_brier":mw["brier"],
                              "interpretation":"post-evaluation target sensitivity; not a weighted-target training run"})
    pd.DataFrame(weighted_rows).to_csv(out/"weighted_target_sensitivity.csv",index=False)

    diag=evaluation[diagnostic].reset_index(drop=True); detail=[]
    for record in saved["records"]:
        p=np.asarray(record["class2_diagnostic_recall_only"]["probabilities"],float)
        for i,row in diag.iterrows():
            votes=[float(row[c]) for c in RAW]; detail.append({"dr7objid":str(row.dr7objid),"asset_id":int(row.asset_id),"raw_votes":votes,
                "vote_margin":float(np.sort(votes)[-1]-np.sort(votes)[-2]),"train_size":record["train_size"],"target":record["target"],"seed":int(record["seed"]),
                "predicted_probabilities":p[i].tolist(),"predicted_class":int(p[i].argmax()),"true_class":int(row.hard_class),"destination":int(p[i].argmax())})
    pd.DataFrame(detail).to_csv(out/"class2_five_object_diagnostics.csv",index=False)
    dev_n=len(list((root/"data/processed/resnet_round2_images").glob("*.jpg"))); eval_n=len(list((root/"data/processed/resnet_round2_eval_images").glob("*.jpg")))
    (out/"resource_estimate.json").write_text(json.dumps({"source":"round2_resnet/results/throughput.json","decoded_float32_bytes_per_image":128*128*3*4,
        "batch_16_decoded_float32_mib":128*128*3*4*16/2**20,"throughput_boundary":"3 warm-up plus 12 timed forward batches including resize/normalization and target loading; not end-to-end training throughput",
        "peak_memory":"not measured; values are decoded-tensor estimates only","image_counts":{"development_cache":dev_n,"evaluation_cache":eval_n}},indent=2)+"\n")
    (out/"analysis_scope.md").write_text("Post-hoc calculations from formal_evaluation.json, new_eval_with_images.csv, and frozen labels. Not used to select a checkpoint, threshold, resolution, training size, or hyperparameter. Entropy/margin groups use natural-pool medians. The five rows are the complete residual star-or-artifact diagnostic pool: case descriptions and recall only, not population-level precision or mechanism claims. Weighted-target columns change the evaluation target; they are not a weighted-target training experiment.\n")
    print(json.dumps({"output":str(out),"group_rows":len(rows),"diagnostic_rows":len(detail),"status":"saved_predictions_only"},indent=2))

if __name__ == "__main__":
    main()

