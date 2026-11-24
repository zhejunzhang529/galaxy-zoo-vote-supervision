#!/usr/bin/env python3
"""Audit the twelve frozen ResNet runs and their saved evaluation evidence.

This is separate from the historical TinyCNN audit. It verifies metadata,
splits, probabilities, recomputed metrics, and paired summaries. Checkpoint
binary verification is optional: a review package without weights reports it as
SKIPPED rather than treating absence as a pass.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW=["raw_smooth_normalized","raw_features_or_disk_normalized","raw_star_or_artifact_normalized"]

def sha256(path: Path) -> str:
    """Hash one available file for manifest comparison."""
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def close(a: float, b: float, tol: float=1e-6) -> bool:
    """Compare saved floating metrics within a reproducibility tolerance."""
    return abs(float(a)-float(b)) <= tol*max(1.0,abs(float(a)),abs(float(b)))

def root_from_script() -> Path:
    """Locate either the original repository or the flattened review package."""
    here=Path(__file__).resolve().parent
    candidate=here.parents[2]
    return candidate if (candidate/"round2_resnet").exists() else here.parent

def metric_check(probability: np.ndarray, frame: pd.DataFrame, mask: np.ndarray, saved: dict) -> bool:
    """Recompute saved natural-pool metrics from probabilities and labels."""
    p=probability; target=frame.loc[mask,RAW].to_numpy(float); hard=frame.loc[mask,"hard_class"].to_numpy(int)
    pred=p.argmax(axis=1); confusion=np.zeros((3,3),dtype=int)
    for truth,guess in zip(hard,pred): confusion[int(truth),int(guess)]+=1
    ce=float(np.mean(-np.sum(target*np.log(np.clip(p,1e-12,1.0)),axis=1)))
    brier=float(np.mean(np.sum((p-target)**2,axis=1)))
    return (close(ce,saved["soft_cross_entropy"]) and close(brier,saved["soft_brier"]) and
            close(float(np.mean(pred==hard)),saved["hard_accuracy"]) and
            saved["target_class_counts"]==np.bincount(hard,minlength=3).astype(int).tolist() and
            saved["predicted_class_counts"]==np.bincount(pred,minlength=3).astype(int).tolist() and
            saved["confusion_matrix_true_rows_predicted_columns"]==confusion.tolist())

def main() -> int:
    """Run required checks, write JSON evidence, and return nonzero on failure."""
    root=root_from_script(); base=root/"round2_resnet" if (root/"round2_resnet").exists() else root
    result_path=base/"results/resnet_package_audit.json"
    config=json.loads((base/"configs/formal.json").read_text())
    runs=json.loads((base/"results/formal_runs.json").read_text())
    evaluation=json.loads((base/"results/formal_evaluation.json").read_text())
    summary=json.loads((base/"results/formal_summary.json").read_text())
    split=pd.read_csv(base/"results/resnet_split.csv",dtype={"dr7objid":str,"pool_role":str})
    eval_split_path=base/"results/new_eval_with_images.csv"
    if not eval_split_path.exists(): eval_split_path=root/"results/new_eval_with_images.csv"
    eval_split=pd.read_csv(eval_split_path,dtype={"dr7objid":str,"pool_role":str})
    required=[(size,target,int(seed)) for size in config["train_sizes"] for target in config["targets"] for seed in config["seeds"]]
    run_keys={(r["train_size"],r["target"],int(r["seed"])) for r in runs["records"]}
    eval_keys={(r["train_size"],r["target"],int(r["seed"])) for r in evaluation["records"]}
    checks={}
    checks["configuration_complete"]={"status":"PASS" if len(required)==12 and set(required)==run_keys==eval_keys else "FAIL","expected_runs":required,"run_count":len(run_keys),"evaluation_count":len(eval_keys)}
    checks["selected_epoch_records"]={"status":"PASS","failures":[]}
    for r in runs["records"]:
        if not isinstance(r.get("selected_epoch"),int) or r.get("selected_epoch")!=r.get("checkpoint_state_epoch"):
            checks["selected_epoch_records"]["failures"].append(r.get("checkpoint"))
    if checks["selected_epoch_records"]["failures"]: checks["selected_epoch_records"]["status"]="FAIL"
    checks["split_hash"]={"status":"PASS" if sha256(base/"results/resnet_split.csv")==json.loads((base/"results/split_manifest.json").read_text())["output_split_sha256"] else "FAIL"}
    checks["split_roles_and_counts"]={"status":"PASS" if len(split)==20755 and set(split.pool_role.dropna()) >= {"fixed_validation","new_natural_evaluation","new_class2_diagnostic_recall_only"} else "FAIL","rows":len(split)}
    checks["evaluation_rows_and_ids"]={"status":"PASS" if len(eval_split)==10005 and eval_split.dr7objid.is_unique and eval_split.image_path.is_unique else "FAIL","rows":len(eval_split)}
    run_map={(r["train_size"],r["target"],int(r["seed"])):r for r in runs["records"]}
    eval_failures=[]; hash_failures=[]; metric_failures=[]; probability_failures=[]
    natural=eval_split.pool_role.eq("new_natural_evaluation").to_numpy()
    for r in evaluation["records"]:
        key=(r["train_size"],r["target"],int(r["seed"])); p=np.asarray(r["natural_evaluation"]["probabilities"],float)
        if p.shape!=(int(natural.sum()),3) or not np.all(np.isfinite(p)) or not np.allclose(p.sum(axis=1),1.0,atol=1e-6) or np.any(p<0) or np.any(p>1): probability_failures.append(key)
        if not metric_check(p,eval_split,natural,r["natural_evaluation"]): metric_failures.append(key)
        if key in run_map and r.get("checkpoint_sha256")!=run_map[key].get("checkpoint_sha256"): hash_failures.append(key)
    checks["probabilities"]={"status":"PASS" if not probability_failures else "FAIL","failures":probability_failures}
    checks["evaluation_metrics_recompute"]={"status":"PASS" if not metric_failures else "FAIL","failures":metric_failures}
    checks["training_evaluation_hash_metadata"]={"status":"PASS" if not hash_failures else "FAIL","failures":hash_failures}
    checks["paired_summary"]={"status":"PASS" if len(summary.get("paired_soft_minus_hard",[]))==6 and len(summary.get("paired_mean_and_sample_sd",[]))==6 else "FAIL","paired_rows":len(summary.get("paired_soft_minus_hard",[]))}
    checkpoint_checks=[]
    for r in runs["records"]:
        path=root/r["checkpoint"] if (root/r["checkpoint"]).exists() else base.parent/r["checkpoint"]
        if path.exists():
            checkpoint_checks.append({"checkpoint":r["checkpoint"],"status":"PASS" if sha256(path)==r["checkpoint_sha256"] else "FAIL"})
        else:
            checkpoint_checks.append({"checkpoint":r["checkpoint"],"status":"SKIPPED","reason":"checkpoint binary excluded from review package"})
    checks["checkpoint_binary_hashes"]={"status":"PASS" if checkpoint_checks and all(x["status"]=="PASS" for x in checkpoint_checks) else ("SKIPPED" if all(x["status"]=="SKIPPED" for x in checkpoint_checks) else "FAIL"),"items":checkpoint_checks}
    checks["historical_tinycnn_audit"]={"status":"NOT_USED","reason":"This audit covers the twelve ResNet runs only; historical TinyCNN evidence remains separate."}
    required_status=[v["status"] for k,v in checks.items() if k not in {"checkpoint_binary_hashes","historical_tinycnn_audit"}]
    overall=all(s=="PASS" for s in required_status)
    source_label = "original-local-repository/DIS" if (root/"round2_resnet").exists() else "review-package/DIS"
    output={"status":"resnet_round2_package_audit","overall_pass":overall,"source_root":source_label,"checks":checks}
    result_path.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"overall_pass":overall,"output":str(result_path),"checkpoint_status":checks["checkpoint_binary_hashes"]["status"]},indent=2))
    return 0 if overall else 1

if __name__=="__main__":
    raise SystemExit(main())

