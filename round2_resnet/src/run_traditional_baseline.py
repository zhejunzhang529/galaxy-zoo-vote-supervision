#!/usr/bin/env python3
"""Fit one fixed low-resolution RGB multinomial logistic baseline."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize

RAW=["raw_smooth_normalized","raw_features_or_disk_normalized","raw_star_or_artifact_normalized"]

def image_path(root: Path, row: pd.Series) -> Path:
    """Resolve an existing processed image for one split row."""
    rel=str(row.image_path)
    return root/"data/processed"/rel

def features(root: Path, frame: pd.DataFrame, size: int, mean: np.ndarray|None=None, scale: np.ndarray|None=None):
    """Decode RGB images, resize to a fixed square, and standardize train-only."""
    values=[]
    for row in frame.itertuples(index=False):
        with Image.open(image_path(root,row)) as image:
            values.append(np.asarray(image.convert("RGB").resize((size,size)),dtype=np.float32).reshape(-1)/255.0)
    x=np.asarray(values,dtype=np.float64)
    if mean is None:
        mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-6]=1.0
    return (x-mean)/scale, mean, scale

def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable row-wise softmax."""
    z=logits-logits.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def objective_and_gradient(flat: np.ndarray, design: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    """Return the mean CE plus mean-weight L2 objective and its gradient."""
    n, design_width = design.shape
    k = 3
    w = flat.reshape(k, design_width)
    probability = softmax(design @ w.T)
    value = -np.mean(np.log(np.clip(probability[np.arange(n), y], 1e-12, 1.0)))
    regularizer = 0.5 * l2 * np.mean(w[:, 1:] ** 2)
    gradient = ((probability - np.eye(k)[y]).T @ design) / n
    gradient[:, 1:] += l2 * w[:, 1:] / w[:, 1:].size
    return float(value + regularizer), gradient.ravel()


def finite_difference_gradient_check(seed: int = 20261120) -> dict:
    """Check the analytic gradient on a random small matrix before fitting."""
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(7, 6))
    labels = rng.integers(0, 3, size=7)
    point = rng.normal(scale=0.2, size=18)
    l2 = 0.7
    _, analytic = objective_and_gradient(point, design, labels, l2)
    epsilon = 1e-6
    numeric = np.empty_like(point)
    for index in range(point.size):
        plus, _ = objective_and_gradient(point + np.eye(1, point.size, index).ravel() * epsilon, design, labels, l2)
        minus, _ = objective_and_gradient(point - np.eye(1, point.size, index).ravel() * epsilon, design, labels, l2)
        numeric[index] = (plus - minus) / (2 * epsilon)
    max_abs = float(np.max(np.abs(analytic - numeric)))
    relative = float(max_abs / max(1.0, np.max(np.abs(numeric))))
    result = {"seed": seed, "epsilon": epsilon, "max_absolute_error": max_abs, "max_relative_error": relative, "passed": bool(relative < 1e-6)}
    if not result["passed"]:
        raise RuntimeError(f"finite-difference gradient check failed: {result}")
    return result


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float=1.0, maxiter: int = 1000) -> tuple[np.ndarray, object]:
    """Fit the fixed L2-regularized multinomial logistic model."""
    n, p = x.shape
    design = np.column_stack([np.ones(n), x])
    initial = np.zeros(3 * (p + 1))
    result = minimize(lambda z: objective_and_gradient(z, design, y, l2), initial, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-8})
    return result.x.reshape(3, p + 1), result

def metric(prob: np.ndarray, frame: pd.DataFrame) -> dict:
    """Report metrics against raw votes and hard argmax labels."""
    target=frame[RAW].to_numpy(float); hard=frame.hard_class.to_numpy(int); pred=prob.argmax(axis=1)
    return {"n":int(len(frame)),"cross_entropy":float(np.mean(-np.sum(target*np.log(np.clip(prob,1e-12,1)),axis=1))),
            "brier":float(np.mean(np.sum((prob-target)**2,axis=1))),"accuracy":float(np.mean(pred==hard)),
            "predicted_class_counts":np.bincount(pred,minlength=3).astype(int).tolist()}

def main() -> None:
    """Run one preregistered 16-pixel RGB baseline and save a post-hoc supplement."""
    root=Path(__file__).resolve().parents[2]; split=pd.read_csv(root/"round2_resnet/results/resnet_split.csv",dtype={"dr7objid":str,"image_path":str,"pool_role":str})
    train=split[split.small_train].copy(); validation=split[split.pool_role.eq("fixed_validation")].copy()
    evaluation=pd.read_csv(root/"round2_resnet/results/new_eval_with_images.csv",dtype={"dr7objid":str,"image_path":str,"pool_role":str})
    natural=evaluation.pool_role.eq("new_natural_evaluation")
    x_train,mean,scale=features(root,train,16); x_val,_,_=features(root,validation,16,mean,scale); x_eval,_,_=features(root,evaluation[natural],16,mean,scale)
    gradient_check = finite_difference_gradient_check()
    maxiter = 1000
    weights, optimizer = fit_logistic(x_train,train.hard_class.to_numpy(int),l2=1.0,maxiter=maxiter)
    probabilities=[softmax(np.column_stack([np.ones(len(x)),x])@weights.T) for x in [x_train,x_val,x_eval]]
    result={"status":"post_hoc_fixed_traditional_baseline","protocol":{"training_rows":len(train),"validation_rows":len(validation),"natural_evaluation_rows":int(natural.sum()),"feature":"RGB 16x16 flattened","standardization":"mean/std fit on training only","solver":"L-BFGS-B multinomial softmax","l2":1.0,"maxiter":1000,"selection":"fixed before scoring; no parameter search","evaluation_note":"existing evaluation pool reused as supplemental post-hoc analysis, not a new blind result"},
            "metrics":{"train":metric(probabilities[0],train),"validation":metric(probabilities[1],validation),"natural_evaluation":metric(probabilities[2],evaluation[natural])},
            "optimization":{"success":bool(optimizer.success),"message":str(optimizer.message),"nit":int(optimizer.nit),"nfev":int(optimizer.nfev),"njev":int(getattr(optimizer,"njev",-1)),"final_objective":float(optimizer.fun),"gradient_inf_norm":float(np.max(np.abs(optimizer.jac))),"weights_shape":list(weights.shape),"maxiter":maxiter,"gradient_check":gradient_check}}
    out=root/"round2_resnet/results/closeout"; out.mkdir(exist_ok=True)
    (out/"traditional_logistic_baseline.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["metrics"],indent=2))

if __name__=="__main__":
    main()

