#!/usr/bin/env python3
"""Score the frozen large-model ResNet checkpoints on the fixed GZ2-extra pool."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18


RAW = [
    "raw_smooth_normalized",
    "raw_features_or_disk_normalized",
    "raw_star_or_artifact_normalized",
]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261123


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a checkpoint or image file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(checkpoint: dict) -> nn.Module:
    """Build the three-output ResNet architecture used by the frozen runs."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def load_images(root: Path, frame: pd.DataFrame, size: int, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode the fixed extra images once using the formal transform."""
    cache: dict[str, torch.Tensor] = {}
    for path in frame["image_path"].astype(str):
        with Image.open(root / "data/processed" / path) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        cache[path] = (tensor - mean) / std
    return cache


def class_metrics(prediction: np.ndarray, hard: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Return classwise precision/recall/F1 and a true-row confusion matrix."""
    confusion = np.zeros((3, 3), dtype=int)
    for truth, guess in zip(hard, prediction):
        confusion[int(truth), int(guess)] += 1
    metrics = []
    for class_id in range(3):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        metrics.append({"class": class_id, "support": support, "precision": precision, "recall": recall, "f1": f1})
    return metrics, confusion


def metric_arrays(probability: np.ndarray, soft: np.ndarray, hard: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-object arrays for metrics and object-level bootstrap."""
    return {
        "ce": -np.sum(soft * np.log(np.clip(probability, 1e-12, 1.0)), axis=1),
        "brier": np.sum((probability - soft) ** 2, axis=1),
        "accuracy": (prediction == hard).astype(float),
    }


def bootstrap_interval(arrays: dict[str, np.ndarray], probability: np.ndarray, hard: np.ndarray, replicates: int, seed: int) -> dict:
    """Compute a percentile interval by resampling fixed object IDs."""
    rng = np.random.default_rng(seed)
    n = len(hard)
    index = rng.integers(0, n, size=(replicates, n))
    scalar = {}
    for name, values in arrays.items():
        estimates = values[index].mean(axis=1)
        scalar[name] = {"lower": float(np.quantile(estimates, 0.025)), "upper": float(np.quantile(estimates, 0.975))}
    class_bootstrap = {str(class_id): {"precision": [], "recall": [], "f1": []} for class_id in range(3)}
    for selected in index:
        metrics, _ = class_metrics(probability[selected].argmax(axis=1), hard[selected])
        for item in metrics:
            class_id = str(item["class"])
            for key in ("precision", "recall", "f1"):
                if item[key] is not None:
                    class_bootstrap[class_id][key].append(float(item[key]))
    class_intervals = {}
    for class_id, values in class_bootstrap.items():
        class_intervals[class_id] = {}
        for key, entries in values.items():
            class_intervals[class_id][key] = None if not entries else {"lower": float(np.quantile(entries, 0.025)), "upper": float(np.quantile(entries, 0.975))}
    return {"objects": n, "replicates": replicates, "seed": seed, "scalar": scalar, "classwise": class_intervals}


def record_metrics(probability: np.ndarray, soft: np.ndarray, hard: np.ndarray, seed: int) -> dict:
    """Build one checkpoint result with predictions, metrics, and object bootstrap."""
    prediction = probability.argmax(axis=1)
    arrays = metric_arrays(probability, soft, hard, prediction)
    per_class, confusion = class_metrics(prediction, hard)
    return {
        "n": int(len(hard)),
        "soft_cross_entropy": float(arrays["ce"].mean()),
        "soft_brier": float(arrays["brier"].mean()),
        "hard_accuracy": float(arrays["accuracy"].mean()),
        "target_class_counts": np.bincount(hard, minlength=3).astype(int).tolist(),
        "predicted_class_counts": np.bincount(prediction, minlength=3).astype(int).tolist(),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "per_class_metrics": per_class,
        "bootstrap_95_percentile": bootstrap_interval(arrays, probability, hard, BOOTSTRAP_REPLICATES, seed),
        "probabilities": probability.tolist(),
        "predictions": prediction.tolist(),
    }


def main() -> None:
    """Score only the six frozen large-model checkpoints and save all evidence."""
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "round2_resnet/configs/formal.json").read_text())
    manifest = json.loads((root / "round2_resnet/results/extra_cross_sample_manifest.json").read_text())
    if manifest["status"] != "frozen_before_extra_cross_sample_inference":
        raise RuntimeError("extra manifest was not frozen before inference")
    split = pd.read_csv(root / "round2_resnet/results/extra_cross_sample_split.csv", dtype={"image_path": "string", "dr7objid": "string"})
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    cache = load_images(root, split, int(config["image_size"]), mean, std)
    soft = split[RAW].to_numpy(dtype=np.float32)
    hard = split["hard_class"].to_numpy(dtype=np.int64)
    checkpoint_records = []
    total_start = time.perf_counter()
    for index, metadata in enumerate(manifest["checkpoints"]):
        checkpoint_path = root / metadata["checkpoint"]
        actual_hash = sha256(checkpoint_path)
        if actual_hash != metadata["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["train_size"] != "large_10000" or checkpoint["target"] != metadata["target"] or int(checkpoint["seed"]) != int(metadata["seed"]):
            raise RuntimeError(f"checkpoint metadata mismatch: {checkpoint_path}")
        model = make_model(checkpoint)
        start = time.perf_counter()
        batches = []
        with torch.no_grad():
            for first in range(0, len(split), 128):
                paths = split.iloc[first : first + 128]["image_path"].astype(str).tolist()
                logits = model(torch.stack([cache[path] for path in paths]))
                batches.append(torch.softmax(logits, dim=1).numpy())
        probability = np.concatenate(batches, axis=0)
        elapsed = time.perf_counter() - start
        metric = record_metrics(probability, soft, hard, BOOTSTRAP_SEED + index)
        checkpoint_records.append({
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": actual_hash,
            "seed": int(metadata["seed"]),
            "target": metadata["target"],
            "train_size": metadata["train_size"],
            "selected_epoch": int(metadata["selected_epoch"]),
            "inference_seconds_forward_only": elapsed,
            **metric,
        })
        print(json.dumps({"scored": metadata["checkpoint"], "seconds": elapsed}), flush=True)
    total_elapsed = time.perf_counter() - total_start
    records = []
    for record in checkpoint_records:
        clean = {key: value for key, value in record.items() if key not in {"probabilities", "predictions"}}
        records.append(clean)
    summary = {"hard": {}, "soft": {}}
    for target in ("hard", "soft"):
        for seed in sorted({record["seed"] for record in checkpoint_records}):
            record = next(item for item in checkpoint_records if item["target"] == target and item["seed"] == seed)
            summary[target][str(seed)] = {key: record[key] for key in ["soft_cross_entropy", "soft_brier", "hard_accuracy", "per_class_metrics", "bootstrap_95_percentile"]}
    paired = []
    for seed in sorted({record["seed"] for record in checkpoint_records}):
        hard_record = next(item for item in checkpoint_records if item["target"] == "hard" and item["seed"] == seed)
        soft_record = next(item for item in checkpoint_records if item["target"] == "soft" and item["seed"] == seed)
        paired.append({
            "seed": seed,
            "soft_minus_hard": {
                "soft_cross_entropy": soft_record["soft_cross_entropy"] - hard_record["soft_cross_entropy"],
                "soft_brier": soft_record["soft_brier"] - hard_record["soft_brier"],
                "hard_accuracy": soft_record["hard_accuracy"] - hard_record["hard_accuracy"],
            },
        })
    for metric_name in ("soft_cross_entropy", "soft_brier", "hard_accuracy"):
        values = [item["soft_minus_hard"][metric_name] for item in paired]
        summary.setdefault("paired_mean_sample_sd", {})[metric_name] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
    predictions = split[["dr7objid", "asset_id", "image_member", "hard_class", *RAW]].copy()
    for record in checkpoint_records:
        name = f"{record['target']}_seed{record['seed']}"
        for class_id in range(3):
            predictions[f"{name}_p{class_id}"] = np.asarray(record["probabilities"])[:, class_id]
        predictions[f"{name}_prediction"] = record["predictions"]
    predictions.to_csv(root / "round2_resnet/results/extra_cross_sample_predictions.csv", index=False)
    output = {
        "status": "extra_cross_sample_inference_completed_without_model_selection",
        "research_label": "original-to-extra cross-sample generalisation under selection shift",
        "policy": "fixed object list, frozen 128-pixel preprocessing, six large_10000 checkpoints, no tuning, calibration, threshold, or checkpoint reselection",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads()},
        "rows": int(len(split)),
        "source_split": "round2_resnet/results/extra_cross_sample_split.csv",
        "source_manifest": "round2_resnet/results/extra_cross_sample_manifest.json",
        "total_forward_seconds": total_elapsed,
        "records": records,
        "summary": summary,
        "predictions": "round2_resnet/results/extra_cross_sample_predictions.csv",
        "interval_definition": "percentile bootstrap over the fixed 1,000 object IDs within each checkpoint; seed mean/sample SD is reported separately and is not a confidence interval",
    }
    path = root / "round2_resnet/results/extra_cross_sample_results.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path), "records": len(records), "total_forward_seconds": total_elapsed}, indent=2))


if __name__ == "__main__":
    main()
