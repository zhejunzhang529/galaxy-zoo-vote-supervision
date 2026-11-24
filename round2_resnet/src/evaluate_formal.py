#!/usr/bin/env python3
"""Score saved formal checkpoints once on the newly frozen evaluation pools."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18


RAW = ["raw_smooth_normalized", "raw_features_or_disk_normalized", "raw_star_or_artifact_normalized"]


def make_model(checkpoint: dict) -> nn.Module:
    """Build the frozen three-class ResNet-18 from one checkpoint."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def metric_record(logits: torch.Tensor, soft: torch.Tensor, hard: torch.Tensor) -> dict:
    """Compute saved probability, hard-label, and confusion diagnostics."""
    probability = torch.softmax(logits, dim=1)
    log_probability = torch.log_softmax(logits, dim=1)
    prediction = probability.argmax(dim=1).numpy()
    hard_np = hard.numpy()
    ce = float((-(soft * log_probability).sum(dim=1)).mean())
    brier = float(((probability - soft) ** 2).sum(dim=1).mean())
    confusion = np.zeros((3, 3), dtype=int)
    for truth, predicted in zip(hard_np, prediction):
        confusion[int(truth), int(predicted)] += 1
    return {
        "n": int(len(hard_np)),
        "soft_cross_entropy": ce,
        "soft_brier": brier,
        "hard_accuracy": float((prediction == hard_np).mean()),
        "target_class_counts": np.bincount(hard_np, minlength=3).astype(int).tolist(),
        "predicted_class_counts": np.bincount(prediction, minlength=3).astype(int).tolist(),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "class2_recall": float(confusion[2, 2] / confusion[2].sum()) if confusion[2].sum() else None,
        "class2_positive_count": int(confusion[2].sum()),
        "class2_predicted_scores": probability[:, 2].numpy().tolist(),
        "probabilities": probability.numpy().tolist(),
    }


def load_images(root: Path, frame: pd.DataFrame, size: int, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode the specified evaluation images with the frozen transform."""
    cache = {}
    for path in frame["image_path"].astype(str):
        if path in cache:
            continue
        with Image.open(root / "data/processed" / path) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        cache[path] = (torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()) - mean) / std
    return cache


def main() -> None:
    """Score each selected checkpoint once on the frozen evaluation pools."""
    root = Path(__file__).resolve().parents[2]
    formal = json.loads((root / "round2_resnet/configs/formal.json").read_text())
    split = pd.read_csv(root / "round2_resnet/results/new_eval_with_images.csv", dtype={"image_path": "string", "pool_role": "string"})
    mean = torch.tensor(formal["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(formal["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    cache = load_images(root, split, int(formal["image_size"]), mean, std)
    records = []
    for checkpoint_path in sorted((root / "round2_resnet/checkpoints/formal").glob("*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = make_model(checkpoint)
        eval_batch_size = 128
        with torch.no_grad():
            logits = torch.cat([model(torch.stack([cache[str(path)] for path in split.iloc[start:start + eval_batch_size]["image_path"]])) for start in range(0, len(split), eval_batch_size)])
        soft = torch.tensor(split[RAW].to_numpy(dtype=np.float32), dtype=torch.float32)
        hard = torch.tensor(split["hard_class"].to_numpy(dtype=np.int64), dtype=torch.long)
        natural = split["pool_role"].eq("new_natural_evaluation").to_numpy()
        diagnostic = split["pool_role"].eq("new_class2_diagnostic_recall_only").to_numpy()
        records.append({
            "checkpoint": str(checkpoint_path.relative_to(root)),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "seed": checkpoint["seed"],
            "target": checkpoint["target"],
            "train_size": checkpoint["train_size"],
            "natural_evaluation": metric_record(logits[natural], soft[natural], hard[natural]),
            "class2_diagnostic_recall_only": metric_record(logits[diagnostic], soft[diagnostic], hard[diagnostic]),
        })
        print(json.dumps({"scored": checkpoint_path.name}), flush=True)
    output = {
        "status": "new_independent_evaluation_scored_once_after_selection",
        "policy": "results are descriptive; no threshold, checkpoint, training, or hyperparameter was changed after scoring",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "evaluation_batch_size": 128,
        "rows": int(len(split)),
        "natural_target": "new residual original-sample pool; not all galaxies or another survey",
        "records": records,
    }
    output_path = root / "round2_resnet/results/formal_evaluation.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
