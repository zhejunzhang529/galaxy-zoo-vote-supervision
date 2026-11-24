"""Evaluate the saved best and last checkpoints from the budget diagnostic.

This is an inference-only analysis.  It uses only the fixed small-training and
validation rows in the existing split manifest; no independent evaluation pool
is loaded and no model parameters are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_formal import RAW, load_images, make_model


def checkpoint_digest(path: Path) -> str:
    """Return the SHA-256 digest of a saved checkpoint file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_checkpoint(
    checkpoint: Path,
    indices: np.ndarray,
    frame: pd.DataFrame,
    cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Compute comparable hard/soft fit and confidence metrics for one split.

    Parameters
    ----------
    checkpoint:
        Existing checkpoint to load without modification.
    indices:
        Row indices for either the small training or fixed validation split.
    frame, cache, device, batch_size:
        Fixed data and inference settings from the diagnostic protocol.

    Returns
    -------
    dict
        Eval-mode hard-label CE, soft-label CE, accuracy, mean maximum
        probability, entropy, and the number of evaluated objects.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(payload["model_state_dict"])
    model.to(device)
    model.eval()
    logits_all, soft_all, hard_all = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = frame.iloc[indices[start : start + batch_size]]
            images = torch.stack([cache[str(path)] for path in rows["image_path"]]).to(device)
            logits_all.append(model(images).cpu())
            soft_all.append(torch.tensor(rows[RAW].to_numpy(dtype=np.float32), dtype=torch.float32))
            hard_all.append(torch.tensor(rows["hard_class"].to_numpy(dtype=np.int64), dtype=torch.long))
    logits = torch.cat(logits_all)
    soft = torch.cat(soft_all)
    hard = torch.cat(hard_all)
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    hard_one_hot = torch.nn.functional.one_hot(hard, num_classes=3).float()
    hard_ce = (-(hard_one_hot * log_probabilities).sum(dim=1)).mean()
    soft_ce = (-(soft * log_probabilities).sum(dim=1)).mean()
    entropy = (-(probabilities * log_probabilities).sum(dim=1)).mean()
    prediction = probabilities.argmax(dim=1)
    return {
        "count": int(len(indices)),
        "hard_label_cross_entropy": float(hard_ce.item()),
        "soft_label_cross_entropy": float(soft_ce.item()),
        "hard_accuracy": float((prediction == hard).float().mean().item()),
        "mean_max_probability": float(probabilities.max(dim=1).values.mean().item()),
        "mean_prediction_entropy": float(entropy.item()),
    }


def selection_sensitivity(record: dict) -> dict:
    """Summarize soft-CE selection versus earliest maximum-accuracy selection."""

    history = record["history"]
    ce_choice = min(
        history,
        key=lambda row: (
            row["validation"]["soft_cross_entropy"],
            row["validation"]["soft_calibration_discrepancy"],
            row["epoch"],
        ),
    )
    accuracy_choice = max(
        history,
        key=lambda row: (row["validation"]["hard_accuracy"], -row["epoch"]),
    )
    return {
        "soft_ce_selected_epoch": int(ce_choice["epoch"]),
        "soft_ce_selected_validation": {
            "soft_cross_entropy": ce_choice["validation"]["soft_cross_entropy"],
            "brier": ce_choice["validation"]["soft_brier"],
            "accuracy": ce_choice["validation"]["hard_accuracy"],
        },
        "earliest_max_accuracy_epoch": int(accuracy_choice["epoch"]),
        "earliest_max_accuracy_validation": {
            "accuracy": accuracy_choice["validation"]["hard_accuracy"],
            "soft_cross_entropy": accuracy_choice["validation"]["soft_cross_entropy"],
            "brier": accuracy_choice["validation"]["soft_brier"],
        },
    }


def main() -> None:
    """Run inference on all eight existing best/last checkpoints."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="round2_resnet/results/closeout/training_budget_diagnostic_runs.json")
    parser.add_argument("--output", default="round2_resnet/results/closeout/training_budget_checkpoint_analysis.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs_path = root / args.runs
    runs = json.loads(runs_path.read_text())
    config = runs["config"]
    frame = pd.read_csv(
        root / config["split"],
        dtype={"dr7objid": "string", "image_path": "string", "pool_role": "string"},
    )
    for column in ["small_train", "large_train"]:
        frame[column] = frame[column].astype(bool)
    train_indices = np.flatnonzero(frame["small_train"].to_numpy(dtype=bool))
    validation_indices = np.flatnonzero(frame["pool_role"].eq("fixed_validation").to_numpy())
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    image_frame = frame[frame["small_train"] | frame["pool_role"].eq("fixed_validation")].copy()
    cache = load_images(root, image_frame, int(config["image_size"]), mean, std)
    mps = getattr(torch.backends, "mps", None)
    device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    output = {
        "status": "complete",
        "policy": "inference-only analysis of four existing best and four existing last checkpoints",
        "independent_pools_loaded": False,
        "source_runs": str(runs_path.relative_to(root)),
        "split": str((root / config["split"]).relative_to(root)),
        "config": {
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "normalization": config["normalization"],
            "train_subset": "small_3100",
            "validation_subset": "fixed_validation",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": str(device),
            "mps_available": bool(mps is not None and mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_threads": torch.get_num_threads(),
        },
        "records": [],
    }
    start_all = time.perf_counter()
    for record in runs["records"]:
        sensitivity = selection_sensitivity(record)
        checkpoint_records = []
        for label, field in [("best", "best_checkpoint"), ("last", "last_checkpoint")]:
            checkpoint = root / record[field]
            start = time.perf_counter()
            train_metrics = evaluate_checkpoint(
                checkpoint, train_indices, frame, cache, device, int(config["batch_size"])
            )
            validation_metrics = evaluate_checkpoint(
                checkpoint, validation_indices, frame, cache, device, int(config["batch_size"])
            )
            checkpoint_records.append(
                {
                    "label": label,
                    "path": str(checkpoint.relative_to(root)),
                    "sha256": checkpoint_digest(checkpoint),
                    "recorded_sha256": record[field + "_sha256"],
                    "sha256_matches_source_record": checkpoint_digest(checkpoint) == record[field + "_sha256"],
                    "checkpoint_epoch": record["selected_epoch"] if label == "best" else record["last_epoch"],
                    "inference_seconds": time.perf_counter() - start,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
        output["records"].append(
            {
                "seed": record["seed"],
                "learning_rate": record["learning_rate"],
                "target": record["target"],
                "train_count": int(len(train_indices)),
                "validation_count": int(len(validation_indices)),
                "original_selected_epoch": record["selected_epoch"],
                "selection_sensitivity": sensitivity,
                "checkpoints": checkpoint_records,
            }
        )
    output["total_inference_seconds"] = time.perf_counter() - start_all
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "records": 4, "checkpoints": 8, "seconds": output["total_inference_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
