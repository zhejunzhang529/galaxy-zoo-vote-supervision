"""Run the fixed 1e-5 and 1e-3 learning-rate endpoint diagnostics.

This script adds only the two pre-specified learning-rate endpoints. It keeps
the existing seed, split, architecture, preprocessing, optimizer family,
batching, data order, and 20-epoch budget fixed. It never loads an independent
evaluation pool and stops with a non-zero exit on a non-finite loss or runtime
failure.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet18_Weights

from train_formal import RAW, build_initial_state, load_images, make_model, set_seed, set_train_mode


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash a serialized initial model state for paired-run verification."""

    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_metrics(
    model: nn.Module,
    indices: np.ndarray,
    frame: pd.DataFrame,
    cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Evaluate all requested probability and hard-label metrics in one pass.

    The hard CE uses one-hot hard classes, while the soft CE uses the raw
    normalized vote fractions. Both are computed from the same eval-mode logits.
    """

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
    brier = ((probabilities - soft) ** 2).sum(dim=1).mean()
    entropy = (-(probabilities * log_probabilities).sum(dim=1)).mean()
    prediction = probabilities.argmax(dim=1)
    discrepancy = 0.0
    for class_id in range(3):
        probability = probabilities[:, class_id].numpy()
        target = soft[:, class_id].numpy()
        for lower in np.linspace(0.0, 1.0, 11)[:-1]:
            upper = lower + 0.1
            mask = (probability >= lower) & (
                probability < upper if upper < 1.0 else probability <= upper
            )
            if mask.any():
                discrepancy += float(mask.mean() * abs(probability[mask].mean() - target[mask].mean()))
    return {
        "count": int(len(indices)),
        "hard_label_cross_entropy": float(hard_ce.item()),
        "soft_label_cross_entropy": float(soft_ce.item()),
        "brier": float(brier.item()),
        "hard_accuracy": float((prediction == hard).float().mean().item()),
        "mean_max_probability": float(probabilities.max(dim=1).values.mean().item()),
        "mean_prediction_entropy": float(entropy.item()),
        "calibration_discrepancy": float(discrepancy),
    }


def save_checkpoint(path: Path, state: dict[str, torch.Tensor], metadata: dict) -> str:
    """Save a checkpoint with state epoch and protocol metadata."""

    torch.save({"model_state_dict": state, **metadata}, path)
    return file_digest(path)


def run_one(
    config: dict,
    frame: pd.DataFrame,
    cache: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    learning_rate: float,
    target: str,
    root: Path,
) -> dict:
    """Train and record one fresh endpoint run."""

    set_seed(int(config["seed"]))
    train_indices = np.flatnonzero(frame["small_train"].to_numpy(dtype=bool))
    validation_indices = np.flatnonzero(frame["pool_role"].eq("fixed_validation").to_numpy())
    model = make_model(copy.deepcopy(initial_state))
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(config["weight_decay"]),
    )
    history = []
    best_ce_state = None
    best_accuracy_state = None
    best_ce_key = None
    best_accuracy_key = None
    start_time = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        set_train_mode(model)
        order = np.random.default_rng(int(config["seed"]) + epoch).permutation(train_indices)
        losses = []
        for start in range(0, len(order), int(config["batch_size"])):
            rows = frame.iloc[order[start : start + int(config["batch_size"])]]
            images = torch.stack([cache[str(path)] for path in rows["image_path"]]).to(device)
            soft = torch.tensor(rows[RAW].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
            hard = torch.tensor(rows["hard_class"].to_numpy(dtype=np.int64), dtype=torch.long).to(device)
            target_tensor = torch.nn.functional.one_hot(hard, num_classes=3).float() if target == "hard" else soft
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (-(target_tensor * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_metrics = evaluate_metrics(model, train_indices, frame, cache, device, int(config["batch_size"]))
        validation_metrics = evaluate_metrics(model, validation_indices, frame, cache, device, int(config["batch_size"]))
        if not np.isfinite(validation_metrics["soft_label_cross_entropy"]):
            raise FloatingPointError(f"non-finite validation soft CE at epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_definition": "hard one-hot cross-entropy" if target == "hard" else "soft normalized vote cross-entropy",
                "train_eval": train_metrics,
                "validation": validation_metrics,
            }
        )
        ce_key = (
            validation_metrics["soft_label_cross_entropy"],
            validation_metrics["calibration_discrepancy"],
            epoch,
        )
        accuracy_key = (-validation_metrics["hard_accuracy"], epoch)
        if best_ce_key is None or ce_key < best_ce_key:
            best_ce_key = ce_key
            best_ce_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_accuracy_key is None or accuracy_key < best_accuracy_key:
            best_accuracy_key = accuracy_key
            best_accuracy_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    elapsed = time.perf_counter() - start_time
    ce_row = min(
        history,
        key=lambda row: (
            row["validation"]["soft_label_cross_entropy"],
            row["validation"]["calibration_discrepancy"],
            row["epoch"],
        ),
    )
    accuracy_row = max(
        history,
        key=lambda row: (row["validation"]["hard_accuracy"], -row["epoch"]),
    )
    initial_hash = state_digest(initial_state)
    common = {
        "config": config,
        "seed": int(config["seed"]),
        "learning_rate": float(learning_rate),
        "target": target,
        "train_size": "small_3100",
        "checkpoint_selection": config["checkpoint_selection"],
        "validation_soft_ce_best_epoch": int(ce_row["epoch"]),
        "validation_accuracy_best_epoch": int(accuracy_row["epoch"]),
        "initial_state_sha256": initial_hash,
    }
    label = f"seed{config['seed']}_lr{learning_rate:g}_{target}"
    checkpoint_dir = root / config["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    if best_ce_state is None or best_accuracy_state is None:
        raise RuntimeError("best checkpoint states were not captured")
    for kind, row, state, filename in [
        ("validation_soft_ce_best", ce_row, best_ce_state, f"{label}_soft_ce_best.pt"),
        ("validation_accuracy_best", accuracy_row, best_accuracy_state, f"{label}_accuracy_best.pt"),
        ("last", history[-1], {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, f"{label}_last.pt"),
    ]:
        path = checkpoint_dir / filename
        metadata = {**common, "checkpoint_kind": kind, "state_epoch": int(row["epoch"])}
        digest = save_checkpoint(path, state, metadata)
        checkpoints.append(
            {
                "kind": kind,
                "path": str(path.relative_to(root)),
                "state_epoch": int(row["epoch"]),
                "sha256": digest,
            }
        )
    return {
        "status": "complete",
        "seed": int(config["seed"]),
        "target": target,
        "learning_rate": float(learning_rate),
        "weight_decay": float(config["weight_decay"]),
        "unfreeze_range": config["unfreeze_range"],
        "train_size": "small_3100",
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "training_seconds": elapsed,
        "validation_soft_ce_best_epoch": int(ce_row["epoch"]),
        "validation_accuracy_best_epoch": int(accuracy_row["epoch"]),
        "initial_state_sha256": initial_hash,
        "checkpoints": checkpoints,
        "history": history,
    }

def main() -> None:
    """Execute the four endpoint runs and write incremental results."""

    root = Path(__file__).resolve().parents[2]
    config_path = root / "round2_resnet/configs/learning_rate_endpoints.json"
    config = json.loads(config_path.read_text())
    frame = pd.read_csv(root / config["split"], dtype={"dr7objid": "string", "image_path": "string", "pool_role": "string"})
    for column in ["small_train", "large_train"]:
        frame[column] = frame[column].astype(bool)
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    image_frame = frame[frame["small_train"] | frame["pool_role"].eq("fixed_validation")].copy()
    cache = load_images(root, image_frame, int(config["image_size"]), mean, std)
    weights = ResNet18_Weights.DEFAULT
    initial_state = build_initial_state(int(config["seed"]), weights)
    mps = getattr(torch.backends, "mps", None)
    device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    output = {
        "status": "running",
        "policy": "two fixed endpoint learning rates; four fresh 20-epoch runs; no independent evaluation pool",
        "config": config,
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
        "weights": {"name": str(weights), "url": weights.url, "cache_file": "resnet18-f37072fd.pth"},
        "records": [],
    }
    output_path = root / config["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    try:
        for learning_rate in config["learning_rates"]:
            for target in config["targets"]:
                print(json.dumps({"starting": {"learning_rate": learning_rate, "target": target}}), flush=True)
                record = run_one(config, frame, cache, initial_state, device, float(learning_rate), target, root)
                output["records"].append(record)
                output["status"] = "partial"
                output_path.write_text(json.dumps(output, indent=2) + "\n")
                print(json.dumps({"finished": {"learning_rate": learning_rate, "target": target}, "seconds": record["training_seconds"]}), flush=True)
    except Exception as error:
        output["status"] = "failed_and_stopped"
        output["failure"] = {"type": type(error).__name__, "message": str(error)}
        output_path.write_text(json.dumps(output, indent=2) + "\n")
        raise
    output["status"] = "complete"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "records": len(output["records"]), "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
