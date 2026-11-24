#!/usr/bin/env python3
"""Run the fixed four-run DIS post-three-epoch convergence diagnostic."""

from __future__ import annotations

import argparse
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
from torchvision.models import ResNet18_Weights

from train_formal import (
    RAW,
    build_initial_state,
    evaluate,
    load_images,
    make_model,
    set_seed,
    set_train_mode,
)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of an in-memory byte sequence."""
    return hashlib.sha256(data).hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Serialize a model state deterministically enough for run comparison."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return sha256_bytes(buffer.getvalue())


def save_checkpoint(path: Path, state: dict[str, torch.Tensor], metadata: dict) -> str:
    """Save a diagnostic checkpoint and return its file digest."""
    torch.save({"model_state_dict": state, **metadata}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Train one fresh 20-epoch target/learning-rate diagnostic run."""
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
    selected_state = None
    selected_key = None
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

        validation = evaluate(model, validation_indices, cache, frame, device, int(config["batch_size"]))
        train_eval = evaluate(model, train_indices, cache, frame, device, int(config["batch_size"]))
        if not np.isfinite(validation["soft_cross_entropy"]):
            raise FloatingPointError(f"non-finite validation soft CE at epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_definition": "hard one-hot cross-entropy" if target == "hard" else "soft normalized vote cross-entropy",
                "train_eval_soft_cross_entropy": train_eval["soft_cross_entropy"],
                "train_eval_soft_brier": train_eval["soft_brier"],
                "train_eval_hard_accuracy": train_eval["hard_accuracy"],
                "validation": validation,
            }
        )
        candidate_key = (
            validation["soft_cross_entropy"],
            validation["soft_calibration_discrepancy"],
            epoch,
        )
        if selected_key is None or candidate_key < selected_key:
            selected_key = candidate_key
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if selected_state is None:
        raise RuntimeError("no best checkpoint state was captured")
    elapsed = time.perf_counter() - start_time
    selected_epoch = int(min(history, key=lambda row: (
        row["validation"]["soft_cross_entropy"],
        row["validation"]["soft_calibration_discrepancy"],
        row["epoch"],
    ))["epoch"])
    label = f"seed{config['seed']}_lr{learning_rate:g}_{target}"
    checkpoint_dir = root / config["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "config": config,
        "seed": int(config["seed"]),
        "learning_rate": float(learning_rate),
        "target": target,
        "train_size": "small_3100",
        "selected_epoch": selected_epoch,
        "checkpoint_selection": config["checkpoint_selection"],
        "initial_state_sha256": state_digest(initial_state),
    }
    best_path = checkpoint_dir / f"{label}_best.pt"
    last_path = checkpoint_dir / f"{label}_last.pt"
    best_hash = save_checkpoint(best_path, selected_state, common_metadata)
    last_hash = save_checkpoint(last_path, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, {**common_metadata, "epoch": int(config["epochs"])})
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
        "selected_epoch": selected_epoch,
        "last_epoch": int(config["epochs"]),
        "checkpoint_selection": config["checkpoint_selection"],
        "initial_state_sha256": state_digest(initial_state),
        "best_checkpoint": str(best_path.relative_to(root)),
        "best_checkpoint_sha256": best_hash,
        "last_checkpoint": str(last_path.relative_to(root)),
        "last_checkpoint_sha256": last_hash,
        "history": history,
    }


def main() -> None:
    """Execute and serialize the four pre-declared diagnostic runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="round2_resnet/configs/training_budget_diagnostic.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / args.config).read_text())
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
        "policy": "four fixed 20-epoch small_3100 diagnostics; no early stopping and no independent evaluation pool",
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
    try:
        for learning_rate in config["learning_rates"]:
            for target in config["targets"]:
                print(json.dumps({"starting": {"learning_rate": learning_rate, "target": target}}), flush=True)
                record = run_one(config, frame, cache, initial_state, device, float(learning_rate), target, root)
                output["records"].append(record)
                output["status"] = "partial"
                output_path.write_text(json.dumps(output, indent=2) + "\n")
                print(json.dumps({"finished": {"learning_rate": learning_rate, "target": target}, "seconds": record["training_seconds"], "selected_epoch": record["selected_epoch"]}), flush=True)
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
