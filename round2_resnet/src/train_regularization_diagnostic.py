"""Run the fixed weight-decay and dropout diagnostics at lr=3e-4.

The reference configuration is reused from the existing 20-epoch result. This
script adds only weight decay 1e-3 and dropout 0.2, each with hard and soft
targets. Dropout is inserted after global average pooling/flattening and before
the original three-output Linear layer, so the classifier initialization is
preserved.
"""

from __future__ import annotations

import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from train_formal import RAW, build_initial_state, load_images, set_seed, set_train_mode
from train_learning_rate_endpoints import evaluate_metrics, save_checkpoint, state_digest


def make_regularized_model(initial_state: dict[str, torch.Tensor], dropout: float) -> nn.Module:
    """Build ResNet-18 while preserving the original classifier parameters.

    Parameters
    ----------
    initial_state:
        ImageNet-pretrained state with a three-output Linear classifier.
    dropout:
        Probability used by a Dropout layer inserted immediately before the
        original classifier. A value of zero leaves the classifier as Linear.

    Returns
    -------
    torch.nn.Module
        ResNet-18 with layer4 and the classifier trainable.
    """

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(copy.deepcopy(initial_state))
    if dropout > 0:
        original_classifier = model.fc
        model.fc = nn.Sequential(nn.Dropout(float(dropout)), original_classifier)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def run_one(
    config: dict,
    run_config: dict,
    frame: pd.DataFrame,
    cache: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    target: str,
    root: Path,
) -> dict:
    """Train one fresh 20-epoch hard or soft regularization run."""

    set_seed(int(config["seed"]))
    train_indices = np.flatnonzero(frame["small_train"].to_numpy(dtype=bool))
    validation_indices = np.flatnonzero(frame["pool_role"].eq("fixed_validation").to_numpy())
    model = make_regularized_model(initial_state, float(run_config["dropout"]))
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
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
                "train_eval_mode": "eval",
                "train": train_metrics,
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
    ce_row = min(history, key=lambda row: (
        row["validation"]["soft_label_cross_entropy"],
        row["validation"]["calibration_discrepancy"],
        row["epoch"],
    ))
    accuracy_row = max(history, key=lambda row: (row["validation"]["hard_accuracy"], -row["epoch"]))
    if best_ce_state is None or best_accuracy_state is None:
        raise RuntimeError("best checkpoint states were not captured")
    initial_hash = state_digest(initial_state)
    common = {
        "config": config,
        "run_config": run_config,
        "seed": int(config["seed"]),
        "learning_rate": float(config["learning_rate"]),
        "target": target,
        "train_size": "small_3100",
        "checkpoint_selection": config["checkpoint_selection"],
        "validation_soft_ce_best_epoch": int(ce_row["epoch"]),
        "validation_accuracy_best_epoch": int(accuracy_row["epoch"]),
        "initial_state_sha256": initial_hash,
    }
    label = f"seed{config['seed']}_{run_config['name']}_{target}"
    checkpoint_dir = root / config["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for kind, row, state, filename in [
        ("validation_soft_ce_best", ce_row, best_ce_state, f"{label}_soft_ce_best.pt"),
        ("validation_accuracy_best", accuracy_row, best_accuracy_state, f"{label}_accuracy_best.pt"),
        ("last", history[-1], {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, f"{label}_last.pt"),
    ]:
        path = checkpoint_dir / filename
        digest = save_checkpoint(
            path,
            state,
            {**common, "checkpoint_kind": kind, "state_epoch": int(row["epoch"])},
        )
        checkpoints.append({
            "kind": kind,
            "path": str(path.relative_to(root)),
            "state_epoch": int(row["epoch"]),
            "sha256": digest,
        })
    return {
        "status": "complete",
        "run_name": run_config["name"],
        "seed": int(config["seed"]),
        "target": target,
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(run_config["weight_decay"]),
        "dropout": float(run_config["dropout"]),
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
    """Execute and serialize the four regularization runs."""

    root = Path(__file__).resolve().parents[2]
    config_path = root / "round2_resnet/configs/regularization_diagnostic.json"
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
        "policy": "two fixed regularization configurations; four fresh 20-epoch runs; no independent evaluation pool",
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
        for run_config in config["runs"]:
            for target in config["targets"]:
                print(json.dumps({"starting": {"run": run_config["name"], "target": target}}), flush=True)
                record = run_one(config, run_config, frame, cache, initial_state, device, target, root)
                output["records"].append(record)
                output["status"] = "partial"
                output_path.write_text(json.dumps(output, indent=2) + "\n")
                print(json.dumps({"finished": {"run": run_config["name"], "target": target}, "seconds": record["training_seconds"]}), flush=True)
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
