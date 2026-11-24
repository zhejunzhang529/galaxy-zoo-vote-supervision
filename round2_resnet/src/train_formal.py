#!/usr/bin/env python3
"""Run the pre-registered paired hard/soft ResNet-18 comparison."""

from __future__ import annotations

import argparse
import copy
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
from torchvision.models import ResNet18_Weights, resnet18


RAW = [
    "raw_smooth_normalized",
    "raw_features_or_disk_normalized",
    "raw_star_or_artifact_normalized",
]


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for a paired formal run."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_initial_state(seed: int, weights: ResNet18_Weights) -> dict[str, torch.Tensor]:
    """Create one reproducible ImageNet-start ResNet state."""
    set_seed(seed)
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def make_model(initial_state: dict[str, torch.Tensor]) -> nn.Module:
    """Build the three-output ResNet from a saved initial state."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(copy.deepcopy(initial_state))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def set_train_mode(model: nn.Module) -> None:
    """Freeze the stem and early blocks while enabling layer4 and head."""
    model.train()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()


def load_images(root: Path, frame: pd.DataFrame, image_size: int, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode and normalize the image cache used by a formal run."""
    cache: dict[str, torch.Tensor] = {}
    for row in frame.itertuples(index=False):
        path = str(getattr(row, "image_path"))
        if path in cache:
            continue
        with Image.open(root / "data/processed" / path) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        cache[path] = (tensor - mean) / std
    return cache


def metric_record(logits: torch.Tensor, soft: torch.Tensor, hard: torch.Tensor) -> dict:
    """Calculate validation metrics and preserve object probabilities."""
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    ce = float((-(soft * log_probabilities).sum(dim=1)).mean().item())
    brier = float(((probabilities - soft) ** 2).sum(dim=1).mean().item())
    prediction = probabilities.argmax(dim=1).cpu().numpy()
    hard_np = hard.cpu().numpy()
    confusion = np.zeros((3, 3), dtype=int)
    for truth, predicted in zip(hard_np, prediction):
        confusion[int(truth), int(predicted)] += 1
    per_class = []
    f1_values = []
    for class_id in range(3):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append({"class": class_id, "precision": precision, "recall": recall, "f1": f1})
    discrepancy = 0.0
    bins = []
    for class_id in range(3):
        probability = probabilities[:, class_id].cpu().numpy()
        target = soft[:, class_id].cpu().numpy()
        class_bins = []
        for lower in np.linspace(0.0, 1.0, 11)[:-1]:
            upper = lower + 0.1
            mask = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
            if mask.any():
                contribution = float(mask.mean() * abs(probability[mask].mean() - target[mask].mean()))
                discrepancy += contribution
                class_bins.append({"lower": float(lower), "upper": float(upper), "count": int(mask.sum()), "contribution": contribution})
        bins.append(class_bins)
    return {
        "soft_cross_entropy": ce,
        "soft_brier": brier,
        "soft_calibration_discrepancy": discrepancy,
        "hard_accuracy": float((prediction == hard_np).mean()),
        "hard_macro_f1": float(np.mean(f1_values)),
        "class2_recall": per_class[2]["recall"],
        "target_class_counts": np.bincount(hard_np, minlength=3).astype(int).tolist(),
        "predicted_class_counts": np.bincount(prediction, minlength=3).astype(int).tolist(),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "per_class_metrics": per_class,
        "calibration_bins": bins,
    }


def evaluate(model: nn.Module, indices: np.ndarray, cache: dict[str, torch.Tensor], frame: pd.DataFrame, device: torch.device, batch_size: int) -> dict:
    """Evaluate a selected row subset without changing model parameters."""
    model.eval()
    logits_all, soft_all, hard_all = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = frame.iloc[indices[start:start + batch_size]]
            images = torch.stack([cache[str(path)] for path in rows["image_path"]])
            soft = torch.tensor(rows[RAW].to_numpy(dtype=np.float32), dtype=torch.float32)
            hard = torch.tensor(rows["hard_class"].to_numpy(dtype=np.int64), dtype=torch.long)
            logits_all.append(model(images.to(device)).cpu())
            soft_all.append(soft)
            hard_all.append(hard)
    return metric_record(torch.cat(logits_all), torch.cat(soft_all), torch.cat(hard_all))


def run_one(config: dict, frame: pd.DataFrame, cache: dict[str, torch.Tensor], initial_state: dict[str, torch.Tensor], seed: int, target: str, train_size: str, device: torch.device, root: Path) -> dict:
    """Run one preregistered target/seed/size configuration."""
    train_flag = "small_train" if train_size == "small_3100" else "large_train"
    train_indices = np.flatnonzero(frame[train_flag].to_numpy(dtype=bool))
    validation_indices = np.flatnonzero(frame["pool_role"].eq("fixed_validation").to_numpy())
    model = make_model(initial_state)
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    history = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected_key: tuple[float, float, int] | None = None
    start_time = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        set_train_mode(model)
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        losses = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch_indices = order[start:start + int(config["batch_size"])]
            rows = frame.iloc[batch_indices]
            images = torch.stack([cache[str(path)] for path in rows["image_path"]]).to(device)
            soft = torch.tensor(rows[RAW].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
            hard = torch.tensor(rows["hard_class"].to_numpy(dtype=np.int64), dtype=torch.long).to(device)
            target_tensor = torch.nn.functional.one_hot(hard, num_classes=3).float() if target == "hard" else soft
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (-(target_tensor * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate(model, validation_indices, cache, frame, device, int(config["batch_size"]))
        history.append({"epoch": epoch, "train_loss_mean": float(np.mean(losses)), "validation": validation})
        candidate_key = (validation["soft_cross_entropy"], validation["soft_calibration_discrepancy"], epoch)
        if selected_key is None or candidate_key < selected_key:
            selected_key = candidate_key
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"seed": seed, "target": target, "train_size": train_size, "epoch": epoch, "validation_ce": validation["soft_cross_entropy"]}), flush=True)
    elapsed = time.perf_counter() - start_time
    selected = min(history, key=lambda row: (row["validation"]["soft_cross_entropy"], row["validation"]["soft_calibration_discrepancy"], row["epoch"]))
    selected_epoch = int(selected["epoch"])
    checkpoint_dir = root / "round2_resnet/checkpoints/formal"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"resnet18_{train_size}_{target}_seed{seed}.pt"
    if selected_state is None:
        raise RuntimeError("no selected checkpoint state was captured")
    torch.save({"model_state_dict": selected_state, "selected_epoch": selected_epoch, "config": config, "seed": seed, "target": target, "train_size": train_size}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "seed": seed,
        "target": target,
        "train_size": train_size,
        "training_seconds": elapsed,
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "selected_epoch": selected_epoch,
        "checkpoint_state_epoch": selected_epoch,
        "selection_rule": config["selection_rule"],
        "selected_validation": selected["validation"],
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": digest,
    }


def main() -> None:
    """Execute the formal paired training grid defined by the config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="round2_resnet/configs/formal.json")
    parser.add_argument("--output", default="round2_resnet/results/formal_runs.json")
    parser.add_argument("--only", default=None, help="comma-separated train_size:target:seed tuples for checkpoint repair")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / args.config).read_text())
    frame = pd.read_csv(root / config["split"], dtype={"dr7objid": "string", "image_path": "string", "pool_role": "string"})
    for column in ["small_train", "large_train"]:
        frame[column] = frame[column].astype(bool)
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    image_frame = frame[frame["small_train"] | frame["large_train"] | frame["pool_role"].eq("fixed_validation")].copy()
    cache = load_images(root, image_frame, int(config["image_size"]), mean, std)
    weights = ResNet18_Weights.DEFAULT
    mps = getattr(torch.backends, "mps", None)
    device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    records = []
    selected_runs = None
    if args.only:
        selected_runs = {tuple(item.split(":")) for item in args.only.split(",")}
    for seed in config["seeds"]:
        initial_state = build_initial_state(int(seed), weights)
        for train_size in config["train_sizes"]:
            for target in config["targets"]:
                if selected_runs is not None and (train_size, target, str(seed)) not in selected_runs:
                    continue
                print(json.dumps({"starting": {"seed": seed, "train_size": train_size, "target": target}}), flush=True)
                records.append(run_one(config, frame, cache, initial_state, int(seed), target, train_size, device, root))
    output = {
        "status": "formal_paired_development_runs_complete",
        "policy": "new evaluation pools were not loaded; no test-driven selection or tuning",
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
        },
        "weights": {"name": str(weights), "url": weights.url, "cache_file": "resnet18-f37072fd.pth"},
        "records": records,
    }
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
