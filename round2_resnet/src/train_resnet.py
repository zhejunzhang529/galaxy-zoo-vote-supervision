#!/usr/bin/env python3
"""Train the small pre-registered ResNet-18 grid on frozen development data."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


RAW = ["raw_smooth_normalized", "raw_features_or_disk_normalized", "raw_star_or_artifact_normalized"]


class VoteDataset(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame, image_size: int, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.root = root
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(self.root / "data/processed" / str(row["image_path"])) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        tensor = (tensor - self.mean) / self.std
        soft = torch.tensor(row[RAW].to_numpy(dtype=np.float32), dtype=torch.float32)
        hard = torch.tensor(int(row["hard_class"]), dtype=torch.long)
        return tensor, soft, hard


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_initial_state(seed: int, weights: ResNet18_Weights) -> dict[str, torch.Tensor]:
    set_seed(seed)
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def make_model(initial_state: dict[str, torch.Tensor], unfreeze_range: str) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(copy.deepcopy(initial_state))
    for parameter in model.parameters():
        parameter.requires_grad = False
    if unfreeze_range == "head_only":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif unfreeze_range == "layer4_plus_head":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown unfreeze range: {unfreeze_range}")
    return model


def set_train_mode(model: nn.Module, unfreeze_range: str) -> None:
    model.train()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()
    if unfreeze_range == "head_only":
        model.layer4.eval()


def metrics(logits: torch.Tensor, soft: torch.Tensor, hard: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    ce = float((-(soft * log_probabilities).sum(dim=1)).mean().item())
    brier = float(((probabilities - soft) ** 2).sum(dim=1).mean().item())
    prediction = probabilities.argmax(dim=1).cpu().numpy()
    hard_np = hard.cpu().numpy()
    confusion = np.zeros((3, 3), dtype=int)
    for truth, predicted in zip(hard_np, prediction):
        confusion[int(truth), int(predicted)] += 1
    f1_values = []
    per_class = []
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    logits_all, soft_all, hard_all = [], [], []
    with torch.no_grad():
        for images, soft, hard in loader:
            logits_all.append(model(images.to(device)).cpu())
            soft_all.append(soft)
            hard_all.append(hard)
    return metrics(torch.cat(logits_all), torch.cat(soft_all), torch.cat(hard_all))


def run_config(config: dict, frame: pd.DataFrame, initial_state: dict[str, torch.Tensor], device: torch.device, root: Path) -> dict:
    image_size = int(config["image_size"])
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    train_frame = frame[frame["small_train"]].copy()
    validation_frame = frame[frame["pool_role"].eq("fixed_validation")].copy()
    train_dataset = VoteDataset(root, train_frame, image_size, mean, std)
    validation_loader = DataLoader(VoteDataset(root, validation_frame, image_size, mean, std), batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    model = make_model(initial_state, config["unfreeze_range"])
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    epoch_orders = []
    train_indices = np.arange(len(train_frame))
    for epoch in range(int(config["epochs"])):
        epoch_orders.append(train_indices[np.random.default_rng(int(config["seed"]) + epoch).permutation(len(train_indices))])
    history = []
    start_time = time.perf_counter()
    for epoch, order in enumerate(epoch_orders, start=1):
        set_train_mode(model, config["unfreeze_range"])
        batches = []
        for start in range(0, len(order), int(config["batch_size"])):
            batches.append(order[start:start + int(config["batch_size"])])
        losses = []
        for batch_indices in batches:
            samples = [train_dataset[int(index)] for index in batch_indices]
            images = torch.stack([sample[0] for sample in samples])
            soft = torch.stack([sample[1] for sample in samples])
            images, soft = images.to(device), soft.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (-(soft * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss_mean": float(np.mean(losses)), "validation": validation})
    elapsed = time.perf_counter() - start_time
    selected = min(history, key=lambda row: (row["validation"]["soft_cross_entropy"], row["validation"]["soft_calibration_discrepancy"], row["epoch"]))
    return {
        "config": config,
        "training_seconds": elapsed,
        "selected_epoch": selected["epoch"],
        "selection_rule": "minimum validation soft CE, then classwise soft calibration discrepancy, then earliest epoch",
        "history": history,
        "selected_validation": selected["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="round2_resnet/configs/grid.json")
    parser.add_argument("--output", default="round2_resnet/results/grid_short.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / args.config).read_text())
    frame = pd.read_csv(root / config["split"], dtype={"dr7objid": "string", "image_path": "string", "pool_role": "string"})
    frame["small_train"] = frame["small_train"].astype(bool)
    weights = ResNet18_Weights.DEFAULT
    initial_state = build_initial_state(int(config["seed"]), weights)
    mps = getattr(torch.backends, "mps", None)
    device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    records = []
    for learning_rate in config["learning_rates"]:
        for weight_decay in config["weight_decays"]:
            for unfreeze_range in config["unfreeze_ranges"]:
                config_instance = {
                    **config,
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "unfreeze_range": unfreeze_range,
                    "epochs": int(config["short_epochs"]),
                    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                }
                print(json.dumps({"starting": {"learning_rate": learning_rate, "weight_decay": weight_decay, "unfreeze_range": unfreeze_range}}), flush=True)
                result = run_config(config_instance, frame, initial_state, device, root)
                records.append(result)
                print(json.dumps({"finished": {"learning_rate": learning_rate, "weight_decay": weight_decay, "unfreeze_range": unfreeze_range}, "seconds": result["training_seconds"], "selected_epoch": result["selected_epoch"], "validation": result["selected_validation"]}), flush=True)
    output = {
        "status": "short_grid_search_complete_before_formal_comparison",
        "policy": "all 8 registered combinations were run for the registered short budget; no evaluation pool was loaded",
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
