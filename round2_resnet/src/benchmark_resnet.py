#!/usr/bin/env python3
"""Small ResNet-18 input-size and batch-throughput feasibility check."""

from __future__ import annotations

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


class VoteImageDataset(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame, image_size: int, transform) -> None:
        self.root = root
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of cached frame rows."""
        return len(self.frame)

    def __getitem__(self, index: int):
        """Decode one image and return its normalized raw-vote target."""
        row = self.frame.iloc[index]
        with Image.open(self.root / "data/processed" / row["image_path"]) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            tensor = self.transform(image)
        target = torch.tensor(
            row[
                [
                    "raw_smooth_normalized",
                    "raw_features_or_disk_normalized",
                    "raw_star_or_artifact_normalized",
                ]
            ].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        return tensor, target


def device_for(torch_module) -> torch.device:
    """Select MPS when available, otherwise CPU; CUDA is not assumed."""
    mps = getattr(torch_module.backends, "mps", None)
    return torch.device("mps" if mps is not None and mps.is_available() else "cpu")


def main() -> None:
    """Measure the recorded feasibility batches for both input sizes."""
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "round2_resnet/configs/throughput.json").read_text())
    frame = pd.read_csv(
        root / config["split"], dtype={"dr7objid": "string", "split": "string", "image_path": "string"}
    )
    frame = frame[frame["split"].isin(["train", "validation"])].head(
        int(config["batch_size"]) * (int(config["batches"]) + int(config["warmup_batches"]))
    )
    weights = ResNet18_Weights.DEFAULT
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)

    def transform(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        return (tensor - mean) / std

    device = device_for(torch)
    records = []
    for image_size in config["image_sizes"]:
        dataset = VoteImageDataset(root, frame, int(image_size), transform)
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        batches = list(loader)
        for images, target in batches[: int(config["warmup_batches"])]:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (-(target * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
            loss.backward()
            optimizer.step()
        durations = []
        samples = 0
        for images, target in batches[int(config["warmup_batches"]):]:
            images, target = images.to(device), target.to(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (-(target * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
            loss.backward()
            optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            durations.append(time.perf_counter() - start)
            samples += len(images)
        records.append(
            {
                "image_size": int(image_size),
                "batch_size": int(config["batch_size"]),
                "warmup_batches": int(config["warmup_batches"]),
                "timed_batches": len(durations),
                "samples": samples,
                "median_batch_seconds": float(np.median(durations)),
                "mean_batch_seconds": float(np.mean(durations)),
                "batch_seconds": durations,
                "samples_per_second_median": float(config["batch_size"] / np.median(durations)),
                "device": str(device),
                "target_loaded": True,
                "resize_and_normalization_applied": True,
            }
        )
    output = {
        "status": "feasibility_throughput_only_no_training_run",
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
        },
        "weights": {"name": str(weights), "url": weights.url, "cache_file": "resnet18-f37072fd.pth"},
        "records": records,
    }
    output_path = root / "round2_resnet/results/throughput.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "device": str(device), "records": records}, indent=2))


if __name__ == "__main__":
    main()
