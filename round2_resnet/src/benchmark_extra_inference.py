#!/usr/bin/env python3
"""Measure the fixed extra-image inference throughput before scoring."""

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
from torchvision.models import resnet18


RAW = [
    "raw_smooth_normalized",
    "raw_features_or_disk_normalized",
    "raw_star_or_artifact_normalized",
]


def load_model(checkpoint_path: Path) -> nn.Module:
    """Build the three-output ResNet from a frozen checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def load_batch(root: Path, paths: list[str], size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Decode and normalize one fixed batch using the formal 128-pixel path."""
    tensors = []
    for path in paths:
        with Image.open(root / "data/processed" / path) as image:
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


def main() -> None:
    """Run a fixed small-batch timing probe and write its estimate."""
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "round2_resnet/configs/formal.json").read_text())
    split = pd.read_csv(root / "round2_resnet/results/extra_cross_sample_split.csv", dtype={"image_path": "string"})
    paths = split["image_path"].astype(str).head(64).tolist()
    mean = torch.tensor(config["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    checkpoint_path = root / "round2_resnet/checkpoints/formal/resnet18_large_10000_hard_seed20261119.pt"
    model = load_model(checkpoint_path)
    torch.set_num_threads(6)
    batches = [paths[start : start + 16] for start in range(0, len(paths), 16)]
    with torch.no_grad():
        for batch_paths in batches[:2]:
            model(load_batch(root, batch_paths, int(config["image_size"]), mean, std))
    elapsed = []
    with torch.no_grad():
        for _ in range(5):
            start = time.perf_counter()
            for batch_paths in batches:
                model(load_batch(root, batch_paths, int(config["image_size"]), mean, std))
            elapsed.append(time.perf_counter() - start)
    median = float(np.median(elapsed))
    throughput = float(len(paths) / median)
    estimate = float(6 * len(split) / throughput)
    output = {
        "status": "small_batch_extra_inference_timing_before_full_score",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
        "image_size": int(config["image_size"]),
        "batch_size": 16,
        "timed_images": len(paths),
        "warmup_passes": 2,
        "repeats": 5,
        "seconds": elapsed,
        "median_seconds": median,
        "images_per_second": throughput,
        "estimated_seconds_for_six_checkpoints_and_1000_images": estimate,
        "timing_boundary": "PIL decode + RGB conversion + bilinear resize + ImageNet normalization + CPU forward; no disk write or metric calculation",
    }
    path = root / "round2_resnet/results/extra_cross_sample_timing.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
