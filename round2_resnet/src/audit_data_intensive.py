#!/usr/bin/env python3
"""Summarise retained DIS data volume, image bytes, and processing boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def file_bytes(paths: list[Path]) -> int:
    """Sum bytes for existing unique files."""
    return int(sum(path.stat().st_size for path in set(paths) if path.exists()))


def main() -> None:
    """Write an evidence table without decoding or training on new data."""
    root = Path(__file__).resolve().parents[2]
    labels = pd.read_csv(root / "data/raw/gz2_hart16.csv.gz", usecols=["dr7objid", "sample"], dtype={"dr7objid": "string", "sample": "string"})
    mapping = pd.read_csv(root / "data/raw/gz2_filename_mapping.csv", dtype={"objid": "string", "sample": "string", "asset_id": "string"})
    archive = root / "data/raw/images_gz2.zip"
    with ZipFile(archive) as zip_file:
        jpg_members = [name for name in zip_file.namelist() if name.lower().endswith(".jpg") and not name.startswith("__MACOSX/")]
    split = pd.read_csv(root / "round2_resnet/results/resnet_split.csv", dtype=str)
    evaluation = pd.read_csv(root / "round2_resnet/results/new_eval_with_images.csv", dtype=str)
    formal_runs = json.loads((root / "round2_resnet/results/formal_runs.json").read_text())
    train_paths = [root / "data/processed" / path for path in split.loc[split["large_train"].eq("True"), "image_path"]]
    validation_paths = [root / "data/processed" / path for path in split.loc[split["pool_role"].eq("fixed_validation"), "image_path"]]
    evaluation_paths = [root / "data/processed" / path for path in evaluation["image_path"]]
    output = {
        "label_rows": int(len(labels)),
        "label_sample_counts": labels["sample"].value_counts().to_dict(),
        "mapping_rows": int(len(mapping)),
        "mapping_sample_counts": mapping["sample"].value_counts().to_dict(),
        "image_archive": {"path": "data/raw/images_gz2.zip", "compressed_bytes": archive.stat().st_size, "jpg_members_excluding_macos_metadata": len(jpg_members)},
        "formal_rows": {"large_train": int(split["large_train"].eq("True").sum()), "fixed_validation": int(split["pool_role"].eq("fixed_validation").sum()), "natural_evaluation": int(evaluation["pool_role"].eq("new_natural_evaluation").sum()), "diagnostic_evaluation": int(evaluation["pool_role"].eq("new_class2_diagnostic_recall_only").sum())},
        "image_storage": {"large_train_files": len(set(train_paths)), "large_train_bytes": file_bytes(train_paths), "validation_files": len(set(validation_paths)), "validation_bytes": file_bytes(validation_paths), "evaluation_files": len(set(evaluation_paths)), "evaluation_bytes": file_bytes(evaluation_paths)},
        "formal_training_seconds": {"runs": len(formal_runs["records"]), "sum": float(sum(record["training_seconds"] for record in formal_runs["records"])), "mean": float(sum(record["training_seconds"] for record in formal_runs["records"]) / len(formal_runs["records"]))},
        "processing_boundary": {"archive_and_mapping": "matching is performed by object ID, asset ID and archive member before image extraction", "formal_and_diagnostic_training": "train_formal.load_images decodes, converts RGB, resizes bilinearly and applies ImageNet normalization before the first epoch; batches then stack cached tensors", "historical_short_grid": "train_resnet.VoteDataset decodes and transforms in __getitem__ with num_workers=0", "formal_and_extra_inference": "the selected evaluation frame is decoded and normalized into a cache before batched forward passes", "memory": "decoded image bytes and batch tensor estimates are available; peak process memory was not measured", "throughput": "existing 128/224 benchmark times forward/loss/backward/optimizer step after materialising loader batches; it is not end-to-end formal fine-tuning throughput", "service_scope": "static local archive experiment; no real-time cutout or inference service"},
    }
    path = root / "round2_resnet/results/data_intensive_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
