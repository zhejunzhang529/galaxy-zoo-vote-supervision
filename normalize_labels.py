#!/usr/bin/env python3
"""Preserve and normalize the GZ2 Task 01 raw/weighted vote fractions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_stats(values: pd.DataFrame) -> dict:
    sums = values.sum(axis=1)
    return {
        "rows": int(len(values)),
        "sum_min": float(sums.min()),
        "sum_median": float(sums.median()),
        "sum_max": float(sums.max()),
        "sum_mean": float(sums.mean()),
        "sum_abs_deviation_max": float(np.max(np.abs(sums - 1.0))),
        "sum_within_1e-5": int(np.isclose(sums, 1.0, atol=1e-5).sum()),
        "nonpositive_sum_rows": int((sums <= 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/gz2_hart16.csv.gz")
    parser.add_argument("--output", default="data/processed/normalized_t01_labels.csv.gz")
    parser.add_argument("--manifest", default="results/normalization_manifest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    input_path = (root / args.input).resolve()
    output_path = (root / args.output).resolve()
    manifest_path = (root / args.manifest).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(
        input_path,
        dtype={"dr7objid": "string", "sample": "string", "gz2_class": "string"},
    )
    raw = [
        "t01_smooth_or_features_a01_smooth_fraction",
        "t01_smooth_or_features_a02_features_or_disk_fraction",
        "t01_smooth_or_features_a03_star_or_artifact_fraction",
    ]
    weighted = [column.replace("_fraction", "_weighted_fraction") for column in raw]
    required = ["dr7objid", "sample", "gz2_class", "total_classifications", "total_votes"] + raw + weighted
    missing = [column for column in required if column not in labels.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    raw_values = labels[raw].apply(pd.to_numeric, errors="raise").astype("float64")
    weighted_values = labels[weighted].apply(pd.to_numeric, errors="raise").astype("float64")
    raw_sums = raw_values.sum(axis=1)
    weighted_sums = weighted_values.sum(axis=1)
    if (raw_sums <= 0).any() or (weighted_sums <= 0).any():
        raise ValueError("Cannot normalize a non-positive vote-fraction sum")

    output = labels[["dr7objid", "sample", "gz2_class", "total_classifications", "total_votes"]].copy()
    for column in raw:
        short = column.replace("t01_smooth_or_features_a01_smooth_", "smooth_")
        short = short.replace("t01_smooth_or_features_a02_features_or_disk_", "features_or_disk_")
        short = short.replace("t01_smooth_or_features_a03_star_or_artifact_", "star_or_artifact_")
        output[f"raw_{short}"] = raw_values[column]
    for column in weighted:
        short = column.replace("t01_smooth_or_features_a01_smooth_", "smooth_")
        short = short.replace("t01_smooth_or_features_a02_features_or_disk_", "features_or_disk_")
        short = short.replace("t01_smooth_or_features_a03_star_or_artifact_", "star_or_artifact_")
        output[f"weighted_{short}"] = weighted_values[column]
    for column, name in zip(raw, ["smooth", "features_or_disk", "star_or_artifact"]):
        output[f"raw_{name}_normalized"] = raw_values[column] / raw_sums
    for column, name in zip(weighted, ["smooth", "features_or_disk", "star_or_artifact"]):
        output[f"weighted_{name}_normalized"] = weighted_values[column] / weighted_sums
    raw_norm_cols = [f"raw_{name}_normalized" for name in ["smooth", "features_or_disk", "star_or_artifact"]]
    output["raw_hard_class"] = output[raw_norm_cols].to_numpy().argmax(axis=1)
    output.to_csv(output_path, index=False, compression="gzip")

    manifest = {
        "input": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "input_sha256": sha256(input_path),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "rows": int(len(output)),
        "id_dtype": "string",
        "target": "GZ2 Task 01: smooth / features_or_disk / star_or_artifact",
        "normalization": "each raw or weighted three-class vector divided by its own positive row sum",
        "raw_original_columns": raw,
        "weighted_original_columns": weighted,
        "raw_stats_before": fraction_stats(raw_values),
        "weighted_stats_before": fraction_stats(weighted_values),
        "raw_stats_after": fraction_stats(output[raw_norm_cols]),
        "weighted_stats_after": fraction_stats(output[[f"weighted_{name}_normalized" for name in ["smooth", "features_or_disk", "star_or_artifact"]]]),
        "note": "Original values are retained; normalized values are derived columns. Debiased columns are intentionally not used here because their row sums are not a probability simplex in the audit.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
