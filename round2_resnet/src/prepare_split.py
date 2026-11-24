#!/usr/bin/env python3
"""Audit and freeze new DIS ResNet development/evaluation pools.

The old development split, old 150-object pilot test, old 10,000-object
natural pool and old 100-object positive-only diagnostic pool are all excluded.
No model or checkpoint is loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


RAW = [
    "raw_smooth_normalized",
    "raw_features_or_disk_normalized",
    "raw_star_or_artifact_normalized",
]
WEIGHTED = [
    "weighted_smooth_normalized",
    "weighted_features_or_disk_normalized",
    "weighted_star_or_artifact_normalized",
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(values: pd.Series) -> str:
    """Hash sorted object identifiers without floating-point conversion."""
    result = hashlib.sha256()
    for value in sorted(values.astype("string").tolist()):
        result.update(value.encode("utf-8"))
        result.update(b"\n")
    return result.hexdigest()


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count integer hard classes in a split frame."""
    return {str(int(k)): int(v) for k, v in frame["hard_class"].value_counts().sort_index().items()}


def clean(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Normalize identifier and path columns while preserving string IDs."""
    output = frame.copy()
    for column in [id_column, "dr7objid", "asset_id", "image_member", "sample", "image_path", "split", "eval_set"]:
        if column in output:
            output[column] = output[column].astype("string").str.strip()
    return output


def main() -> None:
    """Audit exclusions, sample frozen pools, and write the split manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="round2_resnet/configs/split.json")
    parser.add_argument("--output-split", default="round2_resnet/results/resnet_split.csv")
    parser.add_argument("--output-manifest", default="round2_resnet/results/split_manifest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config
    cfg = json.loads(config_path.read_text())

    development = clean(pd.read_csv(root / cfg["development_split"], dtype={"dr7objid": "string", "asset_id": "string", "image_member": "string", "image_path": "string", "split": "string"}), "dr7objid")
    historical = clean(pd.read_csv(root / cfg["historical_eval_split"], dtype={"dr7objid": "string", "asset_id": "string", "image_member": "string"}), "dr7objid")
    for name, frame in [("development", development), ("historical_eval", historical)]:
        for column in ["dr7objid", "asset_id", "image_member"]:
            if frame[column].duplicated().any():
                raise ValueError(f"{name} has duplicate {column}")

    labels = clean(pd.read_csv(root / cfg["labels"], dtype={"dr7objid": "string", "sample": "string"}), "dr7objid")
    mapping = clean(pd.read_csv(root / cfg["mapping"], dtype={"objid": "string", "sample": "string", "asset_id": "string"}), "objid")
    mapping = mapping.rename(columns={"objid": "dr7objid"})
    with zipfile.ZipFile(root / cfg["image_zip"]) as archive:
        members: dict[str, list[str]] = {}
        for member in archive.namelist():
            if member.lower().endswith(".jpg"):
                members.setdefault(Path(member).stem, []).append(member)

    joined = labels.merge(mapping, on=["dr7objid", "sample"], how="inner", validate="one_to_one")
    joined = joined[joined["sample"].eq(cfg["candidate_sample"])].copy()
    joined["image_member"] = joined["asset_id"].map(
        lambda asset: members.get(str(asset), [None])[0] if len(members.get(str(asset), [])) == 1 else None
    )
    joined = joined[joined["image_member"].notna()].drop_duplicates("dr7objid", keep="first")
    joined["hard_class"] = joined[RAW].to_numpy().argmax(axis=1)

    excluded = pd.concat([
        development[["dr7objid", "asset_id", "image_member"]],
        historical[["dr7objid", "asset_id", "image_member"]],
    ], ignore_index=True).drop_duplicates()
    excluded_ids = set(excluded["dr7objid"])
    excluded_assets = set(excluded["asset_id"])
    excluded_members = set(excluded["image_member"])
    candidates = joined[
        ~joined["dr7objid"].isin(excluded_ids)
        & ~joined["asset_id"].isin(excluded_assets)
        & ~joined["image_member"].isin(excluded_members)
    ].copy()
    if any(candidates[column].duplicated().any() for column in ["dr7objid", "asset_id", "image_member"]):
        raise ValueError("candidate pool is not object/image unique")

    natural_count = int(cfg["new_natural_eval_rows"])
    order = np.random.default_rng(int(cfg["new_natural_eval_seed"])).permutation(len(candidates))
    natural = candidates.iloc[order[:natural_count]].copy()
    remaining = candidates[~candidates["dr7objid"].isin(set(natural["dr7objid"]))].copy()
    diagnostic = remaining[remaining["hard_class"].eq(2)].copy()
    diagnostic_order = np.random.default_rng(int(cfg["new_diagnostic_seed"])).permutation(len(diagnostic))
    diagnostic = diagnostic.iloc[diagnostic_order].copy()

    remaining_after_diagnostic = remaining[~remaining["dr7objid"].isin(set(diagnostic["dr7objid"]))].copy()
    additional_count = int(cfg["additional_large_train_rows"])
    if len(remaining_after_diagnostic) < additional_count:
        raise ValueError("not enough residual objects for the large training addition")
    add_order = np.random.default_rng(int(cfg["additional_large_train_seed"])).permutation(len(remaining_after_diagnostic))
    additional = remaining_after_diagnostic.iloc[add_order[:additional_count]].copy()

    current_train = development[development["split"].eq("train")].copy()
    current_validation = development[development["split"].eq("validation")].copy()
    current_train["small_train"] = True
    current_train["large_train"] = True
    current_train["pool_role"] = "development_train_small_and_large"
    additional["small_train"] = False
    additional["large_train"] = True
    additional["pool_role"] = "new_large_train_addition"
    current_validation["small_train"] = False
    current_validation["large_train"] = False
    current_validation["pool_role"] = "fixed_validation"
    natural["small_train"] = False
    natural["large_train"] = False
    natural["pool_role"] = "new_natural_evaluation"
    diagnostic["small_train"] = False
    diagnostic["large_train"] = False
    diagnostic["pool_role"] = "new_class2_diagnostic_recall_only"

    combined = pd.concat([current_train, additional, current_validation, natural, diagnostic], ignore_index=True, sort=False)
    combined["image_path"] = combined.get("image_path", pd.Series(index=combined.index, dtype="string")).astype("string")
    additional_mask = combined["pool_role"].eq("new_large_train_addition")
    image_dir = root / cfg["image_output_dir"]
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / cfg["image_zip"]) as archive:
        for _, row in combined[additional_mask].iterrows():
            target = image_dir / f"{row['asset_id']}.jpg"
            if not target.exists():
                with archive.open(str(row["image_member"]), "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
            combined.loc[combined["dr7objid"].eq(row["dr7objid"]), "image_path"] = str(target.relative_to(root / "data/processed"))

    columns = ["dr7objid", "sample", "asset_id", "image_member", "image_path", "pool_role", "small_train", "large_train", "hard_class", *RAW, *WEIGHTED, "gz2_class", "total_votes"]
    output_split = root / args.output_split
    output_manifest = root / args.output_manifest
    output_split.parent.mkdir(parents=True, exist_ok=True)
    combined[columns].to_csv(output_split, index=False)

    manifest = {
        "status": "new_resnet_protocol_frozen_before_model_scoring",
        "config": cfg,
        "config_sha256": sha256(config_path),
        "development_split_sha256": sha256(root / cfg["development_split"]),
        "historical_eval_split_sha256": sha256(root / cfg["historical_eval_split"]),
        "labels_sha256": sha256(root / cfg["labels"]),
        "mapping_sha256": sha256(root / cfg["mapping"]),
        "image_zip_sha256": sha256(root / cfg["image_zip"]),
        "development_excluded": {
            "rows": int(len(development)),
            "train_rows": int((development["split"] == "train").sum()),
            "validation_rows": int((development["split"] == "validation").sum()),
            "historical_pilot_test_rows": int((development["split"] == "test").sum()),
            "hard_class_counts": class_counts(development),
            "object_ids_sha256": digest(development["dr7objid"]),
        },
        "historical_eval_excluded": {
            "rows": int(len(historical)),
            "natural_rows": int((historical["eval_set"] == "natural_proportion").sum()),
            "class2_diagnostic_rows": int((historical["eval_set"] == "class2_diagnostic").sum()),
            "hard_class_counts": class_counts(historical),
            "object_ids_sha256": digest(historical["dr7objid"]),
        },
        "candidate_pool_after_all_historical_exclusion": {
            "rows": int(len(candidates)),
            "hard_class_counts": class_counts(candidates),
            "object_ids_unique": int(candidates["dr7objid"].nunique()),
            "asset_ids_unique": int(candidates["asset_id"].nunique()),
            "image_members_unique": int(candidates["image_member"].nunique()),
        },
        "new_pools": {
            "natural": {"rows": int(len(natural)), "hard_class_counts": class_counts(natural), "sampling": "uniform without replacement over the full residual candidate pool"},
            "diagnostic": {"rows": int(len(diagnostic)), "hard_class_counts": class_counts(diagnostic), "sampling": "all residual class-2 objects after natural sampling; recall only"},
            "additional_large_train": {"rows": int(len(additional)), "hard_class_counts": class_counts(additional), "sampling": "fixed-seed uniform sample after both new evaluation pools"},
        },
        "nested_development": {
            "small_train_rows": int(current_train.shape[0]),
            "large_train_rows": int(current_train.shape[0] + additional.shape[0]),
            "validation_rows": int(current_validation.shape[0]),
            "small_train_hard_class_counts": class_counts(current_train),
            "large_train_hard_class_counts": class_counts(pd.concat([current_train, additional])),
            "data_quantity_comparison_caveat": "small training is the existing quota-enriched development train; large training adds a natural seeded residual sample, so size and composition both change and are not separately identified",
        },
        "isolation": {
            "combined_object_ids_unique": int(combined["dr7objid"].nunique()) == len(combined),
            "combined_asset_ids_unique": int(combined["asset_id"].nunique()) == len(combined),
            "combined_image_members_unique": int(combined["image_member"].nunique()) == len(combined),
            "new_eval_disjoint_from_development_and_historical": True,
        },
        "output_split_sha256": None,
    }
    manifest["output_split_sha256"] = sha256(output_split)
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
