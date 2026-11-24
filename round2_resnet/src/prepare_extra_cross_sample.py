#!/usr/bin/env python3
"""Freeze an object-level GZ2-extra cross-sample evaluation manifest.

The source is the official GZ2 ``extra`` sample already present in the local
Hart16 catalogue, mapping table, and image archive. Selection is object-level,
without class quotas, and excludes every object, asset, and image member listed
by the retained development or evaluation manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


RAW = [
    "raw_smooth_normalized",
    "raw_features_or_disk_normalized",
    "raw_star_or_artifact_normalized",
]
SEED = 20261122
MAX_OBJECTS = 1000


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exclusion_manifests(root: Path) -> tuple[set[str], set[str], set[str]]:
    """Collect object, asset, and image identifiers from frozen manifests."""
    paths = [
        root / "data/processed/pilot_split.csv",
        root / "data/splits/expanded_split.csv",
        root / "data/splits/independent_eval_split.csv",
        root / "round2_resnet/results/resnet_split.csv",
        root / "round2_resnet/results/new_eval_with_images.csv",
    ]
    object_ids: set[str] = set()
    asset_ids: set[str] = set()
    image_members: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            dtype={
                "dr7objid": "string",
                "asset_id": "string",
                "image_member": "string",
                "image_path": "string",
            },
        )
        if "dr7objid" in frame:
            object_ids.update(frame["dr7objid"].dropna().astype(str))
        if "asset_id" in frame:
            asset_ids.update(frame["asset_id"].dropna().astype(str))
        image_column = "image_member" if "image_member" in frame else "image_path"
        if image_column in frame:
            image_members.update(frame[image_column].dropna().astype(str))
    return object_ids, asset_ids, image_members


def archive_members(path: Path) -> dict[str, str]:
    """Map unique JPG stems to archive members, excluding macOS metadata."""
    members: dict[str, str] = {}
    duplicates: set[str] = set()
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if name.startswith("__MACOSX/") or not name.lower().endswith(".jpg"):
                continue
            stem = Path(name).stem
            if stem in members:
                duplicates.add(stem)
            else:
                members[stem] = name
    for stem in duplicates:
        members.pop(stem, None)
    return members


def checkpoint_metadata(root: Path) -> list[dict]:
    """Return the six frozen 10,000-object checkpoint paths and hashes."""
    formal = json.loads((root / "round2_resnet/results/formal_runs.json").read_text())
    selected = [record for record in formal["records"] if record["train_size"] == "large_10000"]
    return [
        {
            "train_size": record["train_size"],
            "target": record["target"],
            "seed": int(record["seed"]),
            "selected_epoch": int(record["selected_epoch"]),
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        }
        for record in sorted(selected, key=lambda item: (int(item["seed"]), item["target"]))
    ]


def main() -> None:
    """Build the fixed extra manifest and write its audit metadata."""
    root = Path(__file__).resolve().parents[2]
    labels_path = root / "data/processed/normalized_t01_labels.csv.gz"
    mapping_path = root / "data/raw/gz2_filename_mapping.csv"
    archive_path = root / "data/raw/images_gz2.zip"
    labels = pd.read_csv(
        labels_path,
        dtype={"dr7objid": "string", "sample": "string", "gz2_class": "string"},
    )
    mapping = pd.read_csv(
        mapping_path,
        dtype={"objid": "string", "sample": "string", "asset_id": "string"},
    )
    labels = labels[labels["sample"].eq("extra")].copy()
    mapping = mapping[mapping["sample"].eq("extra")].copy()
    joined = labels.merge(
        mapping.rename(columns={"objid": "dr7objid"}),
        on=["dr7objid", "sample"],
        how="inner",
        validate="one_to_one",
    )
    members = archive_members(archive_path)
    joined["image_member"] = joined["asset_id"].astype(str).map(members)
    joined = joined[joined["image_member"].notna()].copy()
    object_ids, asset_ids, image_members = read_exclusion_manifests(root)
    eligible = joined[
        ~joined["dr7objid"].astype(str).isin(object_ids)
        & ~joined["asset_id"].astype(str).isin(asset_ids)
        & ~joined["image_member"].astype(str).isin(image_members)
    ].copy()
    eligible = eligible.drop_duplicates(["dr7objid", "asset_id", "image_member"])
    rng = np.random.default_rng(SEED)
    n_selected = min(MAX_OBJECTS, len(eligible))
    selected_positions = rng.choice(len(eligible), size=n_selected, replace=False)
    selected = eligible.iloc[selected_positions].copy().reset_index(drop=True)
    selected["selection_rank"] = np.arange(len(selected), dtype=int)
    selected["image_path"] = selected.apply(
        lambda row: f"extra_round2_images/{int(row['selection_rank']):04d}_{row['asset_id']}.jpg",
        axis=1,
    )
    selected["pool_role"] = "extra_cross_sample_eval"
    selected["small_train"] = False
    selected["large_train"] = False
    selected["hard_class"] = selected[RAW].to_numpy(dtype=float).argmax(axis=1)
    columns = [
        "dr7objid",
        "sample",
        "asset_id",
        "image_member",
        "image_path",
        "pool_role",
        "small_train",
        "large_train",
        "hard_class",
        *RAW,
        "weighted_smooth_normalized",
        "weighted_features_or_disk_normalized",
        "weighted_star_or_artifact_normalized",
        "gz2_class",
        "total_votes",
        "selection_rank",
    ]
    output_dir = root / "round2_resnet/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(output_dir / "extra_cross_sample_split.csv", index=False)
    checkpoint_info = checkpoint_metadata(root)
    manifest = {
        "status": "frozen_before_extra_cross_sample_inference",
        "research_label": "original-to-extra cross-sample generalisation under selection shift",
        "source_sample": "extra",
        "source_definition": "GZ2 extra Legacy objects added after the original sample because the original query omitted objects carrying both BLENDED and CHILD flags; this is a GZ2 sample distinction, not an OOD proof",
        "official_sources": [
            "https://data.galaxyzoo.org/",
            "https://academic.oup.com/mnras/article/435/4/2835/1022913",
            "https://academic.oup.com/mnras/article/509/3/3966/6378289",
            "https://doi.org/10.5281/zenodo.3565488",
        ],
        "labels": {
            "path": "data/processed/normalized_t01_labels.csv.gz",
            "sha256": sha256(labels_path),
            "target": "raw_smooth_normalized, raw_features_or_disk_normalized, raw_star_or_artifact_normalized",
            "raw_votes_are": "volunteer response fractions for the image, not physical truth",
        },
        "mapping": {"path": "data/raw/gz2_filename_mapping.csv", "sha256": sha256(mapping_path)},
        "image_archive": {"path": "data/raw/images_gz2.zip", "sha256": sha256(archive_path)},
        "candidate_audit": {
            "label_rows_extra": int(len(labels)),
            "mapping_rows_extra": int(len(mapping)),
            "joined_rows": int(len(labels.merge(mapping.rename(columns={"objid": "dr7objid"}), on=["dr7objid", "sample"], how="inner"))),
            "unique_jpg_stems_in_archive": int(len(members)),
            "joined_rows_with_archive_jpg": int(len(joined)),
            "excluded_object_ids": int(len(object_ids)),
            "excluded_asset_ids": int(len(asset_ids)),
            "excluded_image_members": int(len(image_members)),
            "eligible_after_all_exclusions": int(len(eligible)),
        },
        "selection": {
            "seed": SEED,
            "rng": "numpy.random.default_rng, PCG64",
            "replacement": False,
            "class_quota": None,
            "max_objects": MAX_OBJECTS,
            "selected_objects": int(len(selected)),
            "selection_order": "RNG order retained in selection_rank; no performance result was inspected",
            "split_path": "round2_resnet/results/extra_cross_sample_split.csv",
        },
        "exclusion_policy": "exclude object ID, asset ID, and archive image member appearing in every retained development, validation, historical evaluation, and current natural/diagnostic evaluation manifest",
        "preprocessing": {
            "image_size": 128,
            "resize": "PIL bilinear",
            "normalization": "ImageNet mean/std recorded in configs/formal.json",
            "prediction_rule": "argmax of three-class softmax for hard diagnostics; no threshold/calibration change",
        },
        "checkpoints": checkpoint_info,
        "evaluation_policy": "score frozen large_10000 hard/soft checkpoints for three paired seeds; no retraining, tuning, recalibration, or checkpoint reselection",
        "timing_policy": "measure a small fixed batch before full inference; total inference must remain within 10 minutes",
    }
    (output_dir / "extra_cross_sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output_dir / 'extra_cross_sample_manifest.json'), "selected": len(selected), "eligible": len(eligible)}, indent=2))


if __name__ == "__main__":
    main()
