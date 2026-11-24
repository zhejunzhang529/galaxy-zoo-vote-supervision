#!/usr/bin/env python3
"""Read-only audit of the official Galaxy Zoo 2 label table and image mapping.

The script deliberately reads object identifiers as strings. It does not open,
download, or inspect image files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def task_fraction_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in sorted(
        {m.group(1) for c in labels.columns if (m := re.match(r"(t\d+)_", c))}
    ):
        for representation, suffix in [
            ("raw_fraction", "_fraction"),
            ("weighted_fraction", "_weighted_fraction"),
            ("debiased_fraction", "_debiased"),
        ]:
            if suffix == "_fraction":
                matches = lambda c: c.endswith(suffix) and not c.endswith("_weighted_fraction")
            else:
                matches = lambda c: c.endswith(suffix)
            cols = [c for c in labels.columns if c.startswith(task + "_") and matches(c)]
            if not cols:
                continue
            values = labels[cols].apply(pd.to_numeric, errors="coerce")
            sums = values.sum(axis=1, min_count=1)
            rows.append(
                {
                    "task": task,
                    "representation": representation,
                    "fraction_columns": ";".join(cols),
                    "rows_with_any_fraction": int(values.notna().any(axis=1).sum()),
                    "rows_with_all_fractions": int(values.notna().all(axis=1).sum()),
                    "sum_min": float(sums.min()),
                    "sum_median": float(sums.median()),
                    "sum_max": float(sums.max()),
                    "sum_close_to_one": int(np.isclose(sums, 1.0, atol=1e-5).sum()),
                }
            )
    return pd.DataFrame(rows)


def response_summary(labels: pd.DataFrame, task: str = "t01") -> pd.DataFrame:
    rows = []
    for representation, suffix in [
        ("raw_fraction", "_fraction"),
        ("weighted_fraction", "_weighted_fraction"),
        ("debiased_fraction", "_debiased"),
    ]:
        if suffix == "_fraction":
            matches = lambda c: c.endswith(suffix) and not c.endswith("_weighted_fraction")
        else:
            matches = lambda c: c.endswith(suffix)
        for column in sorted(c for c in labels.columns if c.startswith(task + "_") and matches(c)):
            values = pd.to_numeric(labels[column], errors="coerce")
            rows.append(
                {
                    "task": task,
                    "representation": representation,
                    "column": column,
                    "rows": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "zero_rows": int((values == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / args.config).read_text())
    labels_path = root / cfg["labels_path"]
    mapping_path = root / cfg["mapping_path"]
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(
        labels_path,
        dtype={"dr7objid": "string", "sample": "string", "gz2_class": "string"},
    )
    mapping = pd.read_csv(
        mapping_path,
        dtype={"objid": "string", "sample": "string", "asset_id": "string"},
    )
    labels["dr7objid"] = clean_id(labels["dr7objid"])
    labels["sample"] = clean_id(labels["sample"])
    mapping["objid"] = clean_id(mapping["objid"])
    mapping["sample"] = clean_id(mapping["sample"])
    mapping["asset_id"] = clean_id(mapping["asset_id"])

    label_keys = labels[["dr7objid", "sample"]].rename(columns={"dr7objid": "objid"})
    map_key_counts = mapping.groupby(["objid", "sample"], dropna=False).size()
    label_key_counts = label_keys.groupby(["objid", "sample"], dropna=False).size()
    key_index = pd.MultiIndex.from_frame(label_keys)
    map_counts_for_labels = map_key_counts.reindex(key_index, fill_value=0).to_numpy()

    valid_asset = mapping["asset_id"].str.fullmatch(r"\d+").fillna(False)
    matched_mask = map_counts_for_labels > 0
    unique_match_mask = map_counts_for_labels == 1
    mapping_sample_counts = mapping.groupby("objid")["sample"].nunique(dropna=False)

    join_by_sample = []
    for sample, group in labels.groupby("sample", dropna=False):
        keys = pd.MultiIndex.from_frame(
            group[["dr7objid", "sample"]].rename(columns={"dr7objid": "objid"})
        )
        counts = map_key_counts.reindex(keys, fill_value=0).to_numpy()
        join_by_sample.append(
            {
                "sample": sample,
                "label_rows": int(len(group)),
                "matched_rows": int((counts > 0).sum()),
                "unmatched_rows": int((counts == 0).sum()),
                "exactly_one_mapping_rows": int((counts == 1).sum()),
                "multiple_mapping_rows": int((counts > 1).sum()),
            }
        )

    # Candidate pilot is only a deterministic metadata selection. It does not
    # imply that the corresponding image files have been downloaded.
    preferred = str(cfg.get("pilot_sample_preference", "original")).lower()
    candidate_samples = sorted(
        s for s in mapping["sample"].dropna().unique() if preferred in str(s).lower()
    )
    pilot_sample = candidate_samples[0] if candidate_samples else None
    pilot = labels.copy()
    if pilot_sample is not None:
        pilot = pilot[pilot["sample"].eq(pilot_sample)]
    pilot = pilot.drop_duplicates(subset=["dr7objid"], keep="first")
    pilot = pilot.sort_values(["dr7objid"]).head(int(cfg["pilot_candidate_size"]))

    audit = {
        "source": {
            "labels_path": str(labels_path),
            "mapping_path": str(mapping_path),
            "labels_bytes": labels_path.stat().st_size,
            "mapping_bytes": mapping_path.stat().st_size,
            "labels_sha256": sha256(labels_path),
            "mapping_sha256": sha256(mapping_path),
        },
        "labels": {
            "rows": int(len(labels)),
            "columns": int(len(labels.columns)),
            "dr7objid_dtype": str(labels["dr7objid"].dtype),
            "dr7objid_null": int(labels["dr7objid"].isna().sum()),
            "dr7objid_unique": int(labels["dr7objid"].nunique(dropna=True)),
            "dr7objid_duplicate_rows": int(labels["dr7objid"].duplicated(keep=False).sum()),
            "sample_values": labels["sample"].value_counts(dropna=False).to_dict(),
            "gz2_class_top": labels["gz2_class"].value_counts(dropna=False).head(20).to_dict(),
            "total_classifications": {
                "missing": int(labels["total_classifications"].isna().sum()),
                "min": float(labels["total_classifications"].min()),
                "median": float(labels["total_classifications"].median()),
                "max": float(labels["total_classifications"].max()),
            },
            "total_votes": {
                "missing": int(labels["total_votes"].isna().sum()),
                "min": float(labels["total_votes"].min()),
                "median": float(labels["total_votes"].median()),
                "max": float(labels["total_votes"].max()),
            },
        },
        "mapping": {
            "rows": int(len(mapping)),
            "columns": int(len(mapping.columns)),
            "objid_dtype": str(mapping["objid"].dtype),
            "objid_null": int(mapping["objid"].isna().sum()),
            "objid_unique": int(mapping["objid"].nunique(dropna=True)),
            "objid_duplicate_rows": int(mapping["objid"].duplicated(keep=False).sum()),
            "key_duplicate_rows": int((map_key_counts > 1).sum()),
            "sample_values": mapping["sample"].value_counts(dropna=False).to_dict(),
            "objid_with_multiple_sample_values": int((mapping_sample_counts > 1).sum()),
            "asset_id_numeric_rows": int(valid_asset.sum()),
            "asset_id_invalid_rows": int((~valid_asset).sum()),
            "asset_id_unique": int(mapping.loc[valid_asset, "asset_id"].nunique()),
        },
        "join": {
            "label_rows_with_any_mapping": int(matched_mask.sum()),
            "label_rows_without_mapping": int((~matched_mask).sum()),
            "label_rows_with_exactly_one_mapping": int(unique_match_mask.sum()),
            "label_rows_with_multiple_mappings": int((map_counts_for_labels > 1).sum()),
            "label_key_unique": int(len(label_key_counts)),
            "mapping_keys_matching_labels": int(
                map_key_counts.index.isin(label_key_counts.index).sum()
            ),
            "by_sample": join_by_sample,
        },
        "pilot_candidate": {
            "selected_sample_value": pilot_sample,
            "rows": int(len(pilot)),
            "unique_dr7objid": int(pilot["dr7objid"].nunique()),
            "note": "Metadata-only candidate; image files were not downloaded or checked.",
        },
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    task_fraction_summary(labels).to_csv(out / "task_fraction_summary.csv", index=False)
    response_summary(labels).to_csv(out / "t01_response_summary.csv", index=False)
    labels["gz2_class"].value_counts(dropna=False).rename_axis("gz2_class").reset_index(name="rows").to_csv(
        out / "gz2_class_distribution.csv", index=False
    )
    labels["dr7objid"].value_counts().rename_axis("dr7objid").reset_index(name="rows").to_csv(
        out / "label_object_duplicate_counts.csv", index=False
    )
    mapping["objid"].value_counts().rename_axis("objid").reset_index(name="rows").to_csv(
        out / "mapping_object_duplicate_counts.csv", index=False
    )
    pilot.to_csv(out / "pilot_metadata_candidates.csv", index=False)
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
