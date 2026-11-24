#!/usr/bin/env python3
"""Extract new evaluation images after model selection, without changing the split."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "round2_resnet/configs/split.json").read_text())
    split = pd.read_csv(root / "round2_resnet/results/resnet_split.csv", dtype={"dr7objid": "string", "asset_id": "string", "image_member": "string", "image_path": "string", "pool_role": "string"})
    selected = split[split["pool_role"].isin(["new_natural_evaluation", "new_class2_diagnostic_recall_only"])].copy()
    out_dir = root / "data/processed/resnet_round2_eval_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = root / cfg["image_zip"]
    with zipfile.ZipFile(archive_path) as archive:
        for row in selected.itertuples(index=False):
            target = out_dir / f"{row.asset_id}.jpg"
            if not target.exists():
                with archive.open(str(row.image_member), "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(f"missing extracted image: {target}")
    selected["image_path"] = selected["asset_id"].map(lambda value: str(Path("resnet_round2_eval_images") / f"{value}.jpg"))
    output = root / "round2_resnet/results/new_eval_with_images.csv"
    selected.to_csv(output, index=False)
    manifest = {
        "status": "new_eval_images_extracted_after_checkpoint_selection",
        "source_split": "round2_resnet/results/resnet_split.csv",
        "source_split_sha256": sha256(root / "round2_resnet/results/resnet_split.csv"),
        "image_zip": cfg["image_zip"],
        "image_zip_sha256": sha256(archive_path),
        "rows": int(len(selected)),
        "natural_rows": int((selected["pool_role"] == "new_natural_evaluation").sum()),
        "diagnostic_rows": int((selected["pool_role"] == "new_class2_diagnostic_recall_only").sum()),
        "output": str(output.relative_to(root)),
        "output_sha256": sha256(output),
    }
    (root / "round2_resnet/results/new_eval_image_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
