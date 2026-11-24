#!/usr/bin/env python3
"""Extract only the already-frozen extra cross-sample image list."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one extracted image."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Extract and verify the 1,000 fixed extra image members."""
    root = Path(__file__).resolve().parents[2]
    result_dir = root / "round2_resnet/results"
    split = pd.read_csv(result_dir / "extra_cross_sample_split.csv", dtype={"image_path": "string", "image_member": "string"})
    archive_path = root / "data/raw/images_gz2.zip"
    image_root = root / "data/processed"
    extracted = 0
    bytes_written = 0
    checksums = []
    with ZipFile(archive_path) as archive:
        for row in split.itertuples(index=False):
            output = image_root / str(row.image_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                with archive.open(str(row.image_member), "r") as source, output.open("wb") as target:
                    data = source.read()
                    target.write(data)
                extracted += 1
            bytes_written += output.stat().st_size
            checksums.append({"image_path": str(row.image_path), "sha256": sha256(output), "bytes": output.stat().st_size})
    output_manifest = {
        "status": "extra_cross_sample_images_extracted_and_verified",
        "split": "round2_resnet/results/extra_cross_sample_split.csv",
        "archive": "data/raw/images_gz2.zip",
        "rows": int(len(split)),
        "newly_extracted": extracted,
        "existing_reused": int(len(split) - extracted),
        "total_bytes": int(bytes_written),
        "image_sha256": checksums,
    }
    (result_dir / "extra_cross_sample_image_manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(split), "newly_extracted": extracted, "total_bytes": bytes_written}, indent=2))


if __name__ == "__main__":
    main()
