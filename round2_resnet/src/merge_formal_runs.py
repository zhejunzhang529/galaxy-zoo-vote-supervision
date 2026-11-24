#!/usr/bin/env python3
"""Merge corrected checkpoint repairs with the three already-valid epoch-3 runs."""

from __future__ import annotations

import json
from pathlib import Path


def key(record: dict) -> tuple[str, str, int]:
    return record["train_size"], record["target"], int(record["seed"])


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    old_path = root / "round2_resnet/results/invalid_final_epoch_attempt/formal_runs.json"
    repair_path = root / "round2_resnet/results/formal_repair_runs.json"
    old = json.loads(old_path.read_text())
    repaired = json.loads(repair_path.read_text())
    valid = {}
    for record in old["records"]:
        if int(record["selected_epoch"]) == 3:
            record["checkpoint_state_epoch"] = 3
            valid[key(record)] = record
    if len(valid) != 3:
        raise RuntimeError(f"expected three epoch-3 reusable records, found {len(valid)}")
    merged = {**valid, **{key(record): record for record in repaired["records"]}}
    expected = {(size, target, int(seed)) for size in old["config"]["train_sizes"] for target in old["config"]["targets"] for seed in old["config"]["seeds"]}
    if set(merged) != expected:
        raise RuntimeError(f"merged keys do not match expected keys: {sorted(expected ^ set(merged))}")
    ordered = [merged[(size, target, int(seed))] for seed in old["config"]["seeds"] for size in old["config"]["train_sizes"] for target in old["config"]["targets"]]
    output = {
        "status": "formal_paired_development_runs_complete_with_selected_checkpoints",
        "policy": "new evaluation pools were not loaded; no test-driven selection or tuning",
        "repair_note": "The first attempt saved final-epoch weights despite recording the selected epoch. Three records selected epoch 3 and their saved final weights were reusable; the other nine records were rerun with selected-state capture. The first attempt remains under results/invalid_final_epoch_attempt/.",
        "config": old["config"],
        "environment": repaired["environment"],
        "weights": repaired["weights"],
        "records": ordered,
    }
    output_path = root / "round2_resnet/results/formal_runs.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "records": len(ordered), "valid_reused": len(valid), "repaired": len(repaired["records"])}, indent=2))


if __name__ == "__main__":
    main()
