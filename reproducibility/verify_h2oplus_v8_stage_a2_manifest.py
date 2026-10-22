#!/usr/bin/env python3
"""Independently verify the H2O+ v8 Stage A2 artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO / "reproducibility/h2oplus_v8_stage_a2_results/stage_a2_file_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "reproducibility/h2oplus_v8_stage_a2_results/stage_a2_file_manifest_verification.json"
)
CANONICAL = REPO / "reproducibility/h2oplus_v8_stage_a2_results/canonical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return False


def aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda row: row["path"]):
        digest.update(item["path"].encode())
        digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(item["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-h2oplus-v8-stage-a2-file-manifest-v1":
        raise ValueError("unexpected Stage A2 manifest schema")
    records = manifest.get("files", [])
    if len(records) != manifest.get("file_count") or len({row["path"] for row in records}) != len(records):
        raise ValueError("Stage A2 file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root file: {record['path']}")
        size = path.stat().st_size
        digest = sha256(path)
        if size != record["size_bytes"] or digest != record["sha256"]:
            raise ValueError(f"file identity mismatch: {record['path']}")
        total += size
    if total != manifest.get("total_size_bytes") or aggregate(records) != manifest.get(
        "file_records_aggregate_sha256"
    ):
        raise ValueError("Stage A2 file-record aggregate mismatch")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 150
        or artifact.get("unique_job_signature_count") != 150
        or artifact.get("canonical_matrix_count") != 5
        or artifact.get("source_manifest_count") != 10
    ):
        raise ValueError("Stage A2 artifact summary failed")
    if (
        manifest.get("study_stage") != "development"
        or manifest.get("training") is not False
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
        or manifest.get("scientific_analysis", {}).get("stage_b_authorized_or_submitted") is not False
    ):
        raise ValueError("Stage A2 scientific scope drifted")

    signatures: set[str] = set()
    outputs: set[Path] = set()
    source_manifest_hashes: set[str] = set()
    expected_cells = {
        (algorithm, seed, scale)
        for algorithm in ("sac-mixed", "sac-online")
        for seed in range(5101, 5106)
        for scale in (0.85, 1.0, 1.15)
    }
    for index in range(1, 6):
        block = f"v8-development-{index:02d}"
        path = CANONICAL / block / "matrix_manifest.json"
        matrix = json.loads(path.read_text())
        jobs = matrix.get("jobs", [])
        merge = matrix.get("matrix_merge", {})
        if (
            len(jobs) != 30
            or merge.get("all_outputs_verified") is not True
            or merge.get("job_count") != 30
            or len(merge.get("source_manifests", [])) != 2
        ):
            raise ValueError(f"independent canonical matrix check failed: {block}")
        source_manifest_hashes.update(
            str(item.get("sha256")) for item in merge["source_manifests"]
        )
        cells = set()
        for job in jobs:
            method = job["method"]
            cell = (
                str(method["algorithm"]),
                int(method["training_seed"]),
                float(job["scenario"]["demand_scale"]),
            )
            cells.add(cell)
            signature = str(job["signature"])
            output = Path(job["output"]).resolve()
            payload = json.loads(output.read_text())
            if (
                signature in signatures
                or output in outputs
                or payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != signature
                or not finite(payload)
            ):
                raise ValueError(f"independent result check failed: {output}")
            signatures.add(signature)
            outputs.add(output)
        if cells != expected_cells:
            raise ValueError(f"independent Cartesian coverage failed: {block}")
    if len(signatures) != 150 or len(outputs) != 150 or len(source_manifest_hashes) != 10:
        raise ValueError("independent Stage A2 coverage failed")

    return {
        "schema_version": "transit-h2oplus-v8-stage-a2-file-manifest-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
            "file_records_aggregate_sha256": manifest[
                "file_records_aggregate_sha256"
            ],
        },
        "independent_artifact_audit": {
            "passed": True,
            "file_count": len(records),
            "total_size_bytes": total,
            "canonical_matrix_count": 5,
            "source_partition_result_count": 150,
            "unique_job_signature_count": len(signatures),
            "source_manifest_count": len(source_manifest_hashes),
            "all_numeric_values_finite": True,
        },
        "scientific_analysis_completed": False,
        "stage_b_authorized_or_submitted": False,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    payload = verify(manifest)
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
