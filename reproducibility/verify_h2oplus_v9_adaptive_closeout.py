#!/usr/bin/env python3
"""Independently verify the H2O+ v9 adaptive development closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v9_adaptive_results"
MANIFEST = RESULTS / "adaptive_closeout_file_manifest_v3.json"
OUTPUT = RESULTS / "adaptive_closeout_verification_v3.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    records = manifest.get("files", [])
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v9-adaptive-closeout-file-manifest-v3"
        or len(records) != manifest.get("file_count")
        or len({item["path"] for item in records}) != len(records)
    ):
        raise ValueError("v9 closeout file manifest structure failed")
    total = 0
    for item in records:
        path = (REPO / item["path"]).resolve()
        if (
            not path.is_file()
            or REPO not in path.parents
            or path.stat().st_size != item["size_bytes"]
            or sha256(path) != item["sha256"]
        ):
            raise ValueError(f"v9 closeout file identity failed: {item['path']}")
        total += path.stat().st_size
    if (
        total != manifest.get("total_size_bytes")
        or aggregate(records) != manifest.get("file_records_aggregate_sha256")
        or manifest.get("artifact_integrity_passed") is not True
        or manifest.get("development_screening_gate_passed") is not False
        or manifest.get("formal_confirmation_eligible") is not False
        or manifest.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v9 closeout file aggregate or scope failed")

    closeout_path = RESULTS / "adaptive_closeout_v3.json"
    closeout = json.loads(closeout_path.read_text())
    analysis = json.loads((RESULTS / "adaptive_analysis_v3.json").read_text())
    correction = json.loads(
        (RESULTS / "analysis_serialization_correction_v1.json").read_text()
    )
    evaluation_verification = json.loads(
        (RESULTS / "evaluation_file_manifest_verification_v3.json").read_text()
    )
    checks = analysis.get("development_scientific_result", {}).get("checks", {})
    if (
        closeout.get("schema_version")
        != "transit-h2oplus-v9-adaptive-closeout-v3"
        or closeout.get("passed") is not True
        or closeout.get("artifact_integrity", {}).get("passed") is not True
        or closeout.get("training_stability", {}).get("scientific_stability_passed")
        is not False
        or closeout.get("training_stability", {}).get(
            "adaptation_seed6202_late_stability_passed"
        )
        is not False
        or closeout.get("development_scientific_result", {}).get(
            "screening_gate_passed"
        )
        is not False
        or closeout.get("development_scientific_result", {}).get("checks") != checks
        or all(checks.values()) is not False
        or closeout.get("decision", {}).get("formal_confirmation_eligible") is not False
        or closeout.get("decision", {}).get(
            "formal_confirmation_authorized_or_submitted"
        )
        is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_current_formal_held_out_model_effect_claim")
        is not False
        or correction.get("passed") is not True
        or correction.get("correction_scope", {}).get("analysis_metrics_changed")
        is not False
        or correction.get("report", {}).get("sha256")
        != sha256(RESULTS / "adaptive_analysis_v3.json")
        or evaluation_verification.get("passed") is not True
    ):
        raise ValueError("independent v9 closeout content verification failed")
    return {
        "schema_version": "transit-h2oplus-v9-adaptive-closeout-verification-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(MANIFEST.relative_to(REPO)),
            "sha256": sha256(MANIFEST),
            "file_records_aggregate_sha256": manifest[
                "file_records_aggregate_sha256"
            ],
        },
        "closeout": {
            "path": str(closeout_path.relative_to(REPO)),
            "sha256": sha256(closeout_path),
        },
        "independent_artifact_integrity_passed": True,
        "development_screening_gate_passed": False,
        "training_scientific_stability_passed": False,
        "adaptation_seed6202_late_stability_passed": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "passed": True,
    }


def main() -> int:
    payload = verify()
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
