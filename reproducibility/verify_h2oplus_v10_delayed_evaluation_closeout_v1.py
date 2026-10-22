#!/usr/bin/env python3
"""Independently verify the final v10 delayed evaluation closeout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v10_delayed_results"
CLOSEOUT = RESULTS / "evaluation_closeout_v1.json"
OUTPUT = RESULTS / "evaluation_closeout_verification_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _checked_descriptor(item: dict) -> Path:
    path = (REPO / item["path"]).resolve()
    if (
        not path.is_file()
        or REPO not in path.parents
        or path.stat().st_size != item.get("size_bytes")
        or sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"closeout descriptor failed: {item}")
    return path


def verify() -> dict:
    closeout = json.loads(CLOSEOUT.read_text())
    artifact = closeout.get("artifact_integrity", {})
    result = closeout.get("development_scientific_result", {})
    stability = closeout.get("training_scientific_stability", {})
    scheduler = closeout.get("scheduler_closeout", {})
    if (
        closeout.get("schema_version") != "transit-h2oplus-v10-delayed-evaluation-closeout-v1"
        or closeout.get("status") != "closed"
        or closeout.get("passed") is not True
        or artifact.get("passed") is not True
        or artifact.get("physical_evaluation_results") != 540
        or artifact.get("logical_evaluation_rows") != 1260
        or scheduler.get("passed") is not True
        or scheduler.get("tasks") != ["t82706", "t82707"]
        or scheduler.get("token_bound_backend_exit0") is not True
        or stability.get("passed") is not False
        or stability.get("failure_runs") != ["h2oplus_v10c4_delayed_order_seed6205"]
        or result.get("screening_gate_passed") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or closeout.get("supports_formal_held_out_model_effect_claim") is not False
        or closeout.get("formal_confirmation_eligible") is not False
        or closeout.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("independent v10 evaluation closeout scope check failed")
    paths = {
        "audit": _checked_descriptor(artifact["evaluation_audit"]),
        "manifest": _checked_descriptor(artifact["evaluation_file_manifest"]),
        "manifest_verification": _checked_descriptor(
            artifact["evaluation_file_manifest_verification"]
        ),
        "scheduler": _checked_descriptor(scheduler["scheduler_evidence"]),
        "analysis": _checked_descriptor(result["analysis"]),
        "analysis_source": _checked_descriptor(result["analysis_source"]),
        "closeout_builder": _checked_descriptor(closeout["closeout_tools"]["builder"]),
        "closeout_verifier": _checked_descriptor(
            closeout["closeout_tools"]["independent_verifier"]
        ),
    }
    report = json.loads(paths["analysis"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    manifest_verification = json.loads(paths["manifest_verification"].read_text())
    scheduler_evidence = json.loads(paths["scheduler"].read_text())
    checks = report.get("development_scientific_result", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if value is not True)
    passed = sorted(key for key, value in checks.items() if value is True)
    if (
        sha256(paths["analysis_source"])
        != "4542031ccfd73f78001de236e60e6b6df9df36054318669ffbc5558e5437d574"
        or report.get("development_scientific_result", {}).get("screening_gate_passed") is not False
        or report.get("formal_confirmation_eligible") is not False
        or report.get("artifact_integrity", {}).get("physical_evaluation_results") != 540
        or result.get("checks") != checks
        or result.get("failed_checks") != failed
        or result.get("passed_checks") != passed
        or manifest.get("file_records_aggregate_sha256")
        != artifact.get("file_records_aggregate_sha256")
        or manifest_verification.get("manifest", {}).get("sha256")
        != sha256(paths["manifest"])
        or scheduler_evidence.get("token_bound_backend_exit0") is not True
        or set(scheduler_evidence.get("tasks", {})) != {"t82706", "t82707"}
    ):
        raise ValueError("independent v10 closeout identity or gate check failed")
    return {
        "schema_version": "transit-h2oplus-v10-delayed-evaluation-closeout-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "closeout": {
            "path": str(CLOSEOUT.relative_to(REPO)),
            "sha256": sha256(CLOSEOUT),
            "size_bytes": CLOSEOUT.stat().st_size,
        },
        "artifact_integrity_verified": True,
        "physical_evaluation_results_verified": 540,
        "logical_evaluation_rows_verified": 1260,
        "training_scientific_stability_passed": False,
        "training_scientific_stability_verified_separately": True,
        "development_screening_gate_verified": False,
        "failed_checks": failed,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def main() -> int:
    payload = verify()
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(payload["closeout"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
