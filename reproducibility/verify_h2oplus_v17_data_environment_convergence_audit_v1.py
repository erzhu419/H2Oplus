#!/usr/bin/env python3
"""Independently verify the H2O+ v17 data/environment audit envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_path = args.audit.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output}")
    actual = sha256(audit_path)
    if actual != args.expected_audit_sha256:
        raise ValueError("audit identity drifted")
    audit = json.loads(audit_path.read_text())
    scan = audit.get("independent_dataset_scan", {})
    conclusions = audit.get("conclusions", {})
    risk_ids = [item.get("id") for item in audit.get("risk_register", [])]
    expected_risks = {
        "objective-constraint-mismatch",
        "adaptive-development-cell-reuse",
        "sumo-runtime-version-shift",
        "transition-count-overstates-independent-sample-size",
        "sparse-positive-action-support",
        "ratio-diagnostic-generalization",
        "canonical-feature-clipping",
    }
    checks = {
        "schema": audit.get("schema_version")
        == "transit-h2oplus-v17-data-environment-convergence-audit-v1",
        "development_only": audit.get("study_stage") == "development",
        "no_formal_outcomes": audit.get("formal_held_out_outcomes_read") is False,
        "no_local_training_or_sumo": audit.get("local_training_or_sumo_executed") is False,
        "no_incident_scientific_values": audit.get("v17_incident_numeric_outcomes_inspected") is False,
        "dataset_rows": scan.get("rows") == 3_284_718,
        "dataset_episodes": scan.get("episodes") == 240,
        "independent_cells": scan.get("independent_environment_cells") == 30,
        "finite": scan.get("nonfinite_rows") == 0,
        "reward_ledger": scan.get("maximum_reward_ledger_error") == 0.0,
        "risk_register_complete": set(risk_ids) == expected_risks and len(risk_ids) == len(set(risk_ids)),
        "integrity_passed": conclusions.get("artifact_integrity") == "passed",
        "no_false_convergence_claim": conclusions.get("algorithm_convergence_failure_proven") is False,
        "no_formal_effect_claim": conclusions.get("formal_effect_claim_supported") is False,
        "successor_change_bounded": "Python 3.8 compatibility" in conclusions.get("v17_v2_allowed_change", ""),
    }
    if not all(checks.values()):
        raise ValueError({key: value for key, value in checks.items() if not value})
    payload = {
        "schema_version": "transit-h2oplus-v17-data-environment-convergence-audit-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit": {
            "path": str(audit_path.relative_to(ROOT)),
            "bytes": audit_path.stat().st_size,
            "sha256": actual,
        },
        "checks": checks,
        "passed": True,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "passed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
