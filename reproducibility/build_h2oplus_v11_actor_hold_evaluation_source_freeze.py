#!/usr/bin/env python3
"""Freeze v11 evaluator, matrix, dispatch, and analysis before outcomes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256


SOURCES = (
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_h2oplus_v11_actor_hold_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v11_actor_hold.py",
)


def descriptor(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def read_bound(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"identity drift: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema drift: {resolved}")
    return resolved, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-closeout", type=Path, required=True)
    parser.add_argument("--training-verification", type=Path, required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--expected-registration-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-training-closeout-sha256", required=True)
    parser.add_argument("--expected-training-verification-sha256", required=True)
    parser.add_argument("--expected-methods-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v11_actor_hold_evaluation_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registration_path, registration = read_bound(
        args.registration,
        args.expected_registration_sha256,
        "transit-h2oplus-v11-actor-hold-registration-v2",
    )
    protocol_path, protocol = read_bound(
        args.protocol,
        args.expected_protocol_sha256,
        "transit-h2oplus-v11-actor-hold-protocol-v2",
    )
    closeout_path, closeout = read_bound(
        args.training_closeout,
        args.expected_training_closeout_sha256,
        "transit-h2oplus-v11-actor-hold-training-closeout-v2",
    )
    verification_path, verification = read_bound(
        args.training_verification,
        args.expected_training_verification_sha256,
        "transit-h2oplus-v11-actor-hold-training-closeout-verification-v2",
    )
    methods_path, methods = read_bound(
        args.methods_manifest,
        args.expected_methods_sha256,
        "transit-evaluation-methods-v11-actor-hold-v1",
    )
    development = registration.get("development_evaluation", {})
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or development.get("logical_policy_aliases") != 80
        or development.get("maximum_unique_remote_sumo_episodes") != 720
        or protocol.get("formal_held_out_outcomes_read") is not False
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("all_5_training_runs_stability_gate_passed") is not True
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("development_evaluation_submitted") is not False
        or verification.get("passed") is not True
        or verification.get("artifact_integrity_verified") is not True
        or methods.get("physical_method_count") != 45
        or methods.get("logical_method_budget_alias_count") != 80
        or methods.get("expected_remote_sumo_episodes") != 405
        or methods.get("development_evaluation_outcomes_read") is not False
    ):
        raise ValueError("v11 source-freeze inputs drifted")
    sources = []
    for relative in SOURCES:
        path = (ROOT / relative).resolve(strict=True)
        sources.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    payload = {
        "schema_version": "transit-h2oplus-v11-actor-hold-evaluation-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v11-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "evaluation_outcomes_read": False,
        "physical_actor_count": 45,
        "logical_alias_count": 80,
        "expected_remote_sumo_episodes": 405,
        "sources": sources,
        "source_file_count": len(sources),
        "analysis_operationalization": {
            "primary_budget_axis": "same online events",
            "aulc_budgets": [0, 5000, 10000, 25000],
            "candidate": "h2oplus-v11-actor-hold",
            "controls": [
                "sac-online",
                "h2oplus",
                "h2oplus-v10-delayed-actor",
            ],
            "all_registration_gates_applied_without_post_outcome_changes": True,
        },
        "training_stability": {
            "all_5_passed": True,
            "failure_runs": [],
            "separate_from_artifact_integrity": True,
        },
        "registration": descriptor(registration_path),
        "protocol": descriptor(protocol_path),
        "training_closeout": descriptor(closeout_path),
        "training_verification": descriptor(verification_path),
        "methods_manifest": descriptor(methods_path),
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps({"output": str(destination), "sha256": sha256(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
