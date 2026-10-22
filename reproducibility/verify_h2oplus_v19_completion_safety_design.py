#!/usr/bin/env python3
"""Independently verify the frozen v19 completion-safety design audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "reproducibility/h2oplus_v19_completion_safety_design_audit_v2.json"
DEFAULT_OUTPUT = ROOT / (
    "reproducibility/h2oplus_v19_completion_safety_design_verification_v2.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_descriptor(raw: dict) -> Path:
    path = Path(raw["path"])
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=True)


def main() -> None:
    args = parse_args()
    audit_path = args.audit.expanduser().resolve(strict=True)
    payload = json.loads(audit_path.read_text())
    checks = {
        "schema": payload.get("schema_version")
        == "transit-h2oplus-v19-completion-safety-design-audit-v2",
        "binding_correction_declared": "file-level bindings"
        in payload.get("binding_correction", ""),
        "development_only": payload.get("study_stage") == "development-design",
        "formal_boundary": payload.get("formal_held_out_outcomes_read") is False,
        "no_local_sumo": payload.get("local_sumo_executed") is False,
        "no_local_training": payload.get("local_training_executed") is False,
        "core_implementation_verified": payload.get("claim_boundary", {}).get(
            "core_implementation_verified_without_sumo"
        )
        is True,
        "core_tests_passed": payload.get("implementation_verification", {}).get(
            "tests_passed"
        )
        == 5
        and payload.get("implementation_verification", {}).get("tests_failed")
        == 0
        and payload.get("implementation_verification", {}).get(
            "sumo_import_or_execution_required"
        )
        is False,
        "adaptive_design_disclosed": payload.get(
            "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design"
        )
        is True,
        "v11_fails_current_margin": payload.get("why_gpt55_looked_better", {}).get(
            "v11_would_pass_current_completion_margin"
        )
        is False,
        "dataset_rows": payload.get("dataset_audit", {}).get("rows") == 3_284_718,
        "dataset_episodes": payload.get("dataset_audit", {}).get("episodes") == 240,
        "independent_cells": payload.get("dataset_audit", {}).get(
            "independent_environment_cells"
        )
        == 30,
        "terminal_label_gap_material": payload.get("dataset_audit", {})
        .get("endpoint_label_gap", {})
        .get("max_passengers", 0)
        >= 39,
        "candidate_fixed": payload.get("fixed_successor_design", {}).get(
            "predesignated_candidate"
        )
        == "quarter-load-sixty",
        "candidate_physical_bound": payload.get("fixed_successor_design", {}).get(
            "candidate_floor_seconds"
        )
        == 60.0,
        "fresh_cells_required": payload.get("fixed_successor_design", {}).get(
            "fresh_development_cells_required"
        )
        is True,
        "reach_brackets_gate": payload.get("reach_uncertainty", {}).get(
            "conservative_global_response_projection", 1.0
        )
        < 0.003
        < payload.get("reach_uncertainty", {}).get(
            "targeted_response_projection", 0.0
        ),
        "no_effect_claim": payload.get("claim_boundary", {}).get(
            "formal_effect_claim_supported"
        )
        is False,
    }
    descriptors = [
        *payload.get("frozen_inputs", {}).values(),
        payload.get("evaluator"),
        payload.get("matrix"),
        payload.get("superseded_design_audit"),
        payload.get("dataset_audit", {}).get("descriptor"),
        payload.get("implementation_verification", {}).get("test_source"),
    ]
    descriptor_checks = []
    for raw in descriptors:
        path = resolve_descriptor(raw)
        descriptor_checks.append(
            path.stat().st_size == raw.get("bytes") and sha256(path) == raw.get("sha256")
        )
    checks["all_source_bindings"] = bool(descriptor_checks) and all(descriptor_checks)
    passed = all(checks.values())
    result = {
        "schema_version": "transit-h2oplus-v19-completion-safety-design-verification-v2",
        "audit": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "checks": checks,
        "passed": passed,
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
    }
    if not passed:
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v19 verification: {output}")
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "passed": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
