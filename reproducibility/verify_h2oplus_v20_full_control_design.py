#!/usr/bin/env python3
"""Independently verify the frozen v20 no-SUMO design audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEFAULT_AUDIT = REPRO / "h2oplus_v20_full_control_design_audit_v1.json"
DEFAULT_OUTPUT = REPRO / "h2oplus_v20_full_control_design_verification_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(raw: dict) -> Path:
    path = Path(raw["path"])
    return (ROOT / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def main() -> int:
    args = parse_args()
    audit_path = args.audit.expanduser().resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    descriptors = [
        *audit.get("frozen_inputs", {}).values(),
        *audit.get("environment", {}).get("scenario_manifests", []),
    ]
    descriptor_checks = []
    for raw in descriptors:
        path = resolve(raw)
        descriptor_checks.append(
            path.stat().st_size == raw.get("bytes") and sha256(path) == raw.get("sha256")
        )
    methods = audit.get("methods", {})
    environment = audit.get("environment", {})
    accounting = audit.get("episode_accounting", {})
    checks = {
        "schema": audit.get("schema_version")
        == "transit-h2oplus-v20-full-control-design-audit-v1",
        "audit_passed": audit.get("passed") is True,
        "development_only": audit.get("study_stage") == "development-design",
        "formal_outcomes_unread": audit.get("formal_held_out_outcomes_read") is False,
        "v20_outcomes_unread": audit.get("v20_development_evaluation_outcomes_read") is False,
        "no_local_sumo": audit.get("local_sumo_executed") is False,
        "no_local_training": audit.get("local_training_executed") is False,
        "adaptive_design_disclosed": audit.get(
            "prior_v14_through_v19_development_outcomes_used_for_design"
        )
        is True,
        "method_grid": methods.get("physical_method_count") == 45
        and methods.get("algorithm_count") == 9
        and methods.get("algorithm_seed_grid_complete") is True,
        "actor_lineage": methods.get(
            "candidate_direct_and_raw_parent_actor_matched_by_seed"
        )
        is True,
        "no_behavior_selection": methods.get("checkpoint_selection_by_behavior")
        is False,
        "fresh_environment": environment.get("environment_cell_count") == 9
        and environment.get("verified_disjoint") is True
        and environment.get("nonempty_overlap_fields") == {},
        "episode_accounting": accounting.get("physical_methods") == 45
        and accounting.get("environment_cells") == 9
        and accounting.get("expected_remote_sumo_episodes") == 405
        and accounting.get("matrix_manifests") == 3
        and accounting.get("jobs_per_manifest") == 135,
        "parent_evidence": audit.get(
            "parent_v19_integrity_and_stability_explicitly_passed"
        )
        is True,
        "validity_risks_disclosed": audit.get(
            "material_data_environment_validity_factors_remain"
        )
        is True,
        "no_scientific_outcome": audit.get("scientific_outcomes_available") is False,
        "formal_not_eligible": audit.get("formal_confirmation_eligible") is False,
        "formal_not_authorized": audit.get(
            "formal_confirmation_authorized_or_submitted"
        )
        is False,
        "all_file_bindings": bool(descriptor_checks) and all(descriptor_checks),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "transit-h2oplus-v20-full-control-design-verification-v1",
        "audit": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "checks": checks,
        "passed": passed,
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized_or_submitted": False,
    }
    if not passed:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v20 design verification: {output}")
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(encoded).hexdigest(), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
