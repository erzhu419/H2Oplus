#!/usr/bin/env python3
"""Independently verify the frozen v20 formal-confirmation design audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEFAULT_AUDIT = REPRO / "h2oplus_v20_formal_confirmation_design_audit_v1.json"
DEFAULT_OUTPUT = (
    REPRO / "h2oplus_v20_formal_confirmation_design_verification_v1.json"
)


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
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=True)


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
            path.stat().st_size == raw.get("bytes")
            and sha256(path) == raw.get("sha256")
        )

    methods = audit.get("methods", {})
    environment = audit.get("environment", {})
    accounting = audit.get("episode_accounting", {})
    analysis = audit.get("confirmatory_analysis", {})
    checks = {
        "schema": audit.get("schema_version")
        == "transit-h2oplus-v20-formal-confirmation-design-audit-v1",
        "audit_passed": audit.get("passed") is True,
        "preparation_only": audit.get("study_stage")
        == "formal-confirmation-preparation",
        "matrix_type": audit.get("matrix_type")
        == "endpoint-only-formal-confirmation",
        "formal_outcomes_unread": audit.get("formal_held_out_outcomes_read")
        is False
        and audit.get("formal_confirmation_outcomes_read") is False,
        "evaluation_not_authorized": audit.get(
            "formal_confirmation_evaluation_authorized"
        )
        is False,
        "submission_not_authorized": audit.get(
            "formal_confirmation_submission_authorized"
        )
        is False,
        "no_local_sumo_or_training": audit.get("local_sumo_executed") is False
        and audit.get("local_training_executed") is False,
        "adaptive_design_disclosed": audit.get(
            "prior_v14_through_v20_development_outcomes_used_for_design"
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
        "fresh_environment": environment.get("block_count") == 10
        and environment.get("environment_cell_count") == 30
        and environment.get("sumo_seeds") == list(range(8101, 8111))
        and environment.get("demand_realization_seeds") == list(range(8201, 8211))
        and environment.get("demand_scales") == [0.75, 1.0, 1.25]
        and environment.get("verified_disjoint") is True
        and environment.get("prior_used_formal_v7_environment_reused") is False
        and environment.get("nonempty_overlap_fields") == {},
        "episode_accounting": accounting.get("physical_methods") == 45
        and accounting.get("environment_cells") == 30
        and accounting.get("planned_remote_sumo_episodes") == 1350
        and accounting.get("matrix_manifests") == 10
        and accounting.get("jobs_per_manifest") == 135
        and accounting.get("authorized_remote_sumo_episodes") == 0,
        "confirmatory_test": analysis.get("independent_environment_blocks") == 10
        and analysis.get("paired_observations_averaged_per_block") == 15
        and analysis.get("minimum_directional_block_wins") == 9
        and analysis.get("one_sided_alpha") == 0.025
        and analysis.get("completion_noninferiority_margin") == 0.003
        and analysis.get("multiplicity") == "intersection-union"
        and analysis.get("claim_limited_to_direct_behavior_parent") is True,
        "parent_v20_evidence": audit.get(
            "parent_v20_artifact_integrity_and_training_stability_passed"
        )
        is True,
        "validity_risks_disclosed": audit.get(
            "material_data_environment_validity_factors_remain"
        )
        is True,
        "minimal_sync_bound": audit.get("minimal_remote_sync_policy_bound") is True,
        "no_scientific_outcome": audit.get("scientific_outcomes_available")
        is False,
        "no_formal_effect_claim": audit.get(
            "supports_formal_held_out_effect_claim"
        )
        is False,
        "all_file_bindings": bool(descriptor_checks) and all(descriptor_checks),
    }
    passed = all(checks.values())
    result = {
        "schema_version": (
            "transit-h2oplus-v20-formal-confirmation-design-verification-v1"
        ),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "audit": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "checks": checks,
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "formal_confirmation_evaluation_authorized": False,
        "formal_confirmation_submission_authorized": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
        "supports_formal_held_out_effect_claim": False,
    }
    if not passed:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite formal design verification: {output}"
        )
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "passed": True,
                "formal_confirmation_submission_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
