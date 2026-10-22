#!/usr/bin/env python3
"""Freeze v20 formal-confirmation sources without authorizing execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SUMO_DIR = ROOT.parent / "SUMO_ruiguang/online_control"
EDGE_XML = SUMO_DIR / "intersection_delay/a_sorted_busline_edge.xml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256  # noqa: E402
from reproducibility.run_evaluation_matrix_v20_formal_confirmation import (  # noqa: E402
    evaluation_sources,
)


EXPECTED = {
    "authorization": "7a7b79f029a5437a80a31637c1bc057571849623b9b04a92a67bcf66586882c9",
    "rationale": "72e6d2961846e458721f121ae75eded862bed0556c929dc38015a3b2d9103a4d",
    "registration": "c33111cd72b741a801804cd065f42629094d45c0fe03a5b88f9d95dba62149ce",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "environment": "50b831923b76a81c773cf7dc19dcc2e10c729ea8fc7b3be6982fc01b7c3cc141",
    "v20_analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "v20_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "minimal_sync_policy": "20b313b7b8cc13a0a8df7c2a137a4b0713c621a33ef8e2e314fb69d03de1a531",
}
PATHS = {
    "authorization": REPRO
    / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json",
    "rationale": REPRO / "h2oplus_v20_formal_confirmation_rationale_v1.json",
    "registration": REPRO / "h2oplus_v20_formal_confirmation_registration_v1.json",
    "protocol": REPRO / "h2oplus_v20_formal_confirmation_protocol_v1.json",
    "methods": REPRO
    / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json",
    "environment": REPRO / "h2oplus_v20_formal_confirmation_environment_v1.json",
    "design_audit": REPRO / "h2oplus_v20_formal_confirmation_design_audit_v1.json",
    "design_verification": REPRO
    / "h2oplus_v20_formal_confirmation_design_verification_v1.json",
    "v20_analysis": REPRO
    / "h2oplus_v20_full_control_results/full_control_analysis_v1.json",
    "v20_audit": REPRO
    / "h2oplus_v20_full_control_results/evaluation_audit_v1.json",
    "v20_closeout": REPRO
    / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
    "v20_verification": REPRO
    / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
    "data_environment_audit": REPRO
    / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_environment_verification": REPRO
    / "h2oplus_v17_data_environment_convergence_verification_v1.json",
    "minimal_sync_policy": REPRO / "h2oplus_minimal_remote_sync_policy_v1.json",
}
OPERATIONAL_SOURCES = (
    "reproducibility/prepare_h2oplus_v20_formal_confirmation.py",
    "reproducibility/analyze_h2oplus_v20_formal_confirmation.py",
    "reproducibility/audit_h2oplus_v20_formal_confirmation_design.py",
    "reproducibility/verify_h2oplus_v20_formal_confirmation_design.py",
    "reproducibility/build_h2oplus_v20_formal_confirmation_source_freeze.py",
    "reproducibility/sync_h2oplus_v20_formal_confirmation_results.py",
)
DEFAULT_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_source_freeze_v1.json"


def descriptor(path: Path, relative: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": relative if relative is not None else str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def logical_source(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved == ROOT or ROOT in resolved.parents:
        return resolved.relative_to(ROOT).as_posix()
    sumo_root = SUMO_DIR.parent
    if resolved == sumo_root or sumo_root in resolved.parents:
        return "SUMO_ruiguang/" + resolved.relative_to(sumo_root).as_posix()
    raise ValueError(f"unexpected formal-confirmation source root: {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads = {}
    for name, path in PATHS.items():
        resolved = path.resolve(strict=True)
        if name in EXPECTED and sha256(resolved) != EXPECTED[name]:
            raise ValueError(f"frozen formal source input drifted: {name}")
        payloads[name] = json.loads(resolved.read_text())

    authorization = payloads["authorization"]
    scope = authorization.get("authorization_scope", {})
    audit = payloads["design_audit"]
    verification = payloads["design_verification"]
    if (
        scope.get("formal_analysis_source_freeze_authorized") is not True
        or scope.get("formal_confirmation_evaluation_authorized") is not False
        or scope.get("formal_confirmation_submission_authorized") is not False
        or scope.get("formal_held_out_outcome_access_authorized") is not False
        or scope.get("remote_sumo_episode_cap") != 0
        or scope.get("unique_task_count_cap") != 0
        or payloads["registration"].get(
            "formal_confirmation_evaluation_authorized_or_submitted"
        )
        is not False
        or payloads["protocol"].get("evaluation", {}).get("submission_authorized")
        is not False
        or payloads["environment"].get("environment_cells") != 30
        or audit.get("passed") is not True
        or audit.get("scientific_outcomes_available") is not False
        or audit.get("formal_confirmation_submission_authorized") is not False
        or verification.get("passed") is not True
        or verification.get("audit_sha256") != sha256(PATHS["design_audit"])
        or verification.get("formal_confirmation_submission_authorized") is not False
        or payloads["v20_closeout"].get("formal_confirmation_eligible") is not True
        or payloads["v20_closeout"].get(
            "formal_confirmation_authorized_or_submitted"
        )
        is not False
        or payloads["data_environment_audit"].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or payloads["data_environment_verification"].get("passed") is not True
    ):
        raise ValueError("formal source-freeze scope or design audit drifted")

    result_root = REPRO / "h2oplus_v20_formal_confirmation_results"
    if result_root.exists() and list(result_root.rglob("sumo_seed*.json")):
        raise ValueError("refusing to freeze sources after a formal outcome")

    runtime_paths = evaluation_sources(SUMO_DIR, EDGE_XML)
    matrix_sources = [
        descriptor(path, relative=logical_source(path)) for path in runtime_paths
    ]
    if len(matrix_sources) != 57:
        raise ValueError(
            f"formal matrix source closure changed: {len(matrix_sources)} files"
        )
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    protocol = payloads["protocol"]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal source freeze: {output}")
    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation-preparation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "formal_confirmation_evaluation_authorized": False,
        "formal_confirmation_submission_authorized": False,
        "supports_formal_held_out_effect_claim": False,
        "prior_v14_through_v20_development_outcomes_used_for_design": True,
        "new_training_runs": 0,
        "physical_behavior_count": 45,
        "formal_environment_block_count": 10,
        "formal_environment_cell_count": 30,
        "planned_remote_sumo_episodes": 1350,
        "authorized_remote_sumo_episodes": 0,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "analyzer": descriptor(
            ROOT / "reproducibility/analyze_h2oplus_v20_formal_confirmation.py",
            relative="reproducibility/analyze_h2oplus_v20_formal_confirmation.py",
        ),
        "evaluator": descriptor(
            ROOT / "SimpleSAC/evaluate_transit_v20_formal_confirmation.py",
            relative="SimpleSAC/evaluate_transit_v20_formal_confirmation.py",
        ),
        "matrix": descriptor(
            ROOT / "reproducibility/run_evaluation_matrix_v20_formal_confirmation.py",
            relative=(
                "reproducibility/run_evaluation_matrix_v20_formal_confirmation.py"
            ),
        ),
        "result_sync": descriptor(
            ROOT
            / "reproducibility/sync_h2oplus_v20_formal_confirmation_results.py",
            relative=(
                "reproducibility/sync_h2oplus_v20_formal_confirmation_results.py"
            ),
        ),
        "authorization": descriptor(PATHS["authorization"]),
        "rationale": descriptor(PATHS["rationale"]),
        "registration": descriptor(PATHS["registration"]),
        "protocol": descriptor(PATHS["protocol"]),
        "methods": descriptor(PATHS["methods"]),
        "environment_design": descriptor(PATHS["environment"]),
        "design_audit": descriptor(PATHS["design_audit"]),
        "design_verification": descriptor(PATHS["design_verification"]),
        "parent_v20_analysis": descriptor(PATHS["v20_analysis"]),
        "parent_v20_audit": descriptor(PATHS["v20_audit"]),
        "parent_v20_closeout": descriptor(PATHS["v20_closeout"]),
        "parent_v20_verification": descriptor(PATHS["v20_verification"]),
        "data_environment_audit": descriptor(PATHS["data_environment_audit"]),
        "data_environment_verification": descriptor(
            PATHS["data_environment_verification"]
        ),
        "minimal_remote_sync_policy": descriptor(PATHS["minimal_sync_policy"]),
        "dataset": protocol["dataset"],
        "sumo_runtime_lock": protocol["sumo_runtime_lock"],
        "analysis_operationalization": {
            "predesignated_candidate": (
                "h2oplus-v19-completion-safety-quarter-load-sixty"
            ),
            "direct_behavior_parent": "h2oplus-v19-completion-safety-baseline",
            "raw_actor_parent": "h2oplus-v14-terminal-quarter",
            "independent_lineage_control_count": 6,
            "independent_analysis_unit": "formal-environment-block",
            "minimum_block_wins": 9,
            "one_sided_alpha": 0.025,
            "completion_noninferiority_margin": 0.003,
            "endpoint_only": True,
            "aulc_computed_or_claimed": False,
            "formal_confirmation_submission_authorized": False,
        },
        "remote_sync_scope": {
            "results_matrix_manifests_and_logs_only": True,
            "default_logs_included": False,
            "remote_inventory_required_before_transfer": True,
            "rsync_files_from_allowlist_required": True,
            "checkpoint_optimizer_replay_or_workspace_sync": False,
        },
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "matrix_sources": len(matrix_sources),
                "operational_sources": len(operational_sources),
                "formal_confirmation_submission_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
