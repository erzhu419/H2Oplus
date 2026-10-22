#!/usr/bin/env python3
"""Freeze v20 scientific and operational sources before any v20 outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"

MATRIX_SOURCES = tuple(
    [
        f"SimpleSAC/evaluate_transit_v{version}_{name}.py"
        for version, name in (
            (11, "actor_hold"),
            (12, "staged_release"),
            (13, "proximal_release"),
            (14, "terminal_taper"),
            (15, "hold_calibration"),
            (16, "headway_reserve"),
            (17, "tail_service_reserve"),
            (19, "completion_safety_reserve"),
            (20, "full_control"),
        )
    ]
    + [
        f"reproducibility/run_evaluation_matrix_v{version}_{name}.py"
        for version, name in (
            (11, "actor_hold"),
            (12, "staged_release"),
            (13, "proximal_release"),
            (14, "terminal_taper"),
            (15, "hold_calibration"),
            (16, "headway_reserve"),
            (17, "tail_service_reserve"),
            (19, "completion_safety"),
            (20, "full_control"),
        )
    ]
)
OPERATIONAL_SOURCES = (
    "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v19_completion_safety.py",
    "reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v20_full_control.py",
    "reproducibility/audit_h2oplus_v20_full_control_design.py",
    "reproducibility/verify_h2oplus_v20_full_control_design.py",
    "reproducibility/run_h2oplus_v20_full_control_evaluation_dispatch.py",
)
EXPECTED = {
    "authorization": "ab535766ffad546c07e4291bfd2f0519d7cfa0e83f4cb3f4022c369d4cb19a06",
    "rationale": "e93a4114760e8755d27dcd7042ca666f4aefdf17e03d3bb55990068cf552998c",
    "registration": "f930a3612720097d4f5e5b635501405245f04ffd54445f389a23800d06f7f7d8",
    "protocol": "7211d32aeb1416840de442ca43fa997a918dfa282afcec5009091f78834ffa5d",
    "methods": "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9",
    "environment": "7210826548c94aaa8fbbbb73b62eec5450fec131f2ae3dd1d1dbc93922dff2ba",
    "design_audit": "0c583278b57eee5b5d479db78d294643f11b5d04a51840e35eeaf97e14a0015c",
    "design_verification": "66cc5133fa2a25bc323f880d148dba282db9708161af3df4c5f764541c3d70b9",
    "v19_audit": "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    "v19_closeout": "3e95e4cf964983257fd3fabc5cab2987e1a249b5b9a25d3ae455f081932b3f74",
    "v19_verification": "ec44263eebc8ba61049973f34aadf5154d0b07f5ccc096d3659662002b95f9ab",
    "v19_methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "v14_methods": "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, relative: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": relative if relative is not None else str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v20_full_control_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO / "h2oplus_v20_full_control_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v20_full_control_rationale_v1.json",
        "registration": REPRO / "h2oplus_v20_full_control_registration_v1.json",
        "protocol": REPRO / "h2oplus_v20_full_control_protocol_v1.json",
        "methods": REPRO / "h2oplus_v20_full_control_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v20_full_control_environment_v1.json",
        "design_audit": REPRO / "h2oplus_v20_full_control_design_audit_v1.json",
        "design_verification": REPRO / "h2oplus_v20_full_control_design_verification_v1.json",
        "v19_audit": REPRO / "h2oplus_v19_completion_safety_results/evaluation_audit_v2.json",
        "v19_closeout": REPRO / "h2oplus_v19_completion_safety_results/evaluation_closeout_v2.json",
        "v19_verification": REPRO / (
            "h2oplus_v19_completion_safety_results/"
            "evaluation_closeout_verification_v2.json"
        ),
        "v19_methods": REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json",
        "v14_methods": REPRO / "h2oplus_v14_terminal_taper_evaluation_methods_v1.json",
        "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO / (
            "h2oplus_v17_data_environment_convergence_verification_v1.json"
        ),
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }
    for name, path in paths.items():
        observed = sha256(path.resolve(strict=True))
        if observed != EXPECTED[name]:
            raise ValueError(f"frozen v20 source-freeze input drifted: {name}: {observed}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    if (
        payloads["authorization"].get("formal_held_out_outcomes_read") is not False
        or payloads["registration"].get("v20_development_evaluation_outcomes_read")
        is not False
        or payloads["protocol"].get("v20_development_evaluation_outcomes_read")
        is not False
        or payloads["environment"].get("disjointness", {}).get("verified_disjoint")
        is not True
        or payloads["design_audit"].get("scientific_outcomes_available") is not False
        or payloads["design_audit"].get("passed") is not True
        or payloads["design_verification"].get("passed") is not True
        or payloads["v19_audit"].get("artifact_integrity", {}).get("passed")
        is not True
        or payloads["v19_audit"].get("parent_training_stability", {}).get("passed")
        is not True
        or payloads["v19_closeout"].get("passed") is not True
        or payloads["v19_closeout"].get("eligible_for_full_development_control_matrix")
        is not True
        or payloads["v19_verification"].get("passed") is not True
        or payloads["data_environment_audit"].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or payloads["data_environment_verification"].get("passed") is not True
    ):
        raise ValueError("v20 source-freeze scientific scope drifted")
    result_root = REPRO / "h2oplus_v20_full_control_results"
    if result_root.exists() and list(result_root.rglob("sumo_seed*.json")):
        raise ValueError("refusing to freeze v20 after a v20 scientific outcome")
    matrix_sources = [
        descriptor(ROOT / relative, relative=relative) for relative in MATRIX_SOURCES
    ]
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v20 source freeze: {output}")
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v20-development-evaluation-outcome",
        "study_stage": "development",
        "matrix_type": "full-development-control-matrix",
        "formal_held_out_outcomes_read": False,
        "v20_development_evaluation_outcomes_read": False,
        "prior_v14_through_v19_development_outcomes_used_for_design": True,
        "data_environment_audit_material_factors_disclosed": True,
        "fresh_development_environment_used": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "new_training_runs": 0,
        "physical_behavior_count": 45,
        "expected_new_remote_sumo_episodes": 405,
        "expected_matrix_source_file_count": 57,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "analyzer": descriptor(
            ROOT / "reproducibility/analyze_h2oplus_v20_full_control.py",
            relative="reproducibility/analyze_h2oplus_v20_full_control.py",
        ),
        "dispatch": descriptor(
            ROOT / "reproducibility/run_h2oplus_v20_full_control_evaluation_dispatch.py",
            relative=(
                "reproducibility/"
                "run_h2oplus_v20_full_control_evaluation_dispatch.py"
            ),
        ),
        "authorization": descriptor(paths["authorization"]),
        "rationale": descriptor(paths["rationale"]),
        "registration": descriptor(paths["registration"]),
        "protocol": descriptor(paths["protocol"]),
        "methods": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "design_audit": descriptor(paths["design_audit"]),
        "design_verification": descriptor(paths["design_verification"]),
        "parent_v19_audit": descriptor(paths["v19_audit"]),
        "parent_v19_closeout": descriptor(paths["v19_closeout"]),
        "parent_v19_verification": descriptor(paths["v19_verification"]),
        "parent_v19_methods": descriptor(paths["v19_methods"]),
        "parent_v14_methods": descriptor(paths["v14_methods"]),
        "data_environment_audit": descriptor(paths["data_environment_audit"]),
        "data_environment_verification": descriptor(
            paths["data_environment_verification"]
        ),
        "dataset": descriptor(paths["dataset"]),
        "sumo_runtime_lock": descriptor(paths["runtime_lock"]),
        "analysis_operationalization": {
            "predesignated_candidate": (
                "h2oplus-v19-completion-safety-quarter-load-sixty"
            ),
            "direct_behavior_parent": "h2oplus-v19-completion-safety-baseline",
            "raw_actor_parent": "h2oplus-v14-terminal-quarter",
            "independent_lineage_control_count": 6,
            "endpoint_only": True,
            "aulc_computed_or_claimed": False,
            "passing_matrix_creates_formal_confirmation_eligibility": True,
            "formal_confirmation_submission_authorized": False,
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
                "expected_runtime_sources": 57,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
