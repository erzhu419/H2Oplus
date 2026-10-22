#!/usr/bin/env python3
"""Freeze v19 scientific and operational sources before any v19 outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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
        )
    ]
)
OPERATIONAL_SOURCES = (
    "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v19_completion_safety.py",
    "reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py",
)
EXPECTED = {
    "authorization": "d81add0e13aed729412950f79a947aa8dee8e18f1df4eb3fcd5e622992e419c8",
    "rationale": "ee16744487f43645babf9322ea42ffa016ddd3b9fe12a866d5adf11198af9c0b",
    "registration": "536741a185e2d9b10c99ed821292a51c64bf64e8cb35771075ab787565246fd4",
    "protocol": "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184",
    "methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "environment": "5cfb1318418b64e619d4be727c41a0ce0eccab8c096020482586d8286cee3a05",
    "v18_audit": "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    "v18_closeout": "e2a0285e9d95bc999cbe371ac261afb1f4526ff7a6722ba2d9d114f67c4c401c",
    "v18_verification": "6a08fa047a4616589fba04cd39eec41e9450ef1ce1fd100933039cb7354fa35a",
    "design_audit_v1": "c9edb3bb264b3d3984693599d3641ef769eb31a9b207a49eecd0264af7fe33f2",
    "design_audit_v2": "2238ceb015a456b80c2be5d4d06ebd96d2293fdbc88cadcfb08f087d2dafe1e8",
    "design_verification_v2": "3f8c0e03514b2fb6e2ed46249cfe8260cfdc3c9a2872316f6575de1504543a78",
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


def descriptor(path: Path, relative: Optional[str] = None) -> dict:
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
        default=REPRO / "h2oplus_v19_completion_safety_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO / "h2oplus_v19_completion_safety_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v19_completion_safety_rationale_v1.json",
        "registration": REPRO / "h2oplus_v19_completion_safety_registration_v1.json",
        "protocol": REPRO / "h2oplus_v19_completion_safety_protocol_v1.json",
        "methods": REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v19_completion_safety_environment_v1.json",
        "v18_audit": REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
        "v18_closeout": REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
        "v18_verification": REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_verification_v3.json",
        "design_audit_v1": REPRO / "h2oplus_v19_completion_safety_design_audit_v1.json",
        "design_audit_v2": REPRO / "h2oplus_v19_completion_safety_design_audit_v2.json",
        "design_verification_v2": REPRO / "h2oplus_v19_completion_safety_design_verification_v2.json",
        "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v19 source-freeze input drifted: {name}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    if (
        payloads["authorization"].get("formal_held_out_outcomes_read") is not False
        or payloads["registration"].get("v19_development_evaluation_outcomes_read") is not False
        or payloads["protocol"].get("v19_development_evaluation_outcomes_read") is not False
        or payloads["environment"].get("disjointness", {}).get("verified_disjoint") is not True
        or any(payloads[name].get("passed") is not True for name in ("v18_audit", "v18_closeout", "v18_verification"))
        or payloads["design_audit_v2"].get("claim_boundary", {}).get("development_result_available") is not False
        or payloads["design_verification_v2"].get("passed") is not True
        or payloads["data_environment_audit"].get("conclusions", {}).get("material_data_or_environment_factors_present") is not True
        or payloads["data_environment_verification"].get("passed") is not True
    ):
        raise ValueError("v19 source-freeze scientific scope drifted")
    matrix_sources = [
        descriptor(ROOT / relative, relative=relative) for relative in MATRIX_SOURCES
    ]
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v19 source freeze: {output}")
    payload = {
        "schema_version": "transit-h2oplus-v19-completion-safety-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v19-development-evaluation-outcome",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "v19_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "data_environment_audit_material_factors_disclosed": True,
        "fresh_development_environment_used": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "new_training_runs": 0,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 55,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "analyzer": descriptor(
            ROOT / "reproducibility/analyze_h2oplus_v19_completion_safety.py",
            relative="reproducibility/analyze_h2oplus_v19_completion_safety.py",
        ),
        "dispatch": descriptor(
            ROOT / "reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py",
            relative="reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py",
        ),
        "authorization": descriptor(paths["authorization"]),
        "rationale": descriptor(paths["rationale"]),
        "registration": descriptor(paths["registration"]),
        "protocol": descriptor(paths["protocol"]),
        "methods": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "parent_v18_audit": descriptor(paths["v18_audit"]),
        "parent_v18_closeout": descriptor(paths["v18_closeout"]),
        "parent_v18_verification": descriptor(paths["v18_verification"]),
        "design_audit_v1": descriptor(paths["design_audit_v1"]),
        "design_audit_v2": descriptor(paths["design_audit_v2"]),
        "design_verification_v2": descriptor(paths["design_verification_v2"]),
        "data_environment_audit": descriptor(paths["data_environment_audit"]),
        "data_environment_verification": descriptor(paths["data_environment_verification"]),
        "dataset": descriptor(paths["dataset"]),
        "sumo_runtime_lock": descriptor(paths["runtime_lock"]),
        "analysis_operationalization": {
            "predesignated_candidate": "h2oplus-v19-completion-safety-quarter-load-sixty",
            "paired_baseline": "h2oplus-v19-completion-safety-baseline",
            "diagnostic": "h2oplus-v19-completion-safety-quarter-load-fifty",
            "diagnostic_cannot_replace_candidate": True,
            "passing_screen_only_permits_full_development_control_matrix": True,
            "formal_confirmation_eligibility_created": False,
            "formal_confirmation_submission_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256(output),
                "matrix_sources": len(matrix_sources),
                "operational_sources": len(operational_sources),
                "expected_runtime_sources": 55,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
