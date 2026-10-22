#!/usr/bin/env python3
"""Freeze v18 scientific and operational sources before any v18 outcome."""

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
            (18, "empty_tail_reserve"),
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
            (18, "empty_tail_reserve"),
        )
    ]
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
)
EXPECTED = {
    "authorization": "8354467e41971cc39dc99382a2327a75c358f8baa06df22e215854a5c8036f92",
    "rationale": "d19a3fdf2b5daa6407f1cdfe905df89bad43dfa7fb5941109026ea43d6c432b0",
    "registration": "45e160f7104a8a494d3231576c15c09ba9f28f184e3a6da4ac6637a34e3135dc",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "v17_analysis": "84e80b7f08d16f6e356f09c56ff75d1bbdf322578641df4ede9e73f8d4ff4f78",
    "v17_audit": "c23e5a3f0e7b2a9de889cee024f293f3e07f9087689d16f3c649b23fff7d39ec",
    "v17_closeout": "09351ebb883e5aad18d33353be877ed542f1ed7741e1d03b0c9b4368b39e268b",
    "v17_verification": "d781daea3138595989156935367b15a9e8db543aa464bad0a2c58636db95d596",
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
        default=REPRO / "h2oplus_v18_empty_tail_reserve_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO
        / "h2oplus_v18_empty_tail_reserve_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v18_empty_tail_reserve_rationale_v1.json",
        "registration": REPRO
        / "h2oplus_v18_empty_tail_reserve_registration_v1.json",
        "protocol": REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "methods": REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "environment": REPRO
        / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "v17_analysis": REPRO
        / "h2oplus_v17_tail_service_reserve_results/tail_service_reserve_analysis_v2.json",
        "v17_audit": REPRO
        / "h2oplus_v17_tail_service_reserve_results/evaluation_audit_v2.json",
        "v17_closeout": REPRO
        / "h2oplus_v17_tail_service_reserve_results/evaluation_closeout_v2.json",
        "v17_verification": REPRO
        / "h2oplus_v17_tail_service_reserve_results/evaluation_closeout_verification_v2.json",
        "data_environment_audit": REPRO
        / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO
        / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "dataset": ROOT
        / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v18 source-freeze input drifted: {name}")
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if path.suffix == ".json"
    }
    if (
        payloads["authorization"].get("formal_held_out_outcomes_read") is not False
        or payloads["registration"].get("v18_development_evaluation_outcomes_read")
        is not False
        or payloads["protocol"].get("v18_development_evaluation_outcomes_read")
        is not False
        or payloads["environment"].get("disjointness", {}).get(
            "verified_disjoint"
        )
        is not True
        or payloads["v17_audit"].get("passed") is not True
        or payloads["v17_closeout"].get("passed") is not True
        or payloads["v17_verification"].get("passed") is not True
        or payloads["data_environment_audit"].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or payloads["data_environment_verification"].get("passed") is not True
    ):
        raise ValueError("v18 source-freeze scientific scope drifted")
    matrix_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in MATRIX_SOURCES
    ]
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v18 source freeze: {output}")
    payload = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v18-development-evaluation-outcome",
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
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
            ROOT / "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
            relative="reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
        ),
        "dispatch": descriptor(
            ROOT
            / "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
            relative=(
                "reproducibility/"
                "run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py"
            ),
        ),
        "authorization": descriptor(paths["authorization"]),
        "rationale": descriptor(paths["rationale"]),
        "registration": descriptor(paths["registration"]),
        "protocol": descriptor(paths["protocol"]),
        "methods": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "parent_v17_analysis": descriptor(paths["v17_analysis"]),
        "parent_v17_audit": descriptor(paths["v17_audit"]),
        "parent_v17_closeout": descriptor(paths["v17_closeout"]),
        "parent_v17_verification": descriptor(paths["v17_verification"]),
        "data_environment_audit": descriptor(paths["data_environment_audit"]),
        "data_environment_verification": descriptor(
            paths["data_environment_verification"]
        ),
        "dataset": descriptor(paths["dataset"]),
        "sumo_runtime_lock": descriptor(paths["runtime_lock"]),
        "analysis_operationalization": {
            "predesignated_candidate": (
                "h2oplus-v18-empty-tail-reserve-empty-forty"
            ),
            "paired_baseline": "h2oplus-v18-empty-tail-reserve-baseline",
            "diagnostic": "h2oplus-v18-empty-tail-reserve-one-seat-forty",
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
