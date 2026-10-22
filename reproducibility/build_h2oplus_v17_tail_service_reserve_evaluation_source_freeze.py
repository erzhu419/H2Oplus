#!/usr/bin/env python3
"""Freeze v17 evaluation and analysis sources before any v17 outcome exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
MATRIX_SOURCES = (
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "SimpleSAC/evaluate_transit_v13_proximal_release.py",
    "SimpleSAC/evaluate_transit_v14_terminal_taper.py",
    "SimpleSAC/evaluate_transit_v15_hold_calibration.py",
    "SimpleSAC/evaluate_transit_v16_headway_reserve.py",
    "SimpleSAC/evaluate_transit_v17_tail_service_reserve.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
    "reproducibility/run_evaluation_matrix_v16_headway_reserve.py",
    "reproducibility/run_evaluation_matrix_v17_tail_service_reserve.py",
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v17_tail_service_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v17_tail_service_reserve.py",
)
EXPECTED = {
    "authorization": "6871d458b23bbc796aa8f0a981eb21f865f6a5a0d61c3c427173573e44261e4a",
    "offline_design": "5bc0e0366fa19ef9597db20de702d554bb1dc05ddc6ec7ddb1fdedc72818b24e",
    "rationale": "0b41d827e79bdbea83b689caf0afd144d13f83780e74ec44e716e65ca7099e5a",
    "registration": "d5465e677e29971eb1c6e88a926387bacd172de737f669f6fd84426bb7c68364",
    "protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "environment": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v16_source_freeze": "606a15bf7db8b9aa1c4dbefa7b9bcc12bc94ca60e605de5a3e185eefec679577",
    "v16_analysis": "4a88f48946c2ec478ca47274df39e0ccdf3535c1c21ad41141142d529617f75e",
    "v16_audit": "ed5e58e52f1ea50d7195a10b313bd9c36e88b618fa802de2693ffdfe133112c9",
    "v16_closeout": "ef7d456e6c71bb2fbcbd176642c01acf123f8e709cefffd2b8d4ae26a850ddf5",
    "v16_verification": "54b25d3c9307e9f926319b7f9ae543e6747d4639ce9b8af762e123bade9f4c85",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO / "h2oplus_v17_tail_service_reserve_authorization_v1.json",
        "offline_design": REPRO
        / "h2oplus_v17_tail_service_reserve_offline_design_evidence_v1.json",
        "rationale": REPRO / "h2oplus_v17_tail_service_reserve_rationale_v1.json",
        "registration": REPRO / "h2oplus_v17_tail_service_reserve_registration_v1.json",
        "protocol": REPRO / "h2oplus_v17_tail_service_reserve_protocol_v1.json",
        "methods": REPRO / "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v12_staged_release_environment_v2.json",
        "v16_source_freeze": REPRO / "h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
        "v16_analysis": REPRO / "h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json",
        "v16_audit": REPRO / "h2oplus_v16_headway_reserve_results/evaluation_audit_v1.json",
        "v16_closeout": REPRO / "h2oplus_v16_headway_reserve_results/evaluation_closeout_v1.json",
        "v16_verification": REPRO
        / "h2oplus_v16_headway_reserve_results/evaluation_closeout_verification_v1.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v17 source-freeze input drifted: {name}")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v17 source freeze: {output}")
    matrix_sources = [
        descriptor(ROOT / relative) | {"path": relative} for relative in MATRIX_SOURCES
    ]
    operational_sources = [
        descriptor(ROOT / relative) | {"path": relative}
        for relative in OPERATIONAL_SOURCES
    ]
    payload = {
        "schema_version": "transit-h2oplus-v17-tail-service-reserve-evaluation-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v17-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v17_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_development_outcomes_used_for_design": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 53,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "authorization": descriptor(paths["authorization"]),
        "offline_design_evidence": descriptor(paths["offline_design"]),
        "rationale": descriptor(paths["rationale"]),
        "registration": descriptor(paths["registration"]),
        "protocol": descriptor(paths["protocol"]),
        "methods_manifest": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "v16_source_freeze": descriptor(paths["v16_source_freeze"]),
        "v16_development_analysis": descriptor(paths["v16_analysis"]),
        "v16_evaluation_audit": descriptor(paths["v16_audit"]),
        "v16_evaluation_closeout": descriptor(paths["v16_closeout"]),
        "v16_evaluation_verification": descriptor(paths["v16_verification"]),
        "analysis_operationalization": {
            "predesignated_candidate": "h2oplus-v17-tail-service-reserve-thirty",
            "diagnostics_cannot_replace_candidate": True,
            "aulc_budgets": [0, 5000, 10000, 25000],
            "frozen_v14_prefix_budgets": [0, 5000, 10000],
            "new_v17_budget": 25000,
            "all_v16_protocol_gates_retained": True,
            "formal_confirmation_eligibility_does_not_authorize_submission": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256(output),
                "scientific_sources": len(matrix_sources),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
