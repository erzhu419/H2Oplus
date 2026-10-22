#!/usr/bin/env python3
"""Freeze v15 evaluation and analysis sources before any v15 outcome exists."""

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
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v15_hold_calibration_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v15_hold_calibration.py",
)
EXPECTED = {
    "authorization": "5b3eaefa9a163ebfef2596691a54e61fbe2685cda943f76fa1e7790c7e7b971c",
    "rationale": "cd9c6fb56b2a34771675624799d9a28d2ad988349568f8b96a31257a94278de4",
    "registration": "b37d595056923f9bd746226b6c4863c2d7374750e8d45291de564bc9247428b9",
    "protocol": "6cd856504438ad0e9614b49156100cef12f24201b41562bf223e898f15ef0a8c",
    "methods": "7712a35cbb1adbc55acf50bd8843bb3f4f7dc17090d0e65deeeb10ffe5177e5a",
    "environment": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v14_source_freeze": "82e1404e8471fe01ad4fa4311ce442b8348dcc0444855424cf2b89db281b49bb",
    "v14_analysis": "937a5300988311f80a84dc1220dfee3f0f96f2b6f83691d4e309c38d6bbe6856",
    "v14_closeout": "95fef5b3a5206ce14a3da164d9f2948b0dc3b5d61ee1d1b34c52c431d1cd1d32",
    "v14_verification": "0e20ccc1e7855e4feb5860e5cb334cd60237f28032a948b7325037ad0aa61dc0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO / "h2oplus_v15_hold_calibration_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v15_hold_calibration_rationale_v1.json",
        "registration": REPRO / "h2oplus_v15_hold_calibration_registration_v1.json",
        "protocol": REPRO / "h2oplus_v15_hold_calibration_protocol_v1.json",
        "methods": REPRO / "h2oplus_v15_hold_calibration_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v12_staged_release_environment_v2.json",
        "v14_source_freeze": REPRO / "h2oplus_v14_terminal_taper_evaluation_source_freeze_v1.json",
        "v14_analysis": REPRO / "h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
        "v14_closeout": REPRO / "h2oplus_v14_terminal_taper_results/evaluation_closeout_v1.json",
        "v14_verification": REPRO / "h2oplus_v14_terminal_taper_results/evaluation_closeout_verification_v1.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v15 source-freeze input drifted: {name}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v15 source freeze: {output}")
    matrix_sources = [descriptor(ROOT / relative) | {"path": relative} for relative in MATRIX_SOURCES]
    operational_sources = [
        descriptor(ROOT / relative) | {"path": relative} for relative in OPERATIONAL_SOURCES
    ]
    payload = {
        "schema_version": "transit-h2oplus-v15-hold-calibration-evaluation-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v15-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v15_evaluation_outcomes_read": False,
        "prior_v14_development_outcomes_used_for_design": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 49,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "authorization": descriptor(paths["authorization"]),
        "rationale": descriptor(paths["rationale"]),
        "registration": descriptor(paths["registration"]),
        "protocol": descriptor(paths["protocol"]),
        "methods_manifest": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "v14_source_freeze": descriptor(paths["v14_source_freeze"]),
        "v14_development_analysis": descriptor(paths["v14_analysis"]),
        "v14_evaluation_closeout": descriptor(paths["v14_closeout"]),
        "v14_evaluation_verification": descriptor(paths["v14_verification"]),
        "analysis_operationalization": {
            "predesignated_candidate": "h2oplus-v15-hold-quarter",
            "diagnostics_cannot_replace_candidate": True,
            "aulc_budgets": [0, 5000, 10000, 25000],
            "frozen_v14_prefix_budgets": [0, 5000, 10000],
            "new_v15_budget": 25000,
            "all_v14_protocol_gates_retained": True,
            "formal_confirmation_eligibility_does_not_authorize_submission": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "sources": 12}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
