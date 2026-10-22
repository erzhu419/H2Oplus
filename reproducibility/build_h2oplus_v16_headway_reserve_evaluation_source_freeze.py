#!/usr/bin/env python3
"""Freeze v16 evaluation and analysis sources before any v16 outcome exists."""

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
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
    "reproducibility/run_evaluation_matrix_v16_headway_reserve.py",
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v16_headway_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v16_headway_reserve.py",
)
EXPECTED = {
    "authorization": "77f2234b90ba510fea089800494c637c12d8a3f24b2f2fdd236238ae04fb19a1",
    "rationale": "48e0445e1d82b4f08fc2068e57d5ec1931c26c5d31db9761f51d5082eadadf25",
    "registration": "6858bea266a4f46ec919804e643a5b8c6de4c189dd0517ee5bf123f08b64950a",
    "protocol": "978faf1b8ddf51128b5fac570552e8388c3d882e39244c4b6f0ec7615e79a43b",
    "methods": "35047fd2ce616acbccd4592ae333a7ba84a1e4231c1a2505c7768d96b91de06a",
    "environment": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v15_source_freeze": "af025dbd817a4d7c4661227fefae7653ecc5ca9bf666e94a1c0a8ca7e05f642d",
    "v15_analysis": "d122aa79f6ee2cd5010e241607347921ed9bdc80bbed6ba41c3105e601ac9d9c",
    "v15_audit": "a060b49e3155092542681c7d6afef66bcb2b31ab04ffe523273754ae2dcfa7e8",
    "v15_closeout": "2e4598c433eb6ffaf18fb9919b608b00d5cdd17f5a53e466ca587d6c8213feab",
    "v15_verification": "f5edd5c2ef651172068c9cd595e6cb7710a3ffdd66e545671f5b4a014885e0b6",
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
        default=REPRO / "h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO / "h2oplus_v16_headway_reserve_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v16_headway_reserve_rationale_v1.json",
        "registration": REPRO / "h2oplus_v16_headway_reserve_registration_v1.json",
        "protocol": REPRO / "h2oplus_v16_headway_reserve_protocol_v1.json",
        "methods": REPRO / "h2oplus_v16_headway_reserve_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v12_staged_release_environment_v2.json",
        "v15_source_freeze": REPRO / "h2oplus_v15_hold_calibration_evaluation_source_freeze_v1.json",
        "v15_analysis": REPRO / "h2oplus_v15_hold_calibration_results/hold_calibration_analysis_v1.json",
        "v15_audit": REPRO / "h2oplus_v15_hold_calibration_results/evaluation_audit_v1.json",
        "v15_closeout": REPRO / "h2oplus_v15_hold_calibration_results/evaluation_closeout_v1.json",
        "v15_verification": REPRO
        / "h2oplus_v15_hold_calibration_results/evaluation_closeout_verification_v1.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v16 source-freeze input drifted: {name}")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v16 source freeze: {output}")
    matrix_sources = [
        descriptor(ROOT / relative) | {"path": relative} for relative in MATRIX_SOURCES
    ]
    operational_sources = [
        descriptor(ROOT / relative) | {"path": relative}
        for relative in OPERATIONAL_SOURCES
    ]
    payload = {
        "schema_version": "transit-h2oplus-v16-headway-reserve-evaluation-source-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v16-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v16_evaluation_outcomes_read": False,
        "prior_v14_and_v15_development_outcomes_used_for_design": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 51,
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
        "v15_source_freeze": descriptor(paths["v15_source_freeze"]),
        "v15_development_analysis": descriptor(paths["v15_analysis"]),
        "v15_evaluation_audit": descriptor(paths["v15_audit"]),
        "v15_evaluation_closeout": descriptor(paths["v15_closeout"]),
        "v15_evaluation_verification": descriptor(paths["v15_verification"]),
        "analysis_operationalization": {
            "predesignated_candidate": "h2oplus-v16-headway-reserve-three-quarter",
            "diagnostics_cannot_replace_candidate": True,
            "aulc_budgets": [0, 5000, 10000, 25000],
            "frozen_v14_prefix_budgets": [0, 5000, 10000],
            "new_v16_budget": 25000,
            "all_v15_protocol_gates_retained": True,
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
