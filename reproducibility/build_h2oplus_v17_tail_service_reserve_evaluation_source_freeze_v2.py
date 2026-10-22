#!/usr/bin/env python3
"""Freeze the fresh Python 3.8-compatible v17-v2 evaluation lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT_ROOT = (
    REPRO
    / "h2oplus_v17_tail_service_reserve_results/evaluation_incidents"
    / "v1_python38_removeprefix"
)
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
    "reproducibility/run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2.py",
    "reproducibility/analyze_h2oplus_v17_tail_service_reserve.py",
)
EXPECTED = {
    "authorization": "4c61471e71c3190c3b450b2805eba0a48712da8c25a9840806b3bd849029ebd4",
    "correction": "1a521809e08618001b19e5d2d7d1077472d819aa648fcaa46e87423e833268ed",
    "registration": "41d8399dcf07aedea421fed516407fbeff7fa4dda296652b8c70aa24582c90cf",
    "offline_design": "5bc0e0366fa19ef9597db20de702d554bb1dc05ddc6ec7ddb1fdedc72818b24e",
    "rationale": "0b41d827e79bdbea83b689caf0afd144d13f83780e74ec44e716e65ca7099e5a",
    "protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "environment": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "v1_source_freeze": "30e4b23219dc01c96b7eeaf89e4a8564716a887b9e525e8bcf65164ca8026f95",
    "v1_submission": "37795ac948a19bb49667d4a54b7304be6c10952a39a79430da3b18644aeda804",
    "v1_deployment_manifest": "a4bb3706a8f401d347c0e403f7b37ce47f142401f3c4aac3a4507d6e1a64ef0d",
    "incident_closeout": "0d189243e83ee014c414fd154f26ebc52107c934dee51131a5aecaf9cd3e19ed",
    "incident_manifest": "7974f184003663aaea39c0090704be0a701bcb2465a1b35b0a2265a9b6ee4d73",
    "incident_verification": "922b514bbab7228da049eed036cb01e1f85b93c1e3de24bc79ad2a97b23150e0",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
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


def descriptor(path: Path, relative: str | None = None) -> dict:
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
        default=REPRO
        / "h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO
        / "h2oplus_v17_tail_service_reserve_successor_authorization_v2.json",
        "correction": REPRO
        / "h2oplus_v17_tail_service_reserve_python38_correction_v2.json",
        "registration": REPRO
        / "h2oplus_v17_tail_service_reserve_successor_registration_v2.json",
        "offline_design": REPRO
        / "h2oplus_v17_tail_service_reserve_offline_design_evidence_v1.json",
        "rationale": REPRO / "h2oplus_v17_tail_service_reserve_rationale_v1.json",
        "protocol": REPRO / "h2oplus_v17_tail_service_reserve_protocol_v1.json",
        "methods": REPRO
        / "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v12_staged_release_environment_v2.json",
        "v1_source_freeze": REPRO
        / "h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json",
        "v1_submission": REPRO
        / "h2oplus_v17_tail_service_reserve_evaluation_submission_v1.json",
        "v1_deployment_manifest": REPRO
        / "deployments/h2oplus_v17_tail_service_reserve_evaluation_v1/workspace/deployment_manifest.json",
        "incident_closeout": INCIDENT_ROOT / "incident_closeout_v1.json",
        "incident_manifest": INCIDENT_ROOT / "incident_file_manifest_v1.json",
        "incident_verification": INCIDENT_ROOT / "incident_verification_v1.json",
        "data_environment_audit": REPRO
        / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO
        / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "v16_source_freeze": REPRO
        / "h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
        "v16_analysis": REPRO
        / "h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json",
        "v16_audit": REPRO
        / "h2oplus_v16_headway_reserve_results/evaluation_audit_v1.json",
        "v16_closeout": REPRO
        / "h2oplus_v16_headway_reserve_results/evaluation_closeout_v1.json",
        "v16_verification": REPRO
        / "h2oplus_v16_headway_reserve_results/evaluation_closeout_verification_v1.json",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v17-v2 source-freeze input drifted: {name}")

    v1_freeze = json.loads(paths["v1_source_freeze"].read_text())
    old_matrix = {
        row["path"]: row["sha256"] for row in v1_freeze.get("matrix_sources", [])
    }
    matrix_sources = [
        descriptor(ROOT / relative, relative=relative) for relative in MATRIX_SOURCES
    ]
    if {row["path"]: row["sha256"] for row in matrix_sources} != old_matrix:
        raise ValueError("v17-v2 scientific matrix sources differ from frozen v17-v1")
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    analyzer = operational_sources[1]
    if analyzer["sha256"] != v1_freeze["operational_sources"][1]["sha256"]:
        raise ValueError("v17-v2 frozen analyzer differs from v17-v1")
    dispatch_text = (ROOT / OPERATIONAL_SOURCES[0]).read_text()
    if (
        ".removeprefix(" in dispatch_text
        or ".removesuffix(" in dispatch_text
        or "def v17_algorithm_variant(algorithm: str) -> str:" not in dispatch_text
        or "return algorithm[len(prefix):]" not in dispatch_text
    ):
        raise ValueError("v17-v2 dispatch is not Python 3.8 string-API compatible")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v17-v2 source freeze: {output}")
    payload = {
        "schema_version": (
            "transit-h2oplus-v17-tail-service-reserve-evaluation-source-freeze-v2"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v17-v2-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v17_v1_scientific_numeric_outcomes_read": False,
        "v17_v1_incident_results_reused": False,
        "v17_v2_evaluation_outcomes_read": False,
        "prior_v14_v15_v16_development_outcomes_used_for_design": True,
        "data_environment_audit_used_to_change_v17_v2_design": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 53,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "scientific_matrix_sources_unchanged_from_v17_v1": True,
        "frozen_analyzer_unchanged_from_v17_v1": True,
        "python38_compatibility_rewrite_only": True,
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "authorization": descriptor(paths["authorization"]),
        "correction": descriptor(paths["correction"]),
        "registration": descriptor(paths["registration"]),
        "offline_design_evidence": descriptor(paths["offline_design"]),
        "rationale": descriptor(paths["rationale"]),
        "protocol": descriptor(paths["protocol"]),
        "methods_manifest": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "v17_v1_source_freeze": descriptor(paths["v1_source_freeze"]),
        "v17_v1_submission": descriptor(paths["v1_submission"]),
        "v17_v1_deployment_manifest": descriptor(paths["v1_deployment_manifest"]),
        "v17_v1_incident_closeout": descriptor(paths["incident_closeout"]),
        "v17_v1_incident_file_manifest": descriptor(paths["incident_manifest"]),
        "v17_v1_incident_verification": descriptor(paths["incident_verification"]),
        "data_environment_audit": descriptor(paths["data_environment_audit"]),
        "data_environment_verification": descriptor(
            paths["data_environment_verification"]
        ),
        "v16_source_freeze": descriptor(paths["v16_source_freeze"]),
        "v16_development_analysis": descriptor(paths["v16_analysis"]),
        "v16_evaluation_audit": descriptor(paths["v16_audit"]),
        "v16_evaluation_closeout": descriptor(paths["v16_closeout"]),
        "v16_evaluation_verification": descriptor(paths["v16_verification"]),
        "analysis_operationalization": {
            "predesignated_candidate": (
                "h2oplus-v17-tail-service-reserve-thirty"
            ),
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
                "operational_sources": len(operational_sources),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
