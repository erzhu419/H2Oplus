#!/usr/bin/env python3
"""Freeze the parser-corrected v18-v2 evaluation lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
INCIDENT = (
    REPRO
    / "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/v1_parser_contract"
)
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
    + [
        "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py",
        "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v2.py",
    ]
)
OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v2.py",
)
EXPECTED = {
    "authorization": "e1f9a4b3237468e984d463ed42315e53906f95ee004f91d07086fa428713c085",
    "correction": "6fee854c13083c00c05a9b60ab9340191effd827ac460d1bf7953cb4839e3822",
    "registration": "0a36553e568664594a7c25139f10a62f481be16b98af767b27add4bd05a0f6d7",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "v1_source_freeze": "1769740ba02cb14a9ca483460c44f4651109fe9e297dec9d42409bc802df49a7",
    "v1_submission": "6ddda52a1619288451a0c5d0dbb8ab4052a526db6e0f17d0d37b814ca4d613b5",
    "v1_deployment_manifest": "b837b687951c8dc7f84b04defea90158b67fe9f5ec34ace96c204b698852d9b7",
    "incident_closeout": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
    "analyzer": "ebb90f93e5b43fce3670f465c0a8fe68459dd953b6bd3580594054c305acf1e6",
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
        default=REPRO / "h2oplus_v18_empty_tail_reserve_source_freeze_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO
        / "h2oplus_v18_empty_tail_reserve_successor_authorization_v2.json",
        "correction": REPRO
        / "h2oplus_v18_empty_tail_reserve_parser_correction_v2.json",
        "registration": REPRO
        / "h2oplus_v18_empty_tail_reserve_successor_registration_v2.json",
        "protocol": REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "methods": REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "environment": REPRO
        / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "v1_source_freeze": REPRO
        / "h2oplus_v18_empty_tail_reserve_source_freeze_v1.json",
        "v1_submission": REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v1.json",
        "v1_deployment_manifest": REPRO
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v1/workspace/deployment_manifest.json",
        "incident_closeout": INCIDENT / "incident_closeout_v1.json",
        "incident_verification": INCIDENT / "incident_verification_v1.json",
        "analyzer": REPRO / "analyze_h2oplus_v18_empty_tail_reserve.py",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v18-v2 source input drifted: {name}")
    v1_freeze = json.loads(paths["v1_source_freeze"].read_text())
    v1_matrix = {
        row["path"]: row["sha256"] for row in v1_freeze.get("matrix_sources", [])
    }
    matrix_sources = [
        descriptor(ROOT / relative, relative=relative) for relative in MATRIX_SOURCES
    ]
    if {
        row["path"]: row["sha256"] for row in matrix_sources[:16]
    } != v1_matrix:
        raise ValueError("v18-v2 inherited matrix sources differ from v18-v1")
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    if operational_sources[1]["sha256"] != EXPECTED["analyzer"]:
        raise ValueError("v18-v2 frozen analyzer differs from v18-v1")
    correction = json.loads(paths["correction"].read_text())
    incident = json.loads(paths["incident_closeout"].read_text())
    if (
        correction.get("scientific_behavior_changed") is not False
        or correction.get("methods_protocol_scenarios_or_gates_changed") is not False
        or correction.get("parser_stack_smoke_required") is not True
        or incident.get("completed_scientific_result_count") != 0
        or incident.get("scientific_numeric_outcomes_read") is not False
        or incident.get("v18_v1_results_reusable") is not False
    ):
        raise ValueError("v18-v2 correction or incident boundary drifted")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v18-v2 source freeze: {output}")
    payload = {
        # Kept at v1 because the unchanged frozen analyzer validates this schema.
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-source-freeze-v1",
        "evidence_revision": "v2-parser-contract-correction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v18-v2-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
        "v18_v1_scientific_numeric_outcomes_read": False,
        "v18_v1_result_count": 0,
        "v18_v1_results_reused": False,
        "v18_v2_development_evaluation_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 57,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "inherited_v18_v1_matrix_source_count": 16,
        "parser_contract_source_count": 2,
        "scientific_behavior_unchanged_from_v18_v1": True,
        "methods_protocol_scenarios_and_gates_unchanged_from_v18_v1": True,
        "frozen_analyzer_unchanged_from_v18_v1": True,
        "matrix_sources": matrix_sources,
        "operational_sources": operational_sources,
        "analyzer": descriptor(paths["analyzer"]),
        "authorization": descriptor(paths["authorization"]),
        "correction": descriptor(paths["correction"]),
        "registration": descriptor(paths["registration"]),
        "protocol": descriptor(paths["protocol"]),
        "methods": descriptor(paths["methods"]),
        "methods_manifest": descriptor(paths["methods"]),
        "environment_design": descriptor(paths["environment"]),
        "v18_v1_source_freeze": descriptor(paths["v1_source_freeze"]),
        "v18_v1_submission": descriptor(paths["v1_submission"]),
        "v18_v1_deployment_manifest": descriptor(paths["v1_deployment_manifest"]),
        "v18_v1_incident_closeout": descriptor(paths["incident_closeout"]),
        "v18_v1_incident_verification": descriptor(paths["incident_verification"]),
        "analysis_operationalization": v1_freeze["analysis_operationalization"],
        "claim_boundary": (
            "A passing paired development screen permits only a fresh full "
            "development control matrix. It is not formal confirmation eligibility."
        ),
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
                "expected_runtime_sources": 57,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
