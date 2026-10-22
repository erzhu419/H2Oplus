#!/usr/bin/env python3
"""Freeze the final-boundary-corrected v18-v3 evaluation lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
NEW_MATRIX_SOURCES = (
    "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py",
    "reproducibility/run_evaluation_matrix_v18_empty_tail_reserve_v3.py",
)
NEW_OPERATIONAL_SOURCES = (
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v3.py",
)
EXPECTED = {
    "authorization": "9c5872050e0b53d61e22ca3d0ad01bb4da626a88cde51f63d5c71672c536cb74",
    "correction": "f636d7d6e882eca5404c194f47c17789d1b27e985bada74006423735e35076af",
    "registration": "d589cf5ac1114522b5a162d0ffc8c2c1fcbdbe8829fca2ecdb88481ff7e02fa7",
    "protocol": "aced1b8bdaf93bd16a1f4fdd326ebbc610eabb22cf21b52950700339e90d09c0",
    "methods": "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    "environment": "fa8a481bbc839e7cb218be191644521264d68664e37f82cae7d7b35617cf791c",
    "v2_source_freeze": "7be31254cf1433642fd6078781dc483101ca3c2baf9c53be9dd0605ff69819ee",
    "v2_submission": "21b8ddd509fb2b61a7d7127398758c15d628218e18f5ac0cd9708a6ae809ea3c",
    "v2_deployment_manifest": "7f6595be6c1f239e1c5c16f85aad7667ed9e87841c02c8eae9b15d0439004600",
    "v1_incident_closeout": "a47fadd4c8b515d9fe5b1ce0a9781862f4cf974a289ef5100309b2fcaddf2600",
    "v1_incident_verification": "80dc525acc2466a45bc9d26aa9d1ea8f8cbe5738f0529eddb2646d088834ea0e",
    "v2_incident_closeout": "c2cf9e59248887be24df96e53dcd4cb09a7f5c9fb5e5a8e8fd7746c6cf6d3568",
    "v2_incident_verification": "ee32840d6de1e10a7b3c9a622c1ff50ab33d63234b3cb3616d37af18a256f9b8",
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
        default=REPRO / "h2oplus_v18_empty_tail_reserve_source_freeze_v3.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "authorization": REPRO
        / "h2oplus_v18_empty_tail_reserve_successor_authorization_v3.json",
        "correction": REPRO
        / "h2oplus_v18_empty_tail_reserve_namespace_binding_correction_v3.json",
        "registration": REPRO
        / "h2oplus_v18_empty_tail_reserve_successor_registration_v3.json",
        "protocol": REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "methods": REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "environment": REPRO
        / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "v2_source_freeze": REPRO
        / "h2oplus_v18_empty_tail_reserve_source_freeze_v2.json",
        "v2_submission": REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v2.json",
        "v2_deployment_manifest": REPRO
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v2/workspace/deployment_manifest.json",
        "v1_incident_closeout": REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v1_parser_contract/incident_closeout_v1.json"
        ),
        "v1_incident_verification": REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v1_parser_contract/incident_verification_v1.json"
        ),
        "v2_incident_closeout": REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v2_namespace_binding/incident_closeout_v2.json"
        ),
        "v2_incident_verification": REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v2_namespace_binding/incident_verification_v2.json"
        ),
        "analyzer": REPRO / "analyze_h2oplus_v18_empty_tail_reserve.py",
    }
    for name, path in paths.items():
        if sha256(path.resolve(strict=True)) != EXPECTED[name]:
            raise ValueError(f"frozen v18-v3 source input drifted: {name}")
    v2_freeze = json.loads(paths["v2_source_freeze"].read_text())
    matrix_sources = []
    for row in v2_freeze.get("matrix_sources", []):
        observed = descriptor(ROOT / row["path"], relative=row["path"])
        if observed != row:
            raise ValueError(f"v18-v2 matrix source drifted: {row['path']}")
        matrix_sources.append(observed)
    matrix_sources.extend(
        descriptor(ROOT / relative, relative=relative)
        for relative in NEW_MATRIX_SOURCES
    )
    operational_sources = []
    for row in v2_freeze.get("operational_sources", []):
        observed = descriptor(ROOT / row["path"], relative=row["path"])
        if observed != row:
            raise ValueError(f"v18-v2 operational source drifted: {row['path']}")
        operational_sources.append(observed)
    operational_sources.extend(
        descriptor(ROOT / relative, relative=relative)
        for relative in NEW_OPERATIONAL_SOURCES
    )
    if (
        len(matrix_sources) != 20
        or len(operational_sources) != 4
        or operational_sources[1]["sha256"] != EXPECTED["analyzer"]
    ):
        raise ValueError("v18-v3 source membership drifted")
    correction = json.loads(paths["correction"].read_text())
    incident_v2 = json.loads(paths["v2_incident_closeout"].read_text())
    if (
        correction.get("scientific_behavior_changed") is not False
        or correction.get("methods_protocol_scenarios_or_gates_changed") is not False
        or correction.get("final_namespace_binding_smoke_required") is not True
        or incident_v2.get("completed_scientific_result_count") != 0
        or incident_v2.get("scientific_numeric_outcomes_read") is not False
        or incident_v2.get("v18_v2_results_reusable") is not False
    ):
        raise ValueError("v18-v3 correction or incident boundary drifted")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v18-v3 source freeze: {output}")
    payload = {
        # Kept at v1 because the unchanged frozen analyzer validates this schema.
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-source-freeze-v1",
        "evidence_revision": "v3-final-namespace-binding-correction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-any-v18-v3-development-evaluation-outcome",
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "v18_development_evaluation_outcomes_read": False,
        "v18_v1_or_v2_scientific_numeric_outcomes_read": False,
        "v18_v1_result_count": 0,
        "v18_v2_result_count": 0,
        "v18_v1_results_reused": False,
        "v18_v2_results_reused": False,
        "v18_v3_development_evaluation_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "physical_behavior_count": 15,
        "expected_new_remote_sumo_episodes": 135,
        "expected_matrix_source_file_count": 59,
        "matrix_source_count": len(matrix_sources),
        "operational_source_count": len(operational_sources),
        "inherited_v18_v2_matrix_source_count": 18,
        "final_binding_source_count": 2,
        "scientific_behavior_unchanged_from_v18_v1_and_v2": True,
        "methods_protocol_scenarios_and_gates_unchanged_from_v18_v1_and_v2": True,
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
        "v18_v2_source_freeze": descriptor(paths["v2_source_freeze"]),
        "v18_v2_submission": descriptor(paths["v2_submission"]),
        "v18_v2_deployment_manifest": descriptor(paths["v2_deployment_manifest"]),
        "v18_v1_incident_closeout": descriptor(paths["v1_incident_closeout"]),
        "v18_v1_incident_verification": descriptor(
            paths["v1_incident_verification"]
        ),
        "v18_v2_incident_closeout": descriptor(paths["v2_incident_closeout"]),
        "v18_v2_incident_verification": descriptor(
            paths["v2_incident_verification"]
        ),
        "analysis_operationalization": v2_freeze["analysis_operationalization"],
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
                "expected_runtime_sources": 59,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
