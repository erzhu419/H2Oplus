#!/usr/bin/env python3
"""Freeze the v19 protocol-environment binding correction."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.build_h2oplus_v19_completion_safety_source_freeze import (
    MATRIX_SOURCES,
    OPERATIONAL_SOURCES,
    descriptor,
    sha256,
)


EXPECTED_PARENT_FREEZE = (
    "9d8619304dd8f433d568012a25777a8f9a774eea0a4ba56cf7c7fed70007be63"
)
EXPECTED_CORRECTION = (
    "2aec76694036ad038d2d15b4197c1444db8587171898ed0abdb7bdc410b4c282"
)
EXPECTED_V1_DEPLOYMENT_MANIFEST = (
    "24b3e9aeba8ace7b10356ff9fe6d1d935282f3d529cef9b527bf1469513b9787"
)
EXPECTED_V1_DEPLOYMENT_AUDIT = (
    "4b699882df1e8829bf7d937fe3093a410528a068c38cbcbe41f0518d71f85ed7"
)
UPDATED_DISPATCH = (
    "reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v19_completion_safety_source_freeze_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent_path = REPRO / "h2oplus_v19_completion_safety_source_freeze_v1.json"
    correction_path = (
        REPRO / "h2oplus_v19_completion_safety_environment_binding_correction_v2.json"
    )
    v1_manifest_path = (
        REPRO
        / "deployments/h2oplus_v19_completion_safety_evaluation_v1/workspace/deployment_manifest.json"
    )
    v1_audit_path = REPRO / "h2oplus_v19_completion_safety_evaluation_deployment_audit_v1.json"
    expected = {
        parent_path: EXPECTED_PARENT_FREEZE,
        correction_path: EXPECTED_CORRECTION,
        v1_manifest_path: EXPECTED_V1_DEPLOYMENT_MANIFEST,
        v1_audit_path: EXPECTED_V1_DEPLOYMENT_AUDIT,
    }
    for path, digest in expected.items():
        if sha256(path.resolve(strict=True)) != digest:
            raise ValueError(f"v19-v2 correction input drifted: {path}")
    parent = json.loads(parent_path.read_text())
    correction = json.loads(correction_path.read_text())
    v1_manifest = json.loads(v1_manifest_path.read_text())
    v1_audit = json.loads(v1_audit_path.read_text())
    if (
        parent.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-source-freeze-v1"
        or parent.get("v19_development_evaluation_outcomes_read") is not False
        or correction.get("status") != "frozen-predeployment-zero-result-correction"
        or correction.get("execution_boundary", {}).get(
            "completed_scientific_result_count"
        )
        != 0
        or correction.get("execution_boundary", {}).get(
            "scientific_numeric_outcomes_read"
        )
        is not False
        or v1_manifest.get("formal_held_out_outcomes_included") is not False
        or v1_audit.get("passed") is not True
        or v1_audit.get("local_or_remote_sumo_simulation_executed") is not False
    ):
        raise ValueError("v19-v2 correction boundary drifted")

    old_sources = {
        row["path"]: row["sha256"]
        for row in parent["matrix_sources"] + parent["operational_sources"]
    }
    changed = []
    for relative, old_digest in old_sources.items():
        new_digest = sha256((ROOT / relative).resolve(strict=True))
        if new_digest != old_digest:
            changed.append(relative)
    if changed != [UPDATED_DISPATCH]:
        raise ValueError(f"v19-v2 correction source scope drifted: {changed}")

    matrix_sources = [
        descriptor(ROOT / relative, relative=relative) for relative in MATRIX_SOURCES
    ]
    operational_sources = [
        descriptor(ROOT / relative, relative=relative)
        for relative in OPERATIONAL_SOURCES
    ]
    payload = dict(parent)
    payload.update(
        {
            "schema_version": "transit-h2oplus-v19-completion-safety-source-freeze-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "frozen-after-predeployment-environment-binding-correction",
            "evidence_revision": "v2-exact-protocol-environment-binding",
            "matrix_source_count": len(matrix_sources),
            "operational_source_count": len(operational_sources),
            "matrix_sources": matrix_sources,
            "operational_sources": operational_sources,
            "analyzer": descriptor(
                ROOT / "reproducibility/analyze_h2oplus_v19_completion_safety.py",
                relative="reproducibility/analyze_h2oplus_v19_completion_safety.py",
            ),
            "dispatch": descriptor(
                ROOT / UPDATED_DISPATCH, relative=UPDATED_DISPATCH
            ),
            "environment_binding_correction": descriptor(correction_path),
            "superseded_source_freeze_v1": descriptor(parent_path),
            "superseded_local_deployment_manifest_v1": descriptor(v1_manifest_path),
            "superseded_local_deployment_audit_v1": descriptor(v1_audit_path),
            "v19_development_evaluation_outcomes_read": False,
            "formal_held_out_outcomes_read": False,
        }
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v19-v2 source freeze: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256(output),
                "changed_scientific_sources": changed,
                "matrix_sources": len(matrix_sources),
                "operational_sources": len(operational_sources),
                "expected_runtime_sources": payload["expected_matrix_source_file_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
