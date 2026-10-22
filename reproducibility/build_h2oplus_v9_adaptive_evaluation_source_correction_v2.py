#!/usr/bin/env python3
"""Seal the Python 3.8 compatibility correction found before v9 SUMO evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-source-correction-v2"
SOURCE_FREEZE_V1_SHA256 = "434109f7a738d9e642c0b9b724fd352b6cab48b5f54f11a73224170e3e99f474"
DEPLOYMENT_V2_SHA256 = "a688840dc924553ba7a2c3fd6b8d9a84ee66bd1373529b8724fae76e2e439801"
CHANGED = (
    "SimpleSAC/evaluate_transit_v9_adaptive.py",
    "reproducibility/run_evaluation_matrix_v9_adaptive.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    source_freeze = REPRO / "h2oplus_v9_adaptive_evaluation_source_freeze_v1.json"
    deployment = REPRO / "deployments/h2oplus_v9_adaptive_evaluation_v2/workspace/deployment_manifest.json"
    remote_verification = (
        REPRO / "h2oplus_v9_adaptive_evaluation_deployment_v2_jtl110gpu_verification.json"
    )
    portability = REPRO / "h2oplus_v9_adaptive_evaluation_portability_correction_v1.json"
    methods = REPRO / "h2oplus_v9_adaptive_evaluation_methods_v2.json"
    if sha256(source_freeze) != SOURCE_FREEZE_V1_SHA256:
        raise ValueError("v1 source freeze drifted")
    if sha256(deployment) != DEPLOYMENT_V2_SHA256:
        raise ValueError("v2 deployment drifted")
    verification = json.loads(remote_verification.read_text())
    if (
        verification.get("passed") is not False
        or verification.get("remote_returncode") != 1
        or verification.get("remote_sumo_simulation_executed") is not False
        or "removeprefix" not in verification.get("remote_stderr", "")
    ):
        raise ValueError("Python 3.8 failure evidence drifted")
    remote_results = deployment.parent.parent / "remote_results_v2"
    if remote_results.exists() and list(remote_results.rglob("sumo_seed*.json")):
        raise ValueError("v2 unexpectedly produced SUMO outcomes")

    frozen = json.loads(source_freeze.read_text())
    frozen_sources = {row["path"]: row for row in frozen.get("sources", [])}
    changes = []
    replacements = {
        CHANGED[0]: (
            'algorithm.removeprefix("h2oplus-v9-")',
            'algorithm[len("h2oplus-v9-"):]',
        ),
        CHANGED[1]: (
            'raw["algorithm"].removeprefix("h2oplus-v9-")',
            'raw["algorithm"][len("h2oplus-v9-"):]',
        ),
    }
    old_root = deployment.parent / "H2Oplus"
    for relative in CHANGED:
        old_path = old_root / relative
        new_path = ROOT / relative
        old_text = old_path.read_text()
        before, after = replacements[relative]
        if before not in old_text or old_text.replace(before, after) != new_path.read_text():
            raise ValueError(f"source correction is not the exact compatibility rewrite: {relative}")
        if sha256(old_path) != frozen_sources[relative]["sha256"]:
            raise ValueError(f"v1 frozen source evidence drifted: {relative}")
        changes.append(
            {
                "path": relative,
                "before_sha256": sha256(old_path),
                "after_sha256": sha256(new_path),
                "rewrite": {"before": before, "after": after},
            }
        )

    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "v2-invalid-before-sumo-python38-source-corrected",
        "scope": "Python 3.8 compatibility only",
        "reason": (
            "the frozen remote runtime is Python 3.8, where str.removeprefix is unavailable; "
            "equivalent prefix slicing preserves algorithm semantics"
        ),
        "formal_held_out_outcomes_read": False,
        "scientific_evaluation_outcomes_read": False,
        "remote_sumo_episode_count": 0,
        "training_or_policy_artifacts_changed": False,
        "methods_or_aliases_changed": False,
        "changed_source_paths": list(CHANGED),
        "source_changes": changes,
        "invalid_v2_policy": "never submit, merge, analyze, select, or reuse",
        "source_freeze_v1": descriptor(source_freeze),
        "deployment_v2": descriptor(deployment),
        "failed_remote_verification_v2": descriptor(remote_verification),
        "portability_correction": descriptor(portability),
        "methods_manifest": descriptor(methods),
        "next_stage": (
            "freeze the corrected four-file source set, build a fresh deployment, verify it "
            "under the frozen Python 3.8 runtime, then submit new task signatures"
        ),
    }
    output = REPRO / "h2oplus_v9_adaptive_evaluation_source_correction_v2.json"
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "sha256": sha256(output), "changes": changes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
