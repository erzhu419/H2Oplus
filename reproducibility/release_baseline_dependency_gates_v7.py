#!/usr/bin/env python3
"""Release ablation gates from a verified v7 baseline integrity manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.train_transit_v3 import sha256


SCHEMA_VERSION = "transit-baseline-dependency-gate-v7-v1"
HOSTS = ("jtl110gpu", "jtl110gpu2")


def validate(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("status") != "verified"
        or not manifest.get("artifact_integrity_verified")
        or manifest.get("errors")
    ):
        raise RuntimeError("baseline artifact-integrity manifest is not verified")
    if not manifest.get("partition_runs_disjoint"):
        raise RuntimeError("baseline source partitions are not disjoint")
    matrix = manifest.get("independent_matrix_audit", {})
    if not matrix.get("passed") or matrix.get("jobs_audited") != 35:
        raise RuntimeError("baseline independent audit is incomplete")
    if len(manifest.get("merged_results", {}).get("runs", [])) != 35:
        raise RuntimeError("merged baseline run set is incomplete")
    tasks = manifest.get("scheduler_evidence", {}).get("tasks", {})
    for task_id in ("t82016", "t82017"):
        task = tasks.get(task_id, {})
        if task.get("status") != "done" or task.get("exit_code") != 0:
            raise RuntimeError(f"baseline dispatch {task_id} lacks exit-0 evidence")
    stability = manifest.get("scientific_stability", {})
    if sorted(stability.get("failed_seeds", []) + stability.get("passed_seeds", [])) != list(
        range(5101, 5106)
    ):
        raise RuntimeError("baseline stability disclosure is incomplete")


def _write_new(path: Path, payload: dict[str, Any]) -> Path:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return temporary


def release(manifest_path: Path, gate_dir: Path) -> list[Path]:
    manifest_path = manifest_path.expanduser().resolve()
    gate_dir = gate_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    validate(manifest)
    gate_dir.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    paths = [gate_dir / f"baselines_verified_{host}.ready" for host in HOSTS]
    for path in paths:
        temporary = path.with_suffix(path.suffix + ".partial")
        if path.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing dependency release with an occupied gate path: {path}"
            )
    temporary_paths = []
    committed_paths = []
    try:
        for host, path in zip(HOSTS, paths):
            payload = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": created_utc,
                "status": "ready",
                "host": host,
                "artifact_integrity": "verified",
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256(manifest_path),
                },
                "matrix_audit_sha256": manifest["independent_matrix_audit"]["sha256"],
                "rlpd_scientific_stability": {
                    "passed": manifest["scientific_stability"]["passed"],
                    "failed_seeds": manifest["scientific_stability"]["failed_seeds"],
                    "passed_seeds": manifest["scientific_stability"]["passed_seeds"],
                    "audit_sha256": manifest["scientific_stability"]["sha256"],
                    "does_not_block_artifact_dependency": True,
                },
            }
            temporary_paths.append(_write_new(path, payload))
        for temporary, path in zip(temporary_paths, paths):
            os.link(temporary, path)
            committed_paths.append(path)
        for temporary in temporary_paths:
            temporary.unlink()
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        for path in committed_paths:
            path.unlink(missing_ok=True)
        raise
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in release(args.manifest, args.gate_dir):
        print(path)


if __name__ == "__main__":
    main()
