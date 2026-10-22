#!/usr/bin/env python3
"""Release sensitivity gates from a verified v7 ablation manifest."""

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


SCHEMA_VERSION = "transit-ablation-dependency-gate-v7-v1"
VERIFICATION_SCHEMA = "transit-formal-ablation-manifest-verification-v7-v1"
HOSTS = ("jtl110gpu", "jtl110gpu2")
METHODS = ("h2oplus", "h2oplus-no-ratio", "h2oplus-no-warmstart")
SEEDS = tuple(range(5101, 5106))
EXPECTED_RUNS = {
    f"{method}_seed{seed}" for method in METHODS[1:] for seed in SEEDS
}


def validate(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("status") != "verified"
        or not manifest.get("artifact_integrity_verified")
        or manifest.get("errors")
    ):
        raise RuntimeError("ablation artifact-integrity manifest is not verified")
    if not manifest.get("partition_runs_disjoint"):
        raise RuntimeError("ablation source partitions are not disjoint")
    matrix = manifest.get("independent_matrix_audit", {})
    if not matrix.get("passed") or matrix.get("jobs_audited") != 10:
        raise RuntimeError("ablation independent audit is incomplete")
    runs = manifest.get("merged_results", {}).get("runs", [])
    if len(runs) != len(EXPECTED_RUNS) or set(runs) != EXPECTED_RUNS:
        raise RuntimeError("merged ablation run set is incomplete")
    tasks = manifest.get("scheduler_evidence", {}).get("tasks", {})
    for task_id in ("t82053", "t82054"):
        task = tasks.get(task_id, {})
        completion = task.get("completion_evidence") or {}
        queue_success = task.get("status") == "done" and task.get("exit_code") == 0
        direct_success = (
            completion.get("verified") is True
            and completion.get("backend_exit", {}).get("exit_code") == 0
            and completion.get("host_execution", {}).get("completed") is True
        )
        if not (queue_success or direct_success):
            raise RuntimeError(f"ablation dispatch {task_id} lacks exit-0 evidence")
    analysis = manifest.get("mechanism_analysis", {})
    if tuple(analysis.get("methods", [])) != METHODS:
        raise RuntimeError("ablation mechanism-analysis method set is incomplete")
    if analysis.get("seeds") != list(SEEDS):
        raise RuntimeError("ablation mechanism-analysis seed set is incomplete")
    if analysis.get("supports_held_out_model_effect_claim") is not False:
        raise RuntimeError("ablation analysis lacks the held-out-effect disclaimer")


def _write_new(path: Path, payload: dict[str, Any]) -> Path:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return temporary


def release(
    manifest_path: Path, verification_path: Path, gate_dir: Path
) -> list[Path]:
    manifest_path = manifest_path.expanduser().resolve()
    verification_path = verification_path.expanduser().resolve()
    gate_dir = gate_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    validate(manifest)
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA
        or not verification.get("passed")
        or verification.get("errors")
        or verification.get("manifest", {}).get("sha256") != sha256(manifest_path)
    ):
        raise RuntimeError("ablation manifest lacks an independent passing verification")
    gate_dir.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    paths = [gate_dir / f"ablation_verified_{host}.ready" for host in HOSTS]
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
                "manifest_verification": {
                    "path": str(verification_path),
                    "sha256": sha256(verification_path),
                },
                "matrix_audit_sha256": manifest["independent_matrix_audit"][
                    "sha256"
                ],
                "mechanism_analysis": {
                    "sha256": manifest["mechanism_analysis"]["sha256"],
                    "methods": list(METHODS),
                    "seeds": list(SEEDS),
                    "scope": "training-internal mechanism diagnosis",
                    "supports_held_out_model_effect_claim": False,
                    "does_not_select_or_replace_formal_results": True,
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
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in release(args.manifest, args.verification, args.gate_dir):
        print(path)


if __name__ == "__main__":
    main()
