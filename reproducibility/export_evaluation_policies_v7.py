#!/usr/bin/env python3
"""Export completed v7 actors and provenance without replay or optimizer state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.build_training_plan_v7 import (
    FORMAL_SEEDS,
    H2OPLUS_METHODS,
    PLAN_SCHEMA,
    SERVER_EXECUTION_CLASS,
    SERVER_H2OPLUS_EXECUTION_CLASS,
    SERVER_METHODS,
)
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    V7_POLICY_SELECTION_RULE,
)
from reproducibility.run_training_matrix_v4 import _is_complete, load_plan
from SimpleSAC.train_transit_v3 import sha256


EXPORT_SCHEMA = "transit-policy-export-v7-v1"
MANIFEST_NAME = "policy_export_manifest.json"
METHODS_NAME = "evaluation_methods_v7_subset.json"


def _descriptor(path: Path, base: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": os.path.relpath(resolved, base.resolve()),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _copy_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to replace exported policy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"exported policy hash mismatch: {destination}")


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen export: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_plan_registration(
    plan_path: Path,
    payload: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"policy export requires a v7 plan: {plan_path}")
    registration = payload.get("registration", {})
    if registration.get("status") != "frozen-before-formal-outcomes":
        raise ValueError(f"policy export rejects a non-formal plan: {plan_path}")
    execution_class = payload.get("execution_class")
    methods = {job["method"] for job in jobs}
    if execution_class == SERVER_H2OPLUS_EXECUTION_CLASS:
        allowed = set(H2OPLUS_METHODS)
    elif execution_class == SERVER_EXECUTION_CLASS:
        allowed = set(SERVER_METHODS)
    else:
        raise ValueError(f"policy export rejects execution class {execution_class!r}")
    if not methods or not methods <= allowed:
        raise ValueError(f"policy export plan violates its execution boundary: {plan_path}")
    if {job["seed"] for job in jobs} != set(FORMAL_SEEDS):
        raise ValueError(f"policy export plan lacks all formal seeds: {plan_path}")
    artifact = registration.get("formal_policy_artifact", {})
    if artifact.get("schema_version") != "transit-policy-artifact-v7-v1":
        raise ValueError(f"policy export plan lacks the v7 actor contract: {plan_path}")
    evaluation_design = registration.get("evidence", {}).get(
        "evaluation_environment_design", {}
    )
    if not isinstance(evaluation_design.get("sha256"), str):
        raise ValueError(f"policy export plan lacks an evaluation-design lock: {plan_path}")
    return execution_class, evaluation_design


def export_policies(plan_paths: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace policy export: {destination}")
    destination.mkdir(parents=True)
    methods = []
    plans = []
    seen: set[tuple[str, int]] = set()
    admission_hashes = set()
    evaluation_design_hashes = set()
    provenance_file_count = 0
    try:
        for raw_plan_path in plan_paths:
            plan_path = raw_plan_path.expanduser().resolve(strict=True)
            plan_payload = json.loads(plan_path.read_text())
            jobs, evidence = load_plan(plan_path)
            execution_class, evaluation_design = _validate_plan_registration(
                plan_path, plan_payload, jobs
            )
            admission_sha256 = evidence.get("training_admission", {}).get("sha256")
            admission_hashes.add(admission_sha256)
            evaluation_design_hashes.add(evaluation_design["sha256"])
            plan_copy = (
                destination
                / "provenance"
                / "plans"
                / f"{plan_payload.get('plan_id')}.json"
            )
            _copy_new(plan_path, plan_copy)
            provenance_file_count += 1
            plans.append(
                {
                    **_descriptor(plan_copy, destination),
                    "source": {
                        "path": str(plan_path),
                        "sha256": sha256(plan_path),
                    },
                    "plan_id": plan_payload.get("plan_id"),
                    "execution_class": execution_class,
                    "methods": sorted({job["method"] for job in jobs}),
                    "seeds": sorted({job["seed"] for job in jobs}),
                }
            )
            for job in jobs:
                identity = (job["method"], job["seed"])
                if identity in seen:
                    raise ValueError(f"duplicate exported policy identity: {identity}")
                seen.add(identity)
                if not _is_complete(job, evidence):
                    raise ValueError(f"training run is not formally complete: {job['name']}")
                run_dir = job["run_dir"]
                summary = json.loads((run_dir / "summary.json").read_text())
                source_actor = run_dir / "checkpoints" / "final_actor.pt"
                exported_actor = (
                    destination
                    / "policies"
                    / job["method"]
                    / f"seed{job['seed']}"
                    / "final_actor.pt"
                )
                _copy_new(source_actor, exported_actor)
                run_provenance = {}
                provenance_root = (
                    destination
                    / "provenance"
                    / "runs"
                    / job["method"]
                    / f"seed{job['seed']}"
                )
                for filename in (
                    "metadata.json",
                    "summary.json",
                    "phase_metrics.json",
                    "metrics.jsonl",
                    "timings.json",
                ):
                    source = run_dir / filename
                    if not source.is_file():
                        continue
                    copied = provenance_root / filename
                    _copy_new(source, copied)
                    run_provenance[filename] = _descriptor(copied, destination)
                    provenance_file_count += 1
                methods.append(
                    {
                        "name": f"{job['method']}-seed{job['seed']}",
                        "algorithm": job["method"],
                        "training_seed": job["seed"],
                        "type": "checkpoint",
                        "checkpoint": os.path.relpath(
                            exported_actor, destination
                        ),
                        "checkpoint_sha256": sha256(exported_actor),
                        "training_run_signature": job["signature"],
                        "training_admission_sha256": admission_sha256,
                        "training_checkpoint_sha256": summary["checkpoint"][
                            "sha256"
                        ],
                        "training_plan_sha256": evidence["plan_sha256"],
                        "execution_class": execution_class,
                        "offline_initializer_consumed": summary.get(
                            "offline_initializer_consumed"
                        ),
                        "offline_initializer_produced": summary.get(
                            "offline_initializer_produced"
                        ),
                        "training_artifacts": run_provenance,
                    }
                )
        if not plans:
            raise ValueError("policy export requires at least one formal v7 plan")
        if len(admission_hashes) != 1 or None in admission_hashes:
            raise ValueError("exported plans do not share one training admission")
        if len(evaluation_design_hashes) != 1:
            raise ValueError("exported plans do not share one evaluation design")
        methods.sort(key=lambda item: (item["algorithm"], item["training_seed"]))
        methods_payload = {
            "schema_version": METHODS_SCHEMAS["v7"],
            "study_stage": "formal",
            "selection_rule": V7_POLICY_SELECTION_RULE,
            "scope": "verified subset; merge before the full formal matrix",
            "methods": [
                {
                    key: value
                    for key, value in method.items()
                    if key not in {"checkpoint_sha256", "execution_class"}
                }
                for method in methods
            ],
        }
        _write_new(destination / METHODS_NAME, methods_payload)
        manifest = {
            "schema_version": EXPORT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "source_root": str(ROOT),
            "selection_rule": V7_POLICY_SELECTION_RULE,
            "training_admission_sha256": next(iter(admission_hashes)),
            "evaluation_environment_design_sha256": next(
                iter(evaluation_design_hashes)
            ),
            "plans": plans,
            "methods": methods,
            "payload_policy": {
                "compact_final_actor_files": len(methods),
                "training_checkpoint_files": 0,
                "optimizer_or_replay_files": 0,
                "legacy_snapshot_files": 0,
                "audit_provenance_files": provenance_file_count,
            },
            "subset_methods_manifest": _descriptor(
                destination / METHODS_NAME, destination
            ),
            "generator": _descriptor(Path(__file__).resolve(), destination),
        }
        _write_new(destination / MANIFEST_NAME, manifest)
        return manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_policies(args.plan, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir.expanduser().resolve()),
                "policies": len(manifest["methods"]),
                "manifest_sha256": sha256(
                    args.output_dir.expanduser().resolve() / MANIFEST_NAME
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
