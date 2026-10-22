#!/usr/bin/env python3
"""Audit canonical v7 H2O+ offline initializers against a signed plan."""

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

from reproducibility.run_training_matrix_v4 import _is_complete, load_plan
from SimpleSAC.train_transit_v3 import sha256
from SimpleSAC.transit_v7.trainer import (
    OFFLINE_INITIALIZER_SCHEMA,
    inspect_offline_initializer_v7,
)


AUDIT_SCHEMA = "transit-core-initializer-audit-v7-v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit(
    plan_path: Path,
    matrix_root: Path,
    *,
    require_run_complete: bool,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    matrix_root = matrix_root.expanduser().resolve()
    jobs, evidence = load_plan(plan_path)
    plan = json.loads(plan_path.read_text())
    checkpoint_version = plan.get("registration", {}).get("checkpoint_version")
    admission = evidence.get("training_admission") or {}
    expected_admission = {
        key: admission.get(key)
        for key in ("bytes", "path", "schema_version", "selected_arguments", "sha256")
    }
    producer_jobs = [
        job
        for job in jobs
        if job["method"] == "h2oplus"
        and int(job["arguments"].get("offline_updates", -1)) == 50_000
    ]
    errors: list[str] = []
    reports: dict[str, dict[str, Any]] = {}

    if not producer_jobs:
        errors.append("the plan has no canonical H2O+ initializer producers")
    for job in producer_jobs:
        name = job["name"]
        seed = job["seed"]
        prefix = f"{name}: "
        run_dir = matrix_root / name
        initializer = run_dir / "checkpoints" / "offline_initializer.pt"
        if not initializer.is_file():
            errors.append(prefix + f"missing {initializer}")
            continue
        try:
            inspected = inspect_offline_initializer_v7(initializer)
        except Exception as error:
            errors.append(prefix + f"inspection failed: {type(error).__name__}: {error}")
            continue

        artifact_sha256 = sha256(initializer)
        extra = inspected.get("extra") or {}
        checks = {
            "artifact_schema": inspected.get("artifact_schema")
            == OFFLINE_INITIALIZER_SCHEMA,
            "checkpoint_version": inspected.get("checkpoint_version")
            == checkpoint_version,
            "total_updates": inspected.get("total_updates") == 50_000,
            "phase": extra.get("phase") == "offline_pretraining",
            "source_method": extra.get("source_method") == "h2oplus",
            "source_run_name": extra.get("source_run_name") == name,
            "source_run_signature": extra.get("source_run_signature")
            == job["signature"],
            "source_training_seed": extra.get("source_training_seed") == seed,
            "offline_updates_completed": extra.get("offline_updates_completed")
            == 50_000,
            "trainer_total_updates": extra.get("trainer_total_updates") == 50_000,
            "dataset_sha256": extra.get("dataset_sha256")
            == evidence["dataset_sha256"],
            "simulator_runtime_sha256": extra.get("simulator_runtime_sha256")
            == evidence["simulator"]["runtime_sha256"],
            "simulator_aggregate_sha256": extra.get("simulator_aggregate_sha256")
            == evidence["simulator"]["aggregate_sha256"],
            "source_sha256": extra.get("source_sha256") == evidence["source_sha256"],
            "training_admission": extra.get("training_admission")
            == expected_admission,
            "source_config_present": isinstance(inspected.get("source_config"), dict)
            and bool(inspected.get("source_config")),
            "compatibility_config_present": isinstance(
                inspected.get("offline_compatibility_config"), dict
            )
            and bool(inspected.get("offline_compatibility_config")),
        }

        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "metadata.json"
        run_evidence_present = summary_path.is_file() and metadata_path.is_file()
        recorded_initializer = None
        metadata_initializer = None
        if run_evidence_present:
            summary = json.loads(summary_path.read_text())
            metadata = json.loads(metadata_path.read_text())
            recorded_initializer = summary.get("offline_initializer_produced")
            metadata_initializer = metadata.get("offline_initializer_produced")
            expected_record = {
                "bytes": initializer.stat().st_size,
                "offline_updates": 50_000,
                "schema_version": OFFLINE_INITIALIZER_SCHEMA,
                "sha256": artifact_sha256,
                "source_method": "h2oplus",
                "source_run_name": name,
                "source_run_signature": job["signature"],
                "source_training_seed": seed,
            }
            expected_suffix = f"/{name}/checkpoints/offline_initializer.pt"

            def record_matches(record: Any) -> bool:
                return (
                    isinstance(record, dict)
                    and all(record.get(key) == value for key, value in expected_record.items())
                    and str(record.get("path") or "").endswith(expected_suffix)
                )

            checks["summary_initializer_record"] = record_matches(recorded_initializer)
            checks["metadata_initializer_record"] = record_matches(metadata_initializer)
            checks["metadata_training_admission"] = (
                metadata.get("training_admission") == expected_admission
            )
        elif require_run_complete:
            checks["run_evidence_present"] = False

        run_complete = None
        if require_run_complete:
            run_complete = _is_complete(job, evidence)
            checks["signed_run_complete"] = run_complete
        failed_checks = sorted(key for key, passed in checks.items() if not passed)
        if failed_checks:
            errors.append(prefix + "failed checks: " + ", ".join(failed_checks))

        reports[name] = {
            "seed": seed,
            "path": str(initializer),
            "bytes": initializer.stat().st_size,
            "sha256": artifact_sha256,
            "artifact_schema": inspected.get("artifact_schema"),
            "checkpoint_version": inspected.get("checkpoint_version"),
            "source_run_signature": extra.get("source_run_signature"),
            "source_sha256_count": len(extra.get("source_sha256") or {}),
            "training_admission": extra.get("training_admission"),
            "run_evidence_present": run_evidence_present,
            "signed_run_complete": run_complete,
            "checks": checks,
            "failed_checks": failed_checks,
        }

    expected_names = {job["name"] for job in producer_jobs}
    discovered_names = {
        path.parent.parent.name
        for path in matrix_root.glob("h2oplus_seed*/checkpoints/offline_initializer.pt")
    }
    if discovered_names != expected_names:
        errors.append(
            "initializer producer set differs from plan: "
            f"expected={sorted(expected_names)} discovered={sorted(discovered_names)}"
        )

    return {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "matrix_root": str(matrix_root),
        "require_run_complete": require_run_complete,
        "initializers_expected": len(producer_jobs),
        "initializers_audited": len(reports),
        "source_sha256_count": len(evidence["source_sha256"]),
        "initializers": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-run-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        args.plan,
        args.matrix_root,
        require_run_complete=args.require_run_complete,
    )
    if args.output is not None:
        _write_json(args.output, report)
        print(args.output.expanduser().resolve())
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
