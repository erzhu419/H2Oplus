#!/usr/bin/env python3
"""Independently verify the formal v7 H2O+ ablation merge manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "transit-formal-ablation-manifest-verification-v7-v1"
MANIFEST_SCHEMA = "transit-formal-ablation-merge-manifest-v7-v1"
METHODS = ("h2oplus-no-ratio", "h2oplus-no-warmstart")
ANALYSIS_METHODS = ("h2oplus",) + METHODS
SEEDS = tuple(range(5101, 5106))
EXPECTED_RUNS = {
    f"{method}_seed{seed}" for method in METHODS for seed in SEEDS
}
HOSTS = ("jtl110gpu", "jtl110gpu2")
TASKS = ("t82053", "t82054")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def file_records(root: Path, names: list[str]) -> list[dict[str, Any]]:
    return [
        _record(path, root)
        for name in names
        for path in sorted((root / name).rglob("*"))
        if path.is_file()
    ]


def records_sha256(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_descriptor(
    label: str, descriptor: dict[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    try:
        path = Path(descriptor["path"]).expanduser().resolve()
    except (KeyError, TypeError):
        errors.append(f"{label} descriptor lacks a path")
        return None
    if not path.is_file():
        errors.append(f"{label} file is missing: {path}")
        return None
    if descriptor.get("sha256") != _sha256(path):
        errors.append(f"{label} SHA-256 mismatch")
    if "bytes" in descriptor and descriptor.get("bytes") != path.stat().st_size:
        errors.append(f"{label} byte count mismatch")
    try:
        return json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        errors.append(f"{label} is not valid JSON")
        return None


def _execution_jobs(payload: dict[str, Any], errors: list[str], host: str) -> list[str]:
    if payload.get("status") != "completed":
        errors.append(f"{host} host execution is not completed")
    jobs: list[str] = []
    for gpu in payload.get("gpus", []):
        if gpu.get("status") != "completed":
            errors.append(f"{host} GPU execution is not completed")
        for stage in gpu.get("stages", []):
            if stage.get("status") != "completed":
                errors.append(f"{host} stage execution is not completed")
            for worker in stage.get("workers", []):
                if worker.get("status") != "completed" or worker.get("returncode") != 0:
                    errors.append(f"{host} worker lacks return-code-zero completion")
                jobs.extend(str(name) for name in worker.get("jobs", []))
    if len(jobs) != len(set(jobs)):
        errors.append(f"{host} host execution repeats a job")
    return jobs


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    expected = sorted(EXPECTED_RUNS)

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unexpected ablation manifest schema")
    if (
        manifest.get("status") != "verified"
        or not manifest.get("artifact_integrity_verified")
        or manifest.get("errors")
    ):
        errors.append("ablation manifest is not verified and error-free")

    merged = manifest.get("merged_results", {})
    root = Path(merged.get("root", "")).expanduser().resolve()
    if merged.get("runs") != expected:
        errors.append("merged run set or order is not the exact registered matrix")
    actual_records = file_records(root, expected) if root.is_dir() else []
    listed_records = merged.get("files_manifest", [])
    if listed_records != actual_records:
        errors.append("merged file manifest differs from the canonical tree")
    if merged.get("files") != len(actual_records):
        errors.append("merged file count is incorrect")
    if merged.get("bytes") != sum(record["bytes"] for record in actual_records):
        errors.append("merged byte count is incorrect")
    if merged.get("records_sha256") != records_sha256(actual_records):
        errors.append("merged file-record digest is incorrect")

    frozen = manifest.get("frozen_inputs", {})
    plan = _load_descriptor("plan", frozen.get("plan", {}), errors)
    dispatch = _load_descriptor("dispatch", frozen.get("dispatch", {}), errors)
    matrix = _load_descriptor("matrix manifest", manifest.get("matrix_manifest", {}), errors)
    audit = _load_descriptor(
        "independent matrix audit", manifest.get("independent_matrix_audit", {}), errors
    )
    analysis = _load_descriptor(
        "mechanism analysis", manifest.get("mechanism_analysis", {}), errors
    )
    scheduler = _load_descriptor(
        "scheduler evidence", manifest.get("scheduler_evidence", {}), errors
    )

    if plan is not None:
        names = [job.get("name") for job in plan.get("jobs", [])]
        if len(names) != len(expected) or set(names) != EXPECTED_RUNS:
            errors.append("frozen plan does not contain the exact ten jobs")
    if dispatch is not None:
        dispatched = [
            str(name)
            for host in dispatch.get("hosts", [])
            for gpu in host.get("gpus", [])
            for stage in gpu.get("stages", [])
            for worker in stage.get("workers", [])
            for name in worker.get("jobs", [])
        ]
        if len(dispatched) != len(expected) or set(dispatched) != EXPECTED_RUNS:
            errors.append("frozen dispatch does not assign the exact ten jobs once")
    if matrix is not None:
        if matrix.get("status") != "completed":
            errors.append("matrix manifest is not completed")
        if sorted(matrix.get("results", {})) != expected:
            errors.append("matrix manifest result set is incomplete")
        if matrix.get("plan_sha256") != frozen.get("plan", {}).get("sha256"):
            errors.append("matrix manifest does not bind the frozen plan")
        if matrix.get("partition_merge", {}).get("dispatch", {}).get(
            "sha256"
        ) != frozen.get("dispatch", {}).get("sha256"):
            errors.append("matrix manifest does not bind the frozen dispatch")
    if audit is not None:
        if not audit.get("passed") or audit.get("errors"):
            errors.append("independent matrix audit is not error-free")
        if audit.get("jobs_audited") != len(expected) or set(
            audit.get("jobs", {})
        ) != EXPECTED_RUNS:
            errors.append("independent matrix audit does not cover all ten jobs")
        if audit.get("plan", {}).get("sha256") != frozen.get("plan", {}).get(
            "sha256"
        ):
            errors.append("independent matrix audit does not bind the frozen plan")
        if audit.get("matrix_manifest", {}).get("sha256") != manifest.get(
            "matrix_manifest", {}
        ).get("sha256"):
            errors.append("independent matrix audit does not bind the matrix manifest")
        initializer = audit.get("initializer_root") or {}
        initializer_root = Path(initializer.get("path", "")).expanduser().resolve()
        for relative, expected_hash in initializer.get("files", {}).items():
            path = initializer_root / relative
            if not path.is_file() or _sha256(path) != expected_hash:
                errors.append(f"independent audit initializer mismatch: {relative}")
    if analysis is not None:
        if analysis.get("scope") != "exploratory training-internal mechanism diagnosis":
            errors.append("mechanism analysis scope is incorrect")
        interpretation = analysis.get("scientific_interpretation", {})
        if interpretation.get("supports_held_out_model_effect_claim") is not False:
            errors.append("mechanism analysis lacks held-out-effect disclaimer")
        if interpretation.get("supports_formal_method_selection") is not False:
            errors.append("mechanism analysis lacks formal-selection disclaimer")
        if tuple(analysis.get("methods", [])) != ANALYSIS_METHODS:
            errors.append("mechanism analysis method set is incorrect")
        if analysis.get("seeds") != list(SEEDS):
            errors.append("mechanism analysis seed set is incomplete")
    if scheduler is not None:
        task_map = {task.get("id"): task for task in scheduler.get("tasks", [])}
        for task_id in TASKS:
            task = task_map.get(task_id, {})
            completion = task.get("completion_evidence") or {}
            queue_success = task.get("status") == "done" and task.get("exit_code") == 0
            direct_success = (
                completion.get("verified") is True
                and completion.get("status") == "verified"
                and completion.get("backend_exit", {}).get("exit_code") == 0
                and completion.get("host_execution", {}).get("completed") is True
            )
            if not (queue_success or direct_success):
                errors.append(f"scheduler evidence lacks exit-0 completion for {task_id}")
            if not task.get("exit_status_token") or not task.get("exit_status_path"):
                errors.append(f"scheduler evidence lacks token-bound exit status for {task_id}")
            if direct_success:
                for label in ("backend_exit", "host_execution"):
                    descriptor = completion.get(label, {})
                    path = Path(descriptor.get("path", "")).expanduser().resolve()
                    if (
                        not path.is_file()
                        or descriptor.get("sha256") != _sha256(path)
                        or descriptor.get("bytes") != path.stat().st_size
                    ):
                        errors.append(
                            f"scheduler direct-completion {label} mismatch for {task_id}"
                        )

    partitions = manifest.get("source_partitions", {})
    if set(partitions) != set(HOSTS):
        errors.append("source partition host set is incorrect")
    seen_jobs: set[str] = set()
    merged_by_path = {record["path"]: record for record in actual_records}
    source_paths: set[str] = set()
    for host in HOSTS:
        partition = partitions.get(host, {})
        names = partition.get("jobs", [])
        if seen_jobs.intersection(names):
            errors.append(f"source partitions overlap on jobs for {host}")
        seen_jobs.update(names)
        source_root = Path(partition.get("root", "")).expanduser().resolve()
        source_records = file_records(source_root, names) if source_root.is_dir() else []
        if partition.get("files") != len(source_records):
            errors.append(f"{host} source file count is incorrect")
        if partition.get("bytes") != sum(record["bytes"] for record in source_records):
            errors.append(f"{host} source byte count is incorrect")
        if partition.get("records_sha256") != records_sha256(source_records):
            errors.append(f"{host} source file-record digest is incorrect")
        for record in source_records:
            relative = record["path"]
            if relative in source_paths:
                errors.append(f"source partitions overlap on file {relative}")
            source_paths.add(relative)
            if merged_by_path.get(relative) != record:
                errors.append(f"merged file differs from {host}: {relative}")
        execution = _load_descriptor(
            f"{host} host execution", partition.get("host_execution", {}), errors
        )
        if execution is not None and set(_execution_jobs(execution, errors, host)) != set(
            names
        ):
            errors.append(f"{host} execution job set differs from its partition")
    if seen_jobs != EXPECTED_RUNS:
        errors.append("source partition job union is incomplete")
    if source_paths != set(merged_by_path):
        errors.append("source partition file union differs from the merged tree")

    return {
        "schema_version": SCHEMA_VERSION,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "runs_verified": len(expected) if not errors else 0,
        "merged_files_verified": len(actual_records) if not errors else 0,
        "merged_bytes_verified": (
            sum(record["bytes"] for record in actual_records) if not errors else 0
        ),
        "source_partitions_verified": list(HOSTS) if not errors else [],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify(args.manifest)
    _write_json(args.output, report)
    print(args.output.expanduser().resolve())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
