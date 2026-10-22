#!/usr/bin/env python3
"""Fail-closed merge of complete deterministic v7 evaluation partitions."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bus_h2o.sumo_provenance import sha256
from reproducibility import run_evaluation_matrix_v4 as matrix


MERGE_SCHEMA = "transit-evaluation-matrix-merge-v7-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if (
        payload.get("schema_version") != matrix.MATRIX_SCHEMAS["v7"]
        or payload.get("mdp_version") != "v7"
    ):
        raise ValueError(f"not a v7 evaluation matrix manifest: {resolved}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"matrix partition has no jobs: {resolved}")
    partition = payload.get("matrix_partition")
    if not (
        isinstance(partition, dict)
        and partition.get("schema_version") == matrix.MATRIX_PARTITION_SCHEMA
        and partition.get("assignment") == matrix.MATRIX_PARTITION_ASSIGNMENT
    ):
        raise ValueError(f"matrix manifest lacks a v7 deterministic partition: {resolved}")
    return resolved, payload


def _common_identity(payload: dict) -> dict:
    return {
        "schema_version": payload.get("schema_version"),
        "mdp_version": payload.get("mdp_version"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "dataset_size_bytes": payload.get("dataset_size_bytes"),
        "environment_design_sha256": payload.get("environment_design", {}).get(
            "sha256"
        ),
        "evaluator_sha256": payload.get("evaluator_sha256"),
        "runner_sha256": payload.get("runner_sha256"),
        "source_sha256": payload.get("source_sha256"),
        "sumo_version": payload.get("sumo_version"),
        "sumo_backend": payload.get("sumo_backend"),
        "sumo_runtime_signature": payload.get("sumo_runtime_signature"),
        "sumo_runtime_lock_sha256": payload.get("sumo_runtime_lock", {}).get(
            "sha256"
        ),
    }


def _deserialize_job(raw: dict) -> dict:
    job = copy.deepcopy(raw)
    for key in ("dataset", "sumo_dir", "edge_xml", "output", "evaluator"):
        job[key] = Path(job[key]).expanduser().resolve()
    job["event_log"] = (
        Path(job["event_log"]).expanduser().resolve()
        if job.get("event_log") is not None
        else None
    )
    if job.get("sumo_runtime_lock") is not None:
        job["sumo_runtime_lock"] = (
            Path(job["sumo_runtime_lock"]).expanduser().resolve()
        )
    job["scenario"]["path"] = Path(job["scenario"]["path"]).expanduser().resolve()
    if job["method"].get("checkpoint") is not None:
        job["method"]["checkpoint"] = (
            Path(job["method"]["checkpoint"]).expanduser().resolve()
        )
    return job


def _validate_partition_group(records: list[tuple[Path, dict]]) -> list[dict]:
    partitions = [payload["matrix_partition"] for _, payload in records]
    shard_counts = {int(partition.get("shard_count", -1)) for partition in partitions}
    full_counts = {int(partition.get("full_job_count", -1)) for partition in partitions}
    if len(shard_counts) != 1 or len(full_counts) != 1:
        raise ValueError("partition declarations disagree on matrix dimensions")
    shard_count = next(iter(shard_counts))
    full_job_count = next(iter(full_counts))
    indices = [int(partition.get("shard_index", -1)) for partition in partitions]
    if sorted(indices) != list(range(shard_count)):
        raise ValueError(
            f"matrix partition set is incomplete or duplicated: indices={sorted(indices)}"
        )

    indexed_jobs: list[tuple[int, dict]] = []
    for (_, payload), index in zip(records, indices):
        jobs = payload["jobs"]
        if len(jobs) != int(payload["matrix_partition"].get("selected_job_count", -1)):
            raise ValueError("partition selected-job count does not match its manifest")
        selected_hash = matrix.signature(
            {"portable_job_keys": sorted(matrix.portable_job_key(job) for job in jobs)}
        )
        if selected_hash != payload["matrix_partition"].get(
            "selected_job_keys_sha256"
        ):
            raise ValueError("partition selected-job digest does not match its jobs")
        indexed_jobs.extend((index, job) for job in jobs)
    if len(indexed_jobs) != full_job_count:
        raise ValueError(
            f"partition union has {len(indexed_jobs)} jobs; expected {full_job_count}"
        )

    ordered = sorted(
        (matrix.portable_job_key(job), source_index, job)
        for source_index, job in indexed_jobs
    )
    keys = [key for key, _, _ in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("matrix partition union contains duplicate jobs")
    for ordinal, (_, source_index, _) in enumerate(ordered):
        if ordinal % shard_count != source_index:
            raise ValueError("job was emitted by the wrong deterministic partition")
    jobs = [job for _, _, job in ordered]
    declared_signatures = {
        payload["matrix_partition"]["full_matrix_signature"]
        for _, payload in records
    }
    expected_signature = matrix.full_matrix_partition_signature(jobs, records[0][1])
    if declared_signatures != {expected_signature}:
        raise ValueError("partition full-matrix signature does not match its union")
    return jobs


def merge_manifests(
    manifest_paths: list[Path],
    output_root: Path,
    *,
    verify_outputs: bool = True,
) -> Path:
    if not manifest_paths:
        raise ValueError("at least one matrix partition manifest is required")
    records = [_load_manifest(path) for path in manifest_paths]
    resolved_paths = [path for path, _ in records]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("matrix partition manifest paths must be unique")

    identities = {_canonical_json(_common_identity(payload)) for _, payload in records}
    if len(identities) != 1:
        raise ValueError("matrix partitions do not share one evaluation identity")

    groups: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for record in records:
        full_signature = record[1]["matrix_partition"].get("full_matrix_signature")
        if not isinstance(full_signature, str) or len(full_signature) != 64:
            raise ValueError("matrix partition has a malformed full-matrix signature")
        groups[full_signature].append(record)

    jobs = []
    for group in groups.values():
        jobs.extend(_validate_partition_group(group))
    jobs.sort(key=matrix.portable_job_key)
    job_keys = [matrix.portable_job_key(job) for job in jobs]
    output_paths = [str(Path(job["output"]).expanduser().resolve()) for job in jobs]
    if len(job_keys) != len(set(job_keys)) or len(output_paths) != len(set(output_paths)):
        raise ValueError("merged matrix contains duplicate jobs or output paths")

    coverage: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    replicate_cells = set()
    for raw_job in jobs:
        method = raw_job["method"]
        cell = (
            str(raw_job["environment_block"]),
            str(raw_job["scenario"]["manifest_sha256"]),
            int(raw_job["sumo_seed"]),
        )
        coverage[str(method["name"])].add(cell)
        # A method name identifies one policy artifact. Multiple checkpoints from
        # the same algorithm/seed are distinct methods in learning-curve grids.
        replicate_key = (str(method["name"]), cell)
        if replicate_key in replicate_cells:
            raise ValueError(f"duplicate method/environment cell: {replicate_key}")
        replicate_cells.add(replicate_key)
        if verify_outputs and not matrix._is_complete(_deserialize_job(raw_job)):
            raise ValueError(f"evaluation output is incomplete or invalid: {raw_job['output']}")
    expected_cells = next(iter(coverage.values()))
    incomplete = {
        method: {
            "missing": sorted(expected_cells - cells),
            "extra": sorted(cells - expected_cells),
        }
        for method, cells in coverage.items()
        if cells != expected_cells
    }
    if incomplete:
        raise ValueError(f"methods do not share one complete environment design: {incomplete}")

    first = copy.deepcopy(records[0][1])
    method_manifests = sorted(
        {
            (
                str(payload.get("methods_manifest")),
                str(payload.get("methods_manifest_sha256")),
            )
            for _, payload in records
        }
    )
    dataset_stats = sorted(
        {
            (
                int(payload.get("dataset_size_bytes", -1)),
                int(payload.get("dataset_mtime_ns", -1)),
            )
            for _, payload in records
        }
    )
    first.pop("matrix_partition", None)
    first["created_utc"] = datetime.now(timezone.utc).isoformat()
    first["jobs"] = jobs
    first["workers"] = sum(int(payload.get("workers", 0)) for _, payload in records)
    first["worker_selection"] = {
        "mode": "merged-partitions",
        "maximum_declared_concurrent_workers": first["workers"],
    }
    first["methods_manifest"] = method_manifests[0][0] if len(method_manifests) == 1 else None
    first["methods_manifest_sha256"] = (
        method_manifests[0][1]
        if len(method_manifests) == 1
        else matrix.signature({"methods_manifests": method_manifests})
    )
    first["methods_manifests"] = [
        {"path": path, "sha256": digest} for path, digest in method_manifests
    ]
    first["dataset_stat_identities"] = [
        {"size_bytes": size, "mtime_ns": mtime} for size, mtime in dataset_stats
    ]
    first["matrix_merge"] = {
        "schema_version": MERGE_SCHEMA,
        "merger": str(Path(__file__).resolve()),
        "merger_sha256": sha256(Path(__file__).resolve()),
        "source_manifests": [
            {"path": str(path), "sha256": sha256(path)} for path in resolved_paths
        ],
        "full_matrix_signatures": sorted(groups),
        "merged_jobs_sha256": matrix.signature(
            {"job_signatures": sorted(str(job["signature"]) for job in jobs)}
        ),
        "job_count": len(jobs),
        "method_count": len(coverage),
        "environment_cells_per_method": len(expected_cells),
        "all_outputs_verified": verify_outputs,
    }

    destination_root = output_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "matrix_manifest.json"
    if destination in resolved_paths:
        raise ValueError("merged manifest cannot overwrite one of its partition inputs")
    matrix._archive_existing_manifest(destination)
    matrix._write_json(destination, first)
    return destination


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    destination = merge_manifests(args.manifest, args.output_root)
    print(destination)


if __name__ == "__main__":
    main()
