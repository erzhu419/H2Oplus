"""Build a fail-closed audit of diagnostic v7 CPU evaluation throughput."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import re


SCHEMA_VERSION = "transit-v7-cpu-evaluation-throughput-audit-v1"
RUNTIME_SIGNATURE = "1dff3f98dc13d6e33a23a4333346624fc1b2162c96c89192496ce20da5eca78f"
DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_seconds(value: str) -> float:
    parts = [float(item) for item in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unsupported elapsed time: {value}")


def load_timing(path: Path) -> dict:
    text = path.read_text()

    def value(label: str) -> str:
        match = re.search(
            rf"^\s*{re.escape(label)}:\s*(.+)$",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise ValueError(f"missing timing field {label!r} in {path}")
        return match.group(1).strip()

    elapsed = value("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "elapsed_wall_seconds": _elapsed_seconds(elapsed),
        "user_cpu_seconds": float(value("User time (seconds)")),
        "system_cpu_seconds": float(value("System time (seconds)")),
        "cpu_percent": float(value("Percent of CPU this job got").rstrip("%")),
        "maximum_resident_set_kbytes": int(
            value("Maximum resident set size (kbytes)")
        ),
        "major_page_faults": int(value("Major (requiring I/O) page faults")),
        "minor_page_faults": int(value("Minor (reclaiming a frame) page faults")),
        "voluntary_context_switches": int(value("Voluntary context switches")),
        "involuntary_context_switches": int(
            value("Involuntary context switches")
        ),
        "swaps": int(value("Swaps")),
    }


def audit_results(root: Path, expected: int) -> dict:
    paths = sorted(
        Path(path)
        for path in glob.glob(str(root / "**" / "sumo_seed*.json"), recursive=True)
    )
    if len(paths) != expected:
        raise ValueError(f"expected {expected} results under {root}, got {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    if not all(
        row["evaluation_job"]["diagnostic"]
        and row["episode"]["observed_end_time_seconds"] == 1200.0
        and row["episode"]["stopped_by_time_limit"]
        for row in rows
    ):
        raise ValueError(f"partial or non-diagnostic result under {root}")
    if not all(
        row["runtime"]["sumo_backend"] == "libsumo"
        and row["runtime"]["sumo_runtime_fingerprint"]["portable_signature"]
        == RUNTIME_SIGNATURE
        for row in rows
    ):
        raise ValueError(f"SUMO runtime drift under {root}")
    if not all(row["dataset"]["sha256"] == DATASET_SHA256 for row in rows):
        raise ValueError(f"dataset drift under {root}")
    signatures = [row["evaluation_job"]["signature"] for row in rows]
    if len(set(signatures)) != expected:
        raise ValueError(f"duplicate job signatures under {root}")
    counts = Counter(row["method"] for row in rows)
    transitions = [row["episode"]["transitions"] for row in rows]
    return {
        "validated_results": len(rows),
        "unique_job_signatures": len(set(signatures)),
        "method_counts": dict(sorted(counts.items())),
        "transition_sum": sum(transitions),
        "transition_min": min(transitions),
        "transition_max": max(transitions),
    }


def _single_node_level(results_root: Path, gate: str, workers: int, jobs: int) -> dict:
    root = results_root / gate / f"workers{workers:02d}"
    metrics = load_timing(results_root / gate / f"time_workers{workers:02d}.txt")
    metrics.update(audit_results(root, jobs))
    manifest = root / "matrix_manifest.json"
    metrics["matrix_manifest"] = {
        "path": str(manifest.resolve()),
        "sha256": sha256(manifest),
    }
    metrics["jobs_per_wall_second"] = jobs / metrics["elapsed_wall_seconds"]
    return metrics


def _deployment_identity(root: Path) -> dict:
    manifest = root / "deployment_manifest.json"
    payload = json.loads(manifest.read_text())
    return {
        "path": str(root.resolve()),
        "manifest_sha256": sha256(manifest),
        "entry_count": payload["entry_count"],
        "aggregate_sha256": payload["aggregate_sha256"],
    }


def build(args: argparse.Namespace) -> dict:
    results_root = args.results_root.expanduser().resolve()
    small = {
        str(workers): _single_node_level(
            results_root, "throughput_gate_v1", workers, 30
        )
        for workers in (8, 16, 30)
    }
    replicated = {
        str(workers): _single_node_level(
            results_root, "throughput_gate_v2", workers, 120
        )
        for workers in (32, 48, 64, 80)
    }

    multi_root = results_root / "throughput_gate_v2" / "multinode_6x16"
    multi = audit_results(multi_root, 120)
    multi.update({"workers_per_node": 16, "shard_count": 6, "nodes": {}})
    for index in range(6):
        node = f"node{index + 1:03d}"
        metrics = load_timing(
            results_root / "throughput_gate_v2" / f"time_multinode_{node}.txt"
        )
        manifest = multi_root / f"matrix_manifest.shard-{index:04d}-of-0006.json"
        manifest_payload = json.loads(manifest.read_text())
        metrics.update(
            {
                "shard_index": index,
                "selected_jobs": len(manifest_payload["jobs"]),
                "matrix_manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": sha256(manifest),
                },
            }
        )
        metrics["jobs_per_wall_second"] = (
            metrics["selected_jobs"] / metrics["elapsed_wall_seconds"]
        )
        multi["nodes"][node] = metrics
    multi["barrier_wall_seconds"] = max(
        metrics["elapsed_wall_seconds"] for metrics in multi["nodes"].values()
    )
    multi["aggregate_jobs_per_barrier_second"] = (
        multi["validated_results"] / multi["barrier_wall_seconds"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "diagnostic",
        "formal_result_eligible": False,
        "warning": (
            "Replicated zero-hold jobs measure execution throughput only and "
            "must never be used as policy-effectiveness evidence."
        ),
        "data": {"sha256": DATASET_SHA256, "snapshot_data_used": False},
        "sumo_runtime": {
            "backend": "libsumo",
            "version": "1.25.0",
            "portable_signature": RUNTIME_SIGNATURE,
        },
        "benchmark_deployment": _deployment_identity(args.benchmark_deployment),
        "implemented_deployment": _deployment_identity(args.implemented_deployment),
        "isolation": {
            "exclusive_node": False,
            "background_load_present": True,
            "interpretation": (
                "operational shared-cluster gate, not an isolated hardware benchmark"
            ),
        },
        "small_matrix_single_node": small,
        "replicated_matrix_single_node": replicated,
        "replicated_matrix_multinode": multi,
        "decision": {
            "automatic_single_process_cpu_fraction": "2/3",
            "automatic_worker_cap_on_96_physical_cores": 64,
            "shared_cluster_workers_per_node": 16,
            "formal_dispatch": (
                "use more deterministic shards than nodes and assign the next shard "
                "to the next available eligible node; merge only after every shard "
                "manifest passes"
            ),
            "do_not_use_workers": 80,
            "reason": (
                "80 workers was slower and consumed substantially more CPU time than "
                "64; equal six-node sharding exposed a node006 straggler, so dynamic "
                "shard assignment is required"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--benchmark-deployment", type=Path, required=True)
    parser.add_argument("--implemented-deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
