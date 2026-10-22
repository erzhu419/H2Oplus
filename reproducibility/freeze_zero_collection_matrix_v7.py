#!/usr/bin/env python3
"""Freeze and independently audit a completed formal v7 zero-hold matrix."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
from bus_h2o.full_service_trace_v7 import (
    FULL_SERVICE_TRACE_SCHEMA,
    full_service_trace_path,
    load_full_service_trace,
)
from bus_h2o.sumo_provenance import sha256
from collect_policy.build_transit_v7_dataset import load_shard
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
    FORMAL_FIDELITY_REALIZATION_BLOCKS,
)
from reproducibility.run_collection_matrix_v7 import MATRIX_SCHEMA, SHARD_SCHEMA


AUDIT_SCHEMA = "transit-smdp-v7-frozen-zero-matrix-v2"
EXPECTED_DEMAND_SCALES = (0.75, 1.0, 1.25)
EXPECTED_REALIZATION_BLOCKS = tuple(range(10))
EXPECTED_ENVIRONMENT_CELLS = len(EXPECTED_DEMAND_SCALES) * len(
    EXPECTED_REALIZATION_BLOCKS
)


def validate_formal_zero_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the registered 10-by-3 formal zero-hold matrix topology."""

    if manifest.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("zero-hold freeze requires a v7 collection matrix v2")
    if manifest.get("study_stage") != "formal":
        raise ValueError("zero-hold freeze requires a formal collection matrix")
    if int(manifest.get("independent_realization_blocks", -1)) != len(
        EXPECTED_REALIZATION_BLOCKS
    ):
        raise ValueError("formal zero-hold matrix must contain ten realization blocks")
    if int(manifest.get("environment_cells", -1)) != EXPECTED_ENVIRONMENT_CELLS:
        raise ValueError("formal zero-hold matrix must contain 30 environment cells")

    design = manifest.get("environment_design")
    if not isinstance(design, dict):
        raise ValueError("formal zero-hold matrix lacks its environment design")
    declared_scales = tuple(float(value) for value in design.get("demand_scales", ()))
    if declared_scales != EXPECTED_DEMAND_SCALES:
        raise ValueError("formal zero-hold matrix has unregistered demand scales")
    if not design.get("evaluation_disjointness", {}).get("verified_disjoint"):
        raise ValueError("training/evaluation realization disjointness is not verified")

    mixture = manifest.get("behavior_mixture")
    if not isinstance(mixture, dict) or mixture.get("selected_policy_names") != ["zero"]:
        raise ValueError("frozen calibration matrix must select only the zero policy")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_ENVIRONMENT_CELLS:
        raise ValueError("formal zero-hold matrix must declare exactly 30 jobs")

    outputs: set[str] = set()
    signatures: set[str] = set()
    scenario_hashes: set[str] = set()
    block_ids: set[int] = set()
    cells: set[tuple[int, float]] = set()
    realization_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        policy = job.get("policy", {})
        if policy.get("policy") != "zero" or policy.get("policy_name") != "zero":
            raise ValueError("frozen calibration matrix contains a nonzero policy")
        if (
            job.get("split") != "train"
            or bool(job.get("diagnostic"))
            or int(job.get("episode_seconds", -1))
            != int(DEFAULT_CONFIG_V7.horizon_seconds)
            or int(job.get("max_events", -1)) != 0
        ):
            raise ValueError("formal zero-hold job violates the full-horizon contract")
        if job.get("source_sha256") != manifest.get("source_sha256"):
            raise ValueError("zero-hold job source hashes disagree with the matrix")
        if job.get("sumo_version") != manifest.get("sumo_version"):
            raise ValueError("zero-hold job SUMO version disagrees with the matrix")

        output = str(Path(job["output"]).expanduser().resolve())
        signature = str(job.get("signature", ""))
        scenario = job.get("scenario", {})
        scenario_hash = str(scenario.get("sha256", ""))
        block_id = int(job["block_id"])
        realization_id = int(job["realization_block_id"])
        demand_scale = float(scenario["demand_scale"])
        if output in outputs or signature in signatures or scenario_hash in scenario_hashes:
            raise ValueError("formal zero-hold jobs reuse an output, signature, or scenario")
        outputs.add(output)
        signatures.add(signature)
        scenario_hashes.add(scenario_hash)
        block_ids.add(block_id)
        cell = (realization_id, demand_scale)
        if cell in cells:
            raise ValueError("formal zero-hold matrix duplicates an environment cell")
        cells.add(cell)
        realization_rows[realization_id].append(job)

    if block_ids != set(range(EXPECTED_ENVIRONMENT_CELLS)):
        raise ValueError("formal zero-hold block identifiers are not exactly 0 through 29")
    expected_cells = {
        (block, scale)
        for block in EXPECTED_REALIZATION_BLOCKS
        for scale in EXPECTED_DEMAND_SCALES
    }
    if cells != expected_cells:
        raise ValueError("formal zero-hold matrix does not cover the registered 10-by-3 grid")

    demand_seeds: set[int] = set()
    sumo_seeds: set[int] = set()
    for realization_id in EXPECTED_REALIZATION_BLOCKS:
        rows = realization_rows[realization_id]
        realization_demand_seeds = {
            int(row["scenario"]["selection_seed"]) for row in rows
        }
        realization_sumo_seeds = {int(row["sumo_seed"]) for row in rows}
        if len(realization_demand_seeds) != 1 or len(realization_sumo_seeds) != 1:
            raise ValueError("a realization block mixes demand or SUMO seeds")
        demand_seed = next(iter(realization_demand_seeds))
        sumo_seed = next(iter(realization_sumo_seeds))
        if demand_seed in demand_seeds or sumo_seed in sumo_seeds:
            raise ValueError("independent realization blocks reuse demand or SUMO seeds")
        demand_seeds.add(demand_seed)
        sumo_seeds.add(sumo_seed)

    calibration = set(FORMAL_CALIBRATION_REALIZATION_BLOCKS)
    fidelity = set(FORMAL_FIDELITY_REALIZATION_BLOCKS)
    if calibration & fidelity or calibration | fidelity != set(
        EXPECTED_REALIZATION_BLOCKS
    ):
        raise ValueError("registered runtime-fit and fidelity partitions are invalid")
    return {
        "environment_cells": len(cells),
        "independent_realization_blocks": len(realization_rows),
        "demand_scales": list(EXPECTED_DEMAND_SCALES),
        "demand_realization_seeds": sorted(demand_seeds),
        "sumo_seeds": sorted(sumo_seeds),
        "calibration_realization_blocks": sorted(calibration),
        "reserved_fidelity_realization_blocks": sorted(fidelity),
    }


def _require_registered_hash(
    path: Path, expected: str, cache: dict[Path, str], *, label: str
) -> None:
    path = path.expanduser().resolve()
    if path not in cache:
        if not path.is_file():
            raise ValueError(f"registered {label} is missing: {path}")
        cache[path] = sha256(path)
    if cache[path] != expected:
        raise ValueError(f"registered {label} changed: {path}")


def _validate_registered_file_hashes(
    manifest: dict[str, Any], cache: dict[Path, str]
) -> None:
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("zero-hold matrix contains no registered source hashes")
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path).expanduser().resolve()
        _require_registered_hash(path, expected, cache, label="collection source")

    direct_files = {
        Path(manifest["collector"]).expanduser().resolve(): manifest[
            "collector_sha256"
        ],
        (ROOT / "reproducibility" / "run_collection_matrix_v7.py").resolve(): manifest[
            "runner_sha256"
        ],
    }
    design = manifest["environment_design"]
    design_path = None
    if "path" in design:
        design_path = Path(design["path"]).expanduser().resolve()
        direct_files[design_path] = design["sha256"]
    for path, expected in direct_files.items():
        _require_registered_hash(path, expected, cache, label="matrix input")

    if design_path is None:
        raise ValueError("formal zero-hold matrix lacks a registered design path")
    design_payload = json.loads(design_path.read_text())
    generator = Path(design_payload.get("generator", "")).expanduser().resolve()
    _require_registered_hash(
        generator,
        design_payload.get("generator_sha256", ""),
        cache,
        label="training-design generator",
    )
    _require_registered_hash(
        ROOT / "reproducibility" / "build_passenger_scenarios.py",
        design_payload.get("scenario_generator_sha256", ""),
        cache,
        label="passenger-scenario generator",
    )
    for descriptor in design_payload.get("source_sha256", {}).values():
        _require_registered_hash(
            Path(descriptor.get("path", "")),
            descriptor.get("sha256", ""),
            cache,
            label="training-design source",
        )
    for name in ("evaluation_disjointness", "physical_scenario_reuse"):
        descriptor = design_payload.get(name, {})
        _require_registered_hash(
            Path(descriptor.get("path", "")),
            descriptor.get("sha256", ""),
            cache,
            label=name.replace("_", " "),
        )


def _validate_exact_zero_artifact_set(manifest: dict[str, Any]) -> dict[str, Any]:
    outputs = [Path(job["output"]).expanduser().resolve() for job in manifest["jobs"]]
    zero_root = Path(os.path.commonpath([str(path) for path in outputs]))
    if zero_root.name != "zero" or any(zero_root not in path.parents for path in outputs):
        raise ValueError("formal zero-hold outputs do not share the registered zero root")
    expected = set()
    for output in outputs:
        trace = full_service_trace_path(output)
        expected.update(
            {
                output,
                output.with_suffix(output.suffix + ".json"),
                output.with_suffix(output.suffix + ".log"),
                trace,
                trace.with_suffix(trace.suffix + ".json"),
            }
        )
    actual = {path.resolve() for path in zero_root.rglob("*") if path.is_file()}
    missing = sorted(str(path) for path in expected - actual)
    extras = sorted(str(path) for path in actual - expected)
    if missing or extras:
        raise ValueError(
            "formal zero-hold artifact set is not closed: "
            f"missing={missing}, extras={extras}"
        )
    return {
        "root": str(zero_root),
        "registered_files": len(expected),
        "observed_files": len(actual),
        "exact_match": True,
    }


def _validate_scenario_files(
    job: dict[str, Any], cache: dict[Path, str]
) -> None:
    scenario = job["scenario"]
    manifest_path = Path(scenario["path"]).expanduser().resolve()
    _require_registered_hash(
        manifest_path,
        scenario["sha256"],
        cache,
        label="passenger-scenario manifest",
    )
    scenario_payload = json.loads(manifest_path.read_text())
    _require_registered_hash(
        ROOT / "reproducibility" / "build_passenger_scenarios.py",
        scenario_payload.get("generator_sha256", ""),
        cache,
        label="passenger-scenario generator",
    )
    for field in ("source_sha256", "scenario_sha256"):
        for raw_path, expected in scenario_payload.get(field, {}).items():
            _require_registered_hash(
                Path(raw_path), expected, cache, label=f"passenger scenario {field}"
            )
    for raw_path, expected in scenario.get("runtime_input_sha256", {}).items():
        path = Path(raw_path).expanduser().resolve()
        _require_registered_hash(path, expected, cache, label="SUMO runtime input")


def _audit_shard(
    job: dict[str, Any], manifest: dict[str, Any], cache: dict[Path, str]
) -> dict[str, Any]:
    shard = Path(job["output"]).expanduser().resolve()
    summary_path = shard.with_suffix(shard.suffix + ".json")
    log_path = shard.with_suffix(shard.suffix + ".log")
    if not shard.is_file() or not summary_path.is_file() or not log_path.is_file():
        raise ValueError(f"zero-hold job is missing a shard, summary, or log: {shard}")
    log_text = log_path.read_text(errors="replace")
    fatal_log_markers = (
        "Traceback (most recent call last):",
        "FatalTraCIError",
        "RuntimeError:",
        "ValueError:",
        "AssertionError:",
        "Connection closed by SUMO",
    )
    fatal_markers_seen = [marker for marker in fatal_log_markers if marker in log_text]
    if fatal_markers_seen:
        raise ValueError(
            f"zero-hold collection log contains fatal markers {fatal_markers_seen}: "
            f"{log_path}"
        )
    warning_lines = [
        line.strip()
        for line in log_text.splitlines()
        if line.lstrip().startswith("Warning:")
    ]
    _validate_scenario_files(job, cache)
    summary, arrays = load_shard(shard, allow_partial=False)
    expected = {
        "schema_version": SHARD_SCHEMA,
        "job_signature": job["signature"],
        "policy_name": "zero",
        "policy_seed": int(job["policy_seed"]),
        "sumo_seed": int(job["sumo_seed"]),
        "split": "train",
        "diagnostic": False,
        "training_eligible": True,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if summary.get("source_sha256") != manifest.get("source_sha256"):
        mismatches["source_sha256"] = "matrix disagreement"
    if summary.get("scenario", {}).get("manifest_sha256") != job["scenario"][
        "sha256"
    ]:
        mismatches["scenario_manifest_sha256"] = "matrix disagreement"
    if summary.get("scenario", {}).get("runtime_input_sha256") != job[
        "scenario"
    ]["runtime_input_sha256"]:
        mismatches["scenario_runtime_input_sha256"] = "matrix disagreement"
    if summary.get("runtime", {}).get("sumo_version") != manifest.get(
        "sumo_version"
    ):
        mismatches["sumo_version"] = "matrix disagreement"
    if summary.get("bus_service_semantics_audit", {}).get("passed") is not True:
        mismatches["bus_service_semantics_audit"] = "did not pass"
    if mismatches:
        raise ValueError(f"zero-hold shard provenance mismatch: {shard}: {mismatches}")
    if not np.all(arrays["actions"][:, 0] == -1.0):
        raise ValueError(f"zero-hold shard contains a nonzero hold action: {shard}")
    if summary.get("actions") != {
        "exact_zero_rate": 1.0,
        "max_hold_seconds": 0.0,
        "mean_hold_seconds": 0.0,
        "p50_hold_seconds": 0.0,
        "p95_hold_seconds": 0.0,
    }:
        raise ValueError(f"zero-hold action summary is not exact: {shard}")
    service = summary["service"]
    if (
        service["departed_trips"] != service["completed_trips"]
        or int(service.get("active_trips_at_horizon", -1)) != 0
        or float(service.get("completion_rate", -1.0)) != 1.0
    ):
        raise ValueError(f"zero-hold shard has incomplete bus service: {shard}")

    trace_path = full_service_trace_path(shard)
    trace_summary_path = trace_path.with_suffix(trace_path.suffix + ".json")
    trace_summary, trace_arrays = load_full_service_trace(trace_path)
    descriptor = summary.get("full_service_trace", {})
    if (
        descriptor.get("schema_version") != FULL_SERVICE_TRACE_SCHEMA
        or Path(descriptor.get("path", "")).expanduser().resolve() != trace_path
        or descriptor.get("sha256") != sha256(trace_path)
        or descriptor.get("summary_sha256") != sha256(trace_summary_path)
        or trace_summary.get("source_shard", {}).get("sha256")
        != summary["shard_sha256"]
        or trace_summary.get("source_shard", {}).get("job_signature")
        != job["signature"]
        or trace_summary.get("collection_source_sha256")
        != manifest.get("source_sha256")
        or trace_summary.get("full_service_trace", {}).get("completed_trips")
        != service["completed_trips"]
    ):
        raise ValueError(f"zero-hold full-service trace provenance mismatch: {shard}")

    first_decision = float(summary["training_reward"]["first_decision_time_seconds"])
    observed_end = float(summary["episode"]["observed_end_time_seconds"])
    elapsed = float(np.sum(arrays["delta_t_seconds"], dtype=np.float64))
    if abs(elapsed - (observed_end - first_decision)) > 1e-9:
        raise ValueError(f"zero-hold shard duration does not close its horizon: {shard}")
    reward_sum = float(np.sum(arrays["rewards"], dtype=np.float64))
    if abs(reward_sum - float(summary["training_reward"]["reward_sum"])) > 1e-9:
        raise ValueError(f"zero-hold shard reward sum disagrees with its summary: {shard}")

    return {
        "block_id": int(job["block_id"]),
        "realization_block_id": int(job["realization_block_id"]),
        "demand_scale": float(job["scenario"]["demand_scale"]),
        "demand_realization_seed": int(job["scenario"]["selection_seed"]),
        "sumo_seed": int(job["sumo_seed"]),
        "job_signature": job["signature"],
        "shard": {
            "path": str(shard),
            "sha256": sha256(shard),
            "bytes": shard.stat().st_size,
        },
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "log": {
            "path": str(log_path),
            "sha256": sha256(log_path),
            "warning_lines": warning_lines,
            "fatal_markers_seen": fatal_markers_seen,
        },
        "transitions": int(len(arrays["rewards"])),
        "zero_duration_transitions": int(np.sum(arrays["delta_t_seconds"] == 0.0)),
        "positive_duration_seconds": elapsed,
        "reward_sum_passenger_seconds": reward_sum,
        "first_decision_time_seconds": first_decision,
        "observed_end_time_seconds": observed_end,
        "departed_trips": int(service["departed_trips"]),
        "completed_trips": int(service["completed_trips"]),
        "full_service_trace": {
            "path": str(trace_path),
            "sha256": sha256(trace_path),
            "summary_path": str(trace_summary_path),
            "summary_sha256": sha256(trace_summary_path),
            "stop_events": int(len(trace_arrays["stop_bus_code"])),
            "segments": int(len(trace_arrays["segment_bus_code"])),
        },
    }


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.partial-{os.getpid()}-{datetime.now(timezone.utc).timestamp():.6f}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_and_audit(source: Path, output: Path, audit_output: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    audit_output = audit_output.expanduser().resolve()
    if len({source, output, audit_output}) != 3:
        raise ValueError("source, frozen manifest, and audit paths must be distinct")
    if output.exists() or audit_output.exists():
        raise FileExistsError("frozen v7 artifacts are immutable and cannot be overwritten")
    lock_path = source.parent / ".collection.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("cannot freeze a v7 matrix while collection is active") from error
        source_bytes = source.read_bytes()
        source_hash = sha256(source)
        manifest = json.loads(source_bytes)
        topology = validate_formal_zero_manifest(manifest)
        verified_hashes: dict[Path, str] = {}
        _validate_registered_file_hashes(manifest, verified_hashes)
        artifact_set = _validate_exact_zero_artifact_set(manifest)
        shard_reports = [
            _audit_shard(job, manifest, verified_hashes)
            for job in sorted(manifest["jobs"], key=lambda value: value["block_id"])
        ]
        if source.read_bytes() != source_bytes or sha256(source) != source_hash:
            raise RuntimeError("active matrix changed during the freeze audit")

        aggregate_by_demand = {}
        for scale in EXPECTED_DEMAND_SCALES:
            selected = [row for row in shard_reports if row["demand_scale"] == scale]
            aggregate_by_demand[str(scale)] = {
                "shards": len(selected),
                "transitions": sum(row["transitions"] for row in selected),
                "zero_duration_transitions": sum(
                    row["zero_duration_transitions"] for row in selected
                ),
                "positive_duration_seconds": sum(
                    row["positive_duration_seconds"] for row in selected
                ),
                "reward_sum_passenger_seconds": sum(
                    row["reward_sum_passenger_seconds"] for row in selected
                ),
                "departed_trips": sum(row["departed_trips"] for row in selected),
                "completed_trips": sum(row["completed_trips"] for row in selected),
            }
        report = {
            "schema_version": AUDIT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "audit_tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "source_manifest": {"path": str(source), "sha256": source_hash},
            "frozen_manifest": {"path": str(output), "sha256": source_hash},
            "topology": topology,
            "artifact_set": artifact_set,
            "registered_input_files_verified": len(verified_hashes),
            "sumo_version": manifest["sumo_version"],
            "collector_sha256": manifest["collector_sha256"],
            "runner_sha256": manifest["runner_sha256"],
            "source_sha256": manifest["source_sha256"],
            "aggregate_by_demand_scale": aggregate_by_demand,
            "shards": shard_reports,
        }
        audit_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        _atomic_create(output, source_bytes)
        try:
            _atomic_create(audit_output, audit_bytes)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        if sha256(output) != source_hash:
            raise RuntimeError("frozen matrix bytes differ from the active source matrix")
        fcntl.flock(lock, fcntl.LOCK_UN)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = freeze_and_audit(args.source, args.output, args.audit_output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "frozen_manifest": report["frozen_manifest"],
                "audit_output": str(args.audit_output.expanduser().resolve()),
                "shards": len(report["shards"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
