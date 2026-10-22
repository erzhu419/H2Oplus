#!/usr/bin/env python3
"""Freeze and independently audit formal nonzero-policy v7 collections."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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

from bus_h2o.mdp_contract_v7 import (
    DEFAULT_CONFIG_V7,
    policy_action_to_hold_seconds_v7,
)
from bus_h2o.sumo_provenance import sha256
from collect_policy.build_transit_v7_dataset import load_shard
from reproducibility.compose_collection_manifests_v7 import (
    CANONICAL_POLICY_ORDER,
    validate_and_select_manifest,
)
from reproducibility.freeze_zero_collection_matrix_v7 import (
    _atomic_create,
    _validate_registered_file_hashes,
    _validate_scenario_files,
)
from reproducibility.run_collection_matrix_v7 import POLICIES, SHARD_SCHEMA


AUDIT_SCHEMA = "transit-smdp-v7-frozen-behavior-matrix-v1"
EXPECTED_POLICIES = tuple(policy["policy_name"] for policy in POLICIES[1:])
FATAL_LOG_MARKERS = (
    "Traceback (most recent call last):",
    "FatalTraCIError",
    "RuntimeError:",
    "ValueError:",
    "AssertionError:",
    "Connection closed by SUMO",
)


def validate_formal_behavior_manifest(
    manifest: dict[str, Any], *, block_ids: set[int] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = validate_and_select_manifest(manifest, block_ids=block_ids)
    selected_names = tuple(
        selected["behavior_mixture"]["selected_policy_names"]
    )
    if selected_names != EXPECTED_POLICIES:
        raise ValueError("behavior freeze requires exactly the seven nonzero policies")
    for job in selected["jobs"]:
        if (
            job.get("split") != "train"
            or bool(job.get("diagnostic"))
            or int(job.get("episode_seconds", -1))
            != int(DEFAULT_CONFIG_V7.horizon_seconds)
            or int(job.get("max_events", -1)) != 0
        ):
            raise ValueError("behavior job violates the formal full-horizon contract")
    realization_ids = sorted(
        {int(job["realization_block_id"]) for job in selected["jobs"]}
    )
    block_ids_seen = sorted({int(job["block_id"]) for job in selected["jobs"]})
    return selected, {
        "policies": list(selected_names),
        "jobs": len(selected["jobs"]),
        "environment_cells": int(selected["environment_cells"]),
        "independent_realization_blocks": int(
            selected["independent_realization_blocks"]
        ),
        "realization_block_ids": realization_ids,
        "environment_cell_block_ids": block_ids_seen,
        "demand_scales": [
            float(value)
            for value in selected["environment_design"]["demand_scales"]
        ],
    }


def _validate_exact_behavior_artifact_set(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    outputs = [Path(job["output"]).expanduser().resolve() for job in manifest["jobs"]]
    output_root = Path(os.path.commonpath([str(path) for path in outputs]))
    expected_by_policy: dict[str, set[Path]] = defaultdict(set)
    for job, output in zip(manifest["jobs"], outputs):
        policy_name = job["policy"]["policy_name"]
        expected_by_policy[policy_name].update(
            {
                output,
                output.with_suffix(output.suffix + ".json"),
                output.with_suffix(output.suffix + ".log"),
            }
        )
    missing: list[str] = []
    extras: list[str] = []
    observed = 0
    for policy_name, expected in expected_by_policy.items():
        policy_roots = {
            next(
                parent
                for parent in output.parents
                if parent.name == policy_name and output_root in parent.parents
            )
            for output in outputs
            if policy_name in output.parts
        }
        if len(policy_roots) != 1:
            raise ValueError(f"behavior policy {policy_name} has an invalid output root")
        policy_root = next(iter(policy_roots))
        actual = {path.resolve() for path in policy_root.rglob("*") if path.is_file()}
        observed += len(actual)
        missing.extend(str(path) for path in sorted(expected - actual))
        extras.extend(str(path) for path in sorted(actual - expected))
    if missing or extras:
        raise ValueError(
            "formal behavior artifact set is not closed: "
            f"missing={missing}, extras={extras}"
        )
    return {
        "root": str(output_root),
        "registered_files": sum(len(paths) for paths in expected_by_policy.values()),
        "observed_files": observed,
        "exact_match_at_freeze": True,
    }


def _action_summary(holds: np.ndarray) -> dict[str, float]:
    return {
        "exact_zero_rate": float(np.mean(holds == 0.0)),
        "mean_hold_seconds": float(np.mean(holds)),
        "p50_hold_seconds": float(np.percentile(holds, 50)),
        "p95_hold_seconds": float(np.percentile(holds, 95)),
        "max_hold_seconds": float(np.max(holds)),
    }


def _audit_behavior_shard(
    job: dict[str, Any], manifest: dict[str, Any], cache: dict[Path, str]
) -> dict[str, Any]:
    shard = Path(job["output"]).expanduser().resolve()
    summary_path = shard.with_suffix(shard.suffix + ".json")
    log_path = shard.with_suffix(shard.suffix + ".log")
    if not shard.is_file() or not summary_path.is_file() or not log_path.is_file():
        raise ValueError(f"behavior job is missing a shard, summary, or log: {shard}")
    log_text = log_path.read_text(errors="replace")
    fatal_markers = [marker for marker in FATAL_LOG_MARKERS if marker in log_text]
    if fatal_markers:
        raise ValueError(
            f"behavior log contains fatal markers {fatal_markers}: {log_path}"
        )
    warnings = [
        line.strip()
        for line in log_text.splitlines()
        if line.lstrip().startswith("Warning:")
    ]
    _validate_scenario_files(job, cache)
    summary, arrays = load_shard(shard, allow_partial=False)
    policy_name = job["policy"]["policy_name"]
    expected = {
        "schema_version": SHARD_SCHEMA,
        "job_signature": job["signature"],
        "policy_name": policy_name,
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
    scenario = summary.get("scenario", {})
    if scenario.get("manifest_sha256") != job["scenario"]["sha256"]:
        mismatches["scenario_manifest_sha256"] = "matrix disagreement"
    if scenario.get("runtime_input_sha256") != job["scenario"]["runtime_input_sha256"]:
        mismatches["scenario_runtime_input_sha256"] = "matrix disagreement"
    if summary.get("runtime", {}).get("sumo_version") != manifest.get("sumo_version"):
        mismatches["sumo_version"] = "matrix disagreement"
    if summary.get("bus_service_semantics_audit", {}).get("passed") is not True:
        mismatches["bus_service_semantics_audit"] = "did not pass"
    if summary.get("full_service_trace") is not None:
        mismatches["full_service_trace"] = "nonzero policies must not emit traces"
    if mismatches:
        raise ValueError(f"behavior shard provenance mismatch: {shard}: {mismatches}")

    service = summary["service"]
    if (
        service["departed_trips"] != service["completed_trips"]
        or int(service.get("active_trips_at_horizon", -1)) != 0
        or float(service.get("completion_rate", -1.0)) != 1.0
    ):
        raise ValueError(f"behavior shard has incomplete bus service: {shard}")
    holds = np.asarray(
        policy_action_to_hold_seconds_v7(arrays["actions"][:, 0]), np.float64
    )
    reconstructed_actions = _action_summary(holds)
    reported_actions = summary.get("actions", {})
    if set(reported_actions) != set(reconstructed_actions) or any(
        not np.isclose(
            float(reported_actions[name]), value, rtol=1e-7, atol=1e-5
        )
        for name, value in reconstructed_actions.items()
    ):
        raise ValueError(f"behavior action summary does not reconstruct: {shard}")

    first_decision = float(summary["training_reward"]["first_decision_time_seconds"])
    observed_end = float(summary["episode"]["observed_end_time_seconds"])
    elapsed = float(np.sum(arrays["delta_t_seconds"], dtype=np.float64))
    reward_sum = float(np.sum(arrays["rewards"], dtype=np.float64))
    if abs(elapsed - (observed_end - first_decision)) > 1e-9:
        raise ValueError(f"behavior shard duration does not close its horizon: {shard}")
    if abs(reward_sum - float(summary["training_reward"]["reward_sum"])) > 1e-9:
        raise ValueError(f"behavior shard reward sum disagrees with summary: {shard}")
    return {
        "block_id": int(job["block_id"]),
        "realization_block_id": int(job["realization_block_id"]),
        "policy_name": policy_name,
        "demand_scale": float(job["scenario"]["demand_scale"]),
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
            "warning_lines": warnings,
            "fatal_markers_seen": fatal_markers,
        },
        "transitions": int(len(arrays["rewards"])),
        "zero_duration_transitions": int(np.sum(arrays["delta_t_seconds"] == 0.0)),
        "positive_duration_seconds": elapsed,
        "reward_sum_passenger_seconds": reward_sum,
        "actions": reconstructed_actions,
        "minimum_hold_seconds": float(np.min(holds)),
        "departed_trips": int(service["departed_trips"]),
        "completed_trips": int(service["completed_trips"]),
    }


def freeze_and_audit(
    source: Path,
    output: Path,
    audit_output: Path,
    *,
    block_ids: set[int] | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    audit_output = audit_output.expanduser().resolve()
    if len({source, output, audit_output}) != 3:
        raise ValueError("source, frozen manifest, and audit paths must be distinct")
    if output.exists() or audit_output.exists():
        raise FileExistsError("frozen behavior artifacts are immutable")
    lock_path = source.parent / ".collection.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("cannot freeze a behavior matrix during collection") from error
        source_bytes = source.read_bytes()
        source_hash = sha256(source)
        source_manifest = json.loads(source_bytes)
        manifest, topology = validate_formal_behavior_manifest(
            source_manifest, block_ids=block_ids
        )
        verified_hashes: dict[Path, str] = {}
        _validate_registered_file_hashes(source_manifest, verified_hashes)
        artifact_set = _validate_exact_behavior_artifact_set(manifest)
        shard_reports = [
            _audit_behavior_shard(job, manifest, verified_hashes)
            for job in sorted(
                manifest["jobs"],
                key=lambda value: (
                    CANONICAL_POLICY_ORDER.index(value["policy"]["policy_name"]),
                    int(value["block_id"]),
                ),
            )
        ]
        if source.read_bytes() != source_bytes or sha256(source) != source_hash:
            raise RuntimeError("active behavior matrix changed during the freeze audit")

        frozen_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
        aggregate: dict[str, dict[str, Any]] = {}
        for policy_name in EXPECTED_POLICIES:
            selected_rows = [
                row for row in shard_reports if row["policy_name"] == policy_name
            ]
            aggregate[policy_name] = {
                "shards": len(selected_rows),
                "transitions": sum(row["transitions"] for row in selected_rows),
                "zero_duration_transitions": sum(
                    row["zero_duration_transitions"] for row in selected_rows
                ),
                "reward_sum_passenger_seconds": sum(
                    row["reward_sum_passenger_seconds"] for row in selected_rows
                ),
                "minimum_hold_seconds": min(
                    row["minimum_hold_seconds"]
                    for row in selected_rows
                ),
                "maximum_hold_seconds": max(
                    row["actions"]["max_hold_seconds"] for row in selected_rows
                ),
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
            "frozen_manifest": {"path": str(output), "sha256": frozen_hash},
            "topology": topology,
            "artifact_set": artifact_set,
            "registered_input_files_verified": len(verified_hashes),
            "sumo_version": manifest["sumo_version"],
            "collector_sha256": manifest["collector_sha256"],
            "runner_sha256": manifest["runner_sha256"],
            "source_sha256": manifest["source_sha256"],
            "aggregate_by_policy": aggregate,
            "shards": shard_reports,
        }
        audit_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        _atomic_create(output, frozen_bytes)
        try:
            _atomic_create(audit_output, audit_bytes)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        if sha256(output) != frozen_hash:
            raise RuntimeError("frozen behavior manifest bytes changed during creation")
        fcntl.flock(lock, fcntl.LOCK_UN)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--block-ids", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = freeze_and_audit(
        args.source,
        args.output,
        args.audit_output,
        block_ids=set(args.block_ids) if args.block_ids is not None else None,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "frozen_manifest": report["frozen_manifest"],
                "audit_output": str(args.audit_output.expanduser().resolve()),
                "shards": len(report["shards"]),
                "topology": report["topology"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
