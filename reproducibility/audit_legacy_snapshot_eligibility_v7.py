#!/usr/bin/env python3
"""Reject legacy snapshot archives that cannot satisfy the v7 reset contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
from bus_h2o.envs.bus_sim_env import MultiLineSimEnv
from bus_h2o.mdp_contract_v7 import ACTION_DIM, CANONICAL_STATE_DIM
from bus_h2o.sim_core.sim import env_bus


SCHEMA_VERSION = "transit-legacy-snapshot-eligibility-audit-v1"
REQUIRED_RESET_SEMANTICS = (
    "all directional services are restored",
    "active passenger identities and destinations are restored",
    "transfer stages and remaining itinerary legs are restored",
    "waiting and onboard memberships reconcile with passenger identities",
    "the passenger-time ledger and its clock are restored",
    "timetable launch state and vehicle-trip identity are restored",
    "the demand realization and all stochastic generator states are restored",
    "the first post-reset v7 observation and reward are reproducible",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
    }
    if include_sha256:
        result["sha256"] = sha256(path)
    return result


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    if size <= 0:
        raise ValueError("snapshot source is empty")
    count = min(max(int(count), 1), size)
    return sorted(set(np.linspace(0, size - 1, count, dtype=np.int64).tolist()))


def inspect_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {"mapping": False}
    buses = snapshot.get("all_buses", [])
    stations = snapshot.get("all_stations", [])
    bus_keys = sorted(
        set().union(*(row.keys() for row in buses if isinstance(row, dict)))
    )
    station_keys = sorted(
        set().union(*(row.keys() for row in stations if isinstance(row, dict)))
    )
    return {
        "mapping": True,
        "top_level_keys": sorted(snapshot),
        "bus_keys": bus_keys,
        "station_keys": station_keys,
        "active_bus_records": len(buses),
        "station_records": len(stations),
        "has_passenger_identity_records": any(
            key in snapshot
            for key in (
                "passengers",
                "journeys",
                "waiting_passengers",
                "onboard_passengers",
            )
        ),
        "has_passenger_time_ledger": any(
            key in snapshot
            for key in (
                "passenger_time_ledger",
                "passenger_time_snapshot",
                "accrued_wait_seconds",
                "accrued_in_vehicle_seconds",
            )
        ),
        "has_rng_state": any(
            key in snapshot
            for key in ("rng_state", "random_state", "numpy_rng_state")
        ),
        "has_demand_realization": any(
            key in snapshot
            for key in (
                "demand_realization_seed",
                "scenario_manifest_sha256",
                "journey_demand",
            )
        ),
    }


def audit_archive(archive_dir: Path, samples_per_file: int) -> dict[str, Any]:
    manifest_path = archive_dir / "file_manifest.json"
    merged_path = archive_dir / "merged_all_v2.h5"
    if not manifest_path.is_file() or not merged_path.is_file():
        raise FileNotFoundError("legacy manifest or merged replay is missing")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("legacy snapshot manifest is empty or invalid")

    with h5py.File(merged_path, "r") as merged:
        merged_contract = {
            "observations": list(merged["observations"].shape),
            "actions": list(merged["actions"].shape),
            "attrs": {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in merged.attrs.items()
            },
        }
        action_sample = np.asarray(
            merged["actions"][evenly_spaced_indices(len(merged["actions"]), 4096)],
            dtype=np.float64,
        )

    sampled: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    missing_reset_snapshot_files: list[str] = []
    for filename, expected_rows in manifest:
        path = archive_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"snapshot source is missing: {path}")
        with h5py.File(path, "r") as handle:
            source_descriptor = descriptor(path, include_sha256=False)
            source_descriptor["datasets"] = sorted(handle)
            source_descriptor["expected_rows"] = int(expected_rows)
            source_files.append(source_descriptor)
            if "snapshot_T1" not in handle:
                missing_reset_snapshot_files.append(filename)
                continue
            dataset = handle["snapshot_T1"]
            if len(dataset) != int(expected_rows):
                raise ValueError(f"snapshot row count drifted in {path}")
            for index in evenly_spaced_indices(len(dataset), samples_per_file):
                snapshot = pickle.loads(bytes(dataset[index]))
                sampled.append(
                    {
                        "file": filename,
                        "row": index,
                        "snapshot": inspect_snapshot(snapshot),
                    }
                )

    all_mappings = all(row["snapshot"].get("mapping") for row in sampled)
    missing_identity = all(
        not row["snapshot"].get("has_passenger_identity_records", False)
        for row in sampled
    )
    missing_ledger = all(
        not row["snapshot"].get("has_passenger_time_ledger", False)
        for row in sampled
    )
    missing_rng = all(
        not row["snapshot"].get("has_rng_state", False) for row in sampled
    )
    missing_realization = all(
        not row["snapshot"].get("has_demand_realization", False)
        for row in sampled
    )
    second_action_nonconstant = bool(
        action_sample.shape[1] > 1
        and np.ptp(action_sample[:, 1]) > np.finfo(np.float32).eps
    )

    restore_source = inspect.getsource(env_bus.restore_full_system_snapshot)
    legacy_multiline_reset = inspect.getsource(MultiLineSimEnv.reset)
    canonical_reset = inspect.getsource(AllLinesSimEnv.reset)
    implementation_checks = {
        "legacy_restore_pads_unknown_passengers": (
            "np.array([None]" in restore_source
            and "[None] * count" in restore_source
        ),
        "legacy_multiline_reset_targets_only_7x": (
            "self._x7.restore_full_system_snapshot" in legacy_multiline_reset
        ),
        "canonical_v7_rejects_snapshot_reset": (
            "snapshot reset is intentionally excluded" in canonical_reset
        ),
    }

    blockers = [
        {
            "id": "manifest-sources-do-not-all-contain-reset-snapshots",
            "evidence": {
                "missing_snapshot_T1_files": missing_reset_snapshot_files,
                "missing_file_count": len(missing_reset_snapshot_files),
            },
        },
        {
            "id": "contract-dimension-mismatch",
            "evidence": {
                "legacy_observation_dim": merged_contract["observations"][1],
                "legacy_action_dim": merged_contract["actions"][1],
                "v7_observation_dim": CANONICAL_STATE_DIM,
                "v7_action_dim": ACTION_DIM,
            },
        },
        {
            "id": "legacy-speed-control-present",
            "evidence": {
                "second_action_component_nonconstant_in_sample": (
                    second_action_nonconstant
                ),
                "v7_speed_ratio": 1.0,
            },
        },
        {
            "id": "passenger-identities-and-itineraries-missing",
            "evidence": {"all_sampled_snapshots_missing": missing_identity},
        },
        {
            "id": "passenger-time-ledger-missing",
            "evidence": {"all_sampled_snapshots_missing": missing_ledger},
        },
        {
            "id": "stochastic-state-and-demand-realization-missing",
            "evidence": {
                "all_sampled_snapshots_missing_rng": missing_rng,
                "all_sampled_snapshots_missing_realization": missing_realization,
            },
        },
        {
            "id": "legacy-restore-fabricates-count-only-passengers",
            "evidence": implementation_checks,
        },
        {
            "id": "legacy-multiline-restore-is-7x-only",
            "evidence": implementation_checks,
        },
        {
            "id": "no-v7-reset-replay-equivalence-audit",
            "evidence": {
                "required_semantics": list(REQUIRED_RESET_SEMANTICS),
            },
        },
    ]
    return {
        "manifest": descriptor(manifest_path),
        "merged_replay": descriptor(merged_path),
        "source_file_count": len(source_files),
        "missing_reset_snapshot_files": missing_reset_snapshot_files,
        "source_bytes": sum(row["bytes"] for row in source_files),
        "source_files": source_files,
        "merged_contract": merged_contract,
        "sampled_snapshot_count": len(sampled),
        "all_sampled_rows_are_mappings": all_mappings,
        "sampled_snapshots": sampled,
        "implementation_sources": {
            "env_bus_restore": descriptor(
                Path(inspect.getsourcefile(env_bus.restore_full_system_snapshot))
            ),
            "legacy_multiline_reset": descriptor(
                Path(inspect.getsourcefile(MultiLineSimEnv.reset))
            ),
            "canonical_all_lines_reset": descriptor(
                Path(inspect.getsourcefile(AllLinesSimEnv.reset))
            ),
        },
        "implementation_checks": implementation_checks,
        "blockers": blockers,
        "passed": False,
        "decision": (
            "legacy snapshots are retained as historical artifacts but are "
            "ineligible for v7 training, reset initialization, evaluation, "
            "or manuscript evidence"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "bus_h2o" / "datasets_v2",
    )
    parser.add_argument("--samples-per-file", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "legacy_snapshot_eligibility_audit_v7.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_file <= 0:
        raise ValueError("samples-per-file must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal-eligibility-gate",
        "archive": audit_archive(
            args.archive_dir.expanduser().resolve(), args.samples_per_file
        ),
        "auditor": descriptor(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "passed": False}, indent=2))


if __name__ == "__main__":
    main()
