from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from bus_h2o.mdp_contract_v4 import (
    DEFAULT_CONFIG_V4,
    RAW_EVENT_STATE_NAMES,
    continuous_time_discount_v4,
    transition_duration_seconds_v4,
    transition_reward_v4,
)
from collect_policy.build_transit_v4_dataset import (
    build_dataset,
    validate_collection_matrix,
)
from reproducibility.audit_transit_v4_dataset import audit


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(station, time, line=0):
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time,
        "forward_headway_seconds": 300 + 10 * station,
        "backward_headway_seconds": 300 - 10 * station,
        "target_forward_headway_seconds": 300,
        "target_backward_headway_seconds": 300,
        "eligible_stranded_passengers": 0,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 8,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def _write_shard(
    root: Path,
    name: str,
    policy: str,
    seed: int,
    *,
    source_headway_jitter: float | None = None,
):
    path = root / f"{name}.npz"
    observations = np.stack([_event(0, 10), _event(1, 70)])
    next_observations = np.stack([_event(1, 70), _event(2, 130)])
    if source_headway_jitter is not None:
        next_observations[0, RAW_EVENT_STATE_NAMES.index("forward_headway_seconds")] = (
            source_headway_jitter
        )
    actions = np.asarray([[-1.0], [0.0]], np.float32)
    duration = transition_duration_seconds_v4(observations, next_observations)
    rewards = transition_reward_v4(
        observations,
        actions,
        next_observations,
        correct_measurement_jitter=False,
    )
    discounts = continuous_time_discount_v4(duration)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        discounts=discounts,
        terminals=np.zeros(2, np.uint8),
        delta_t_seconds=duration,
        trajectory_local_id=np.zeros(2, np.int64),
    )
    summary = {
        "schema_version": "transit-smdp-v4-shard-v1",
        "contract_schema": "transit-smdp-v4",
        "contract": DEFAULT_CONFIG_V4.to_dict(),
        "raw_state_names": list(RAW_EVENT_STATE_NAMES),
        "shard_sha256": _hash(path),
        "split": "train",
        "policy_name": policy,
        "policy_seed": seed,
        "sumo_seed": seed + 100,
        "job_signature": name,
        "scenario": {
            "manifest_sha256": "scenario-a",
            "demand_scale": 1.0,
        },
        "episode": {
            "max_events": 2,
            "configured_seconds": 24_000,
            "observed_end_time_seconds": 130,
            "transitions": 2,
        },
    }
    path.with_suffix(".npz.json").write_text(json.dumps(summary))
    return path


def test_v4_builder_preserves_raw_audit_fields_and_cost_identity(tmp_path):
    first = _write_shard(
        tmp_path, "a", "zero", 1, source_headway_jitter=-0.5
    )
    second = _write_shard(tmp_path, "b", "random", 2)
    output = tmp_path / "dataset.h5"
    summary = build_dataset([first, second], output, allow_partial=True)
    assert summary["rows"] == 4
    assert summary["episodes"] == 2
    assert summary["exact_zero_action_rate"] == 0.5
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == "transit-smdp-v4"
        assert handle["observations"].shape == (4, 13)
        assert handle["raw_observations"].shape == (4, 14)
        np.testing.assert_array_equal(
            handle["rewards"][:],
            -(handle["waiting_cost"][:] + handle["holding_cost"][:]),
        )
        assert len(np.unique(handle["episode_id"][:])) == 2
        assert len(np.unique(handle["trajectory_id"][:])) == 2
        assert np.min(handle["raw_next_observations"][:, 3]) == 0.0
        provenance = json.loads(handle.attrs["build_provenance"])
        assert provenance["measurement_resolution_correction"][
            "total_corrected_values"
        ] == 1
        assert handle.attrs["content_signature"] == summary["content_signature"]
    report = audit(output)
    assert report["passed"] is True
    assert report["logical_content_signature_matches"] is True
    assert report["reconstruction_max_absolute_error"] == {
        "observations": 0.0,
        "next_observations": 0.0,
        "waiting_cost": 0.0,
        "holding_cost": 0.0,
        "reward_identity": 0.0,
        "reward_from_raw": 0.0,
        "delta_t_seconds": 0.0,
        "discounts": 0.0,
    }


def test_v4_builder_rejects_partial_shards_for_formal_data(tmp_path):
    shard = _write_shard(tmp_path, "partial", "zero", 1)
    with pytest.raises(ValueError, match="partial event-limited"):
        build_dataset([shard], tmp_path / "dataset.h5", allow_partial=False)


def test_collection_matrix_requires_exact_provenance_locked_shards(tmp_path):
    first = _write_shard(tmp_path, "a", "zero", 1)
    second = _write_shard(tmp_path, "b", "random", 2)
    source_hashes = {"collector.py": "source-hash"}
    runtime_hashes = {"scenario.sumocfg": "runtime-hash"}
    sumo_version = "SUMO test-version"
    jobs = []
    for path, name, policy, seed in (
        (first, "a", "zero", 1),
        (second, "b", "random", 2),
    ):
        summary_path = path.with_suffix(".npz.json")
        summary = json.loads(summary_path.read_text())
        summary["source_sha256"] = source_hashes
        summary["runtime"] = {"sumo_version": sumo_version}
        summary["scenario"]["runtime_input_sha256"] = runtime_hashes
        summary_path.write_text(json.dumps(summary))
        jobs.append(
            {
                "output": str(path.resolve()),
                "signature": name,
                "policy": {"policy_name": policy},
                "policy_seed": seed,
                "sumo_seed": seed + 100,
                "scenario": {
                    "sha256": "scenario-a",
                    "runtime_input_sha256": runtime_hashes,
                },
            }
        )
    manifest = tmp_path / "matrix_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "transit-smdp-v4-collection-matrix-v1",
                "collector_sha256": "collector-hash",
                "runner_sha256": "runner-hash",
                "source_sha256": source_hashes,
                "sumo_version": sumo_version,
                "jobs": jobs,
            }
        )
    )

    provenance = validate_collection_matrix(manifest, [first, second])
    assert provenance["jobs"] == 2
    assert provenance["sumo_version"] == sumo_version
    with pytest.raises(ValueError, match="missing="):
        validate_collection_matrix(manifest, [first])
