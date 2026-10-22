from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from bus_h2o.mdp_contract_v7 import (
    DEFAULT_CONFIG_V7,
    RAW_EVENT_STATE_NAMES,
    RAW_STATE_INDEX,
    RAW_STATE_NAMES,
    continuous_time_discount_v7,
    pack_event_batch_v7,
    terminal_raw_state_v7,
    transition_cost_components_v7,
    transition_reward_v7,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot
from collect_policy.build_transit_v7_dataset import (
    _dataset_content_sha256_hdf5,
    build_dataset,
    shard_paths_from_manifest,
    validate_collection_matrix,
    validate_serialized_sequence,
)
from collect_policy.build_transit_v4_dataset import dataset_content_sha256
from reproducibility.audit_transit_v7_dataset import audit
from reproducibility.audit_serialized_credit_v7 import audit as audit_credit
from reproducibility.analyze_collection_matrix_v7 import clustered_paired_summary
from reproducibility.build_training_environment_design_v7 import (
    TRAINING_DESIGN_SCHEMA,
    build_design,
)
from reproducibility.run_collection_matrix_v7 import POLICIES, policy_seed


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(line: int, station: int, time_seconds: float) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 360.0,
        "target_forward_headway_seconds": 330.0,
        "target_backward_headway_seconds": 330.0,
        "eligible_stranded_passengers": 4.0,
        "onboard_passengers": 20.0,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 8.0,
        "forward_bus_present": 1.0,
        "backward_bus_present": 1.0,
        "bus_capacity": 50.0,
        "system_waiting_passengers": 100.0,
        "system_in_vehicle_passengers": 200.0,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def _write_shard(root: Path) -> Path:
    path = root / "zero" / "scenario" / "sumo_seed1.npz"
    path.parent.mkdir(parents=True)
    events = np.stack((_event(0, 2, 10), _event(3, 5, 10)))
    first = pack_event_batch_v7(events, 0)
    second = pack_event_batch_v7(
        events, 1, np.asarray([0.25, -1.0], np.float32)
    )
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(70.0, 90, 180, 7_001.0, 14_003.0)
    terminal = terminal_raw_state_v7(second, end)
    components = transition_cost_components_v7(start, end)
    rewards = np.asarray([0.0, transition_reward_v7(start, end)], np.float64)
    durations = np.asarray([0.0, 60.0], np.float64)
    terminals = np.asarray([0, 1], np.uint8)
    discounts = np.asarray(continuous_time_discount_v7(durations), np.float32)
    discounts[terminals.astype(bool)] = 0.0
    arrays = {
        "observations": np.stack((first, second)),
        "actions": np.asarray([[0.25], [-1.0]], np.float32),
        "rewards": rewards,
        "next_observations": np.stack((second, terminal)),
        "discounts": discounts,
        "terminals": terminals,
        "delta_t_seconds": durations,
        "trajectory_local_id": np.zeros(2, np.int64),
        "wait_passenger_seconds": np.asarray(
            [0.0, components["wait_passenger_seconds"]], np.float64
        ),
        "in_vehicle_passenger_seconds": np.asarray(
            [0.0, components["in_vehicle_passenger_seconds"]], np.float64
        ),
        "event_bus_code": np.asarray([0, 1], np.int32),
        "event_line_index": np.asarray([0, 3], np.int16),
        "event_station_index": np.asarray([2, 5], np.int16),
        "event_decision_time_seconds": np.asarray([10.0, 10.0], np.float64),
        "event_base_dwell_seconds": np.asarray([8.0, 8.0], np.float64),
    }
    np.savez_compressed(path, **arrays)
    summary = {
        "schema_version": "transit-smdp-v7-shard-v1",
        "contract_schema": "transit-smdp-v7",
        "contract": DEFAULT_CONFIG_V7.to_dict(),
        "raw_state_names": list(RAW_STATE_NAMES),
        "event_identity": {
            "bus_code_to_id": ["bus-0", "bus-1"],
            "line_index_to_id": ["line-0", "line-1", "line-2", "line-3"],
            "semantics": "test metadata",
        },
        "shard_sha256": _hash(path),
        "job_signature": None,
        "split": "train",
        "diagnostic": True,
        "training_eligible": False,
        "policy_name": "zero",
        "policy_seed": 1,
        "sumo_seed": 1,
        "scenario": {
            "manifest_sha256": "scenario",
            "selection_seed": 101,
            "demand_scale": 1.0,
        },
        "episode": {
            "configured_seconds": 70,
            "observed_end_time_seconds": 70,
            "control_decisions": 2,
            "transitions": 2,
        },
        "training_reward": {
            "reward_sum": float(rewards.sum()),
            "chronological_coverage_errors": {
                "wait_passenger_seconds": 0.0,
                "in_vehicle_passenger_seconds": 0.0,
                "reward": 0.0,
            },
        },
        "ledger_audit": {
            "legacy_sampled_trip_histories_used_as_ground_truth": False,
            "count_source_audit": {
                "max_waiting_count_error": 0,
                "max_in_vehicle_count_error": 0,
            },
        },
        "service": {
            "departed_trips": 1,
            "completed_trips": 0,
            "active_trips_at_horizon": 1,
        },
    }
    path.with_suffix(".npz.json").write_text(json.dumps(summary))
    return path


def test_v7_builder_and_independent_audit_preserve_chronology(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    dataset = tmp_path / "dataset.h5"
    summary = build_dataset([shard], dataset, allow_partial=True)
    assert summary["rows"] == 2
    assert summary["trajectories"] == 1
    assert summary["terminal_rows"] == 1
    assert summary["zero_duration_rows"] == 1
    with h5py.File(dataset, "r") as handle:
        np.testing.assert_array_equal(handle["demand_realization_seed"][:], 101)
        assert handle.attrs["discount_semantics"].startswith(
            "finite-horizon undiscounted objective"
        )
        assert "training scale is external" in handle.attrs["reward_semantics"]
        assert handle.attrs["raw_state_dim"] == 245
        assert handle.attrs["state_dim"] == 229
    report = audit(dataset)
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["chronological_reward_sums"]["0"] == pytest.approx(
        summary["reward"]["sum"]
    )
    chunked_report = audit(dataset, chunk_rows=1)
    assert chunked_report["passed"] is True
    assert chunked_report["reconstruction_errors"] == report[
        "reconstruction_errors"
    ]
    assert chunked_report["actions"] == report["actions"]
    assert report["actions"]["exact_zero_hold_rows"] == 1
    assert report["actions"]["exact_max_hold_rows"] == 0
    assert report["actions"]["positive_hold_bin_rows"] == [0, 0, 1, 0]
    assert report["actions"]["coverage_gate"]["applicable"] is False
    credit_report = audit_credit(dataset)
    assert credit_report["passed"] is True
    assert credit_report["physical_batches"] == 1
    assert credit_report["multi_action_batches"] == 1
    assert credit_report["bellman_credit"][
        "maximum_absolute_slot_target_error"
    ] == 0.0


def test_v7_streamed_logical_hash_matches_in_memory_hash(tmp_path):
    arrays = {
        "matrix": np.arange(10_003 * 3, dtype=np.float32).reshape(-1, 3),
        "vector": np.arange(10_003, dtype=np.int16),
    }
    metadata = {"schema": "streaming-hash-test", "rows": 10_003}
    path = tmp_path / "hash.h5"
    with h5py.File(path, "w") as handle:
        for name, values in arrays.items():
            handle.create_dataset(name, data=values, chunks=True)
        streamed = _dataset_content_sha256_hdf5(handle, metadata)

    assert streamed == dataset_content_sha256(arrays, metadata)


def test_v7_streaming_builder_preserves_sorted_shard_boundaries(tmp_path):
    first = _write_shard(tmp_path / "a-shards")
    second = _write_shard(tmp_path / "z-shards")
    for path, policy_name, policy_seed in (
        (first, "a-policy", 11),
        (second, "z-policy", 22),
    ):
        summary_path = path.with_suffix(".npz.json")
        summary = json.loads(summary_path.read_text())
        summary["policy_name"] = policy_name
        summary["policy_seed"] = policy_seed
        summary_path.write_text(json.dumps(summary))

    output = tmp_path / "two-shards.h5"
    summary = build_dataset(
        [second, first], output, allow_partial=True
    )

    assert summary["rows"] == 4
    assert summary["source_shards"] == 2
    with h5py.File(output, "r") as handle:
        assert json.loads(handle.attrs["policy_names"]) == {
            "a-policy": 0,
            "z-policy": 1,
        }
        np.testing.assert_array_equal(handle["policy_id"][:], [0, 0, 1, 1])
        np.testing.assert_array_equal(
            handle["policy_seed"][:], [11, 11, 22, 22]
        )
        np.testing.assert_array_equal(
            handle["source_shard_id"][:], [0, 0, 1, 1]
        )


def test_v7_credit_audit_rejects_cost_on_an_intermediate_token(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    dataset = tmp_path / "dataset.h5"
    build_dataset([shard], dataset, allow_partial=True)
    with h5py.File(dataset, "r+") as handle:
        handle["rewards"][0] = -1.0

    report = audit_credit(dataset)

    assert report["passed"] is False
    assert report["counters"]["intermediate_reward_errors"] == 1


def test_v7_validation_rejects_lost_same_time_action(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    with np.load(shard, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]).copy() for name in handle.files}
    action_position = RAW_STATE_INDEX["slot_00_prior_policy_action"]
    arrays["next_observations"][0, action_position] = -1.0
    with pytest.raises(ValueError, match="lost the preceding action"):
        validate_serialized_sequence(arrays, shard)


def test_v7_validation_rejects_multiple_terminal_rows(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    with np.load(shard, allow_pickle=False) as handle:
        arrays = {name: np.asarray(handle[name]).copy() for name in handle.files}
    arrays["terminals"][0] = 1
    with pytest.raises(ValueError, match="exactly one terminal"):
        validate_serialized_sequence(arrays, shard)


def test_v7_builder_accepts_and_audits_collection_matrix_v2(tmp_path):
    shard = _write_shard(tmp_path / "shards")
    manifest = tmp_path / "matrix.json"
    payload = {
        "schema_version": "transit-smdp-v7-collection-matrix-v2",
        "study_stage": "formal",
        "environment_design": {"demand_scales": [1.0]},
        "independent_realization_blocks": 1,
        "environment_cells": 1,
        "source_sha256": None,
        "jobs": [
            {
                "block_id": 0,
                "realization_block_id": 0,
                "output": str(shard.resolve()),
                "signature": None,
                "policy": {"policy_name": "zero"},
                "policy_seed": 1,
                "sumo_seed": 1,
                "split": "train",
                "diagnostic": True,
                "scenario": {
                    "sha256": "scenario",
                    "selection_seed": 101,
                    "demand_scale": 1.0,
                },
            }
        ],
    }
    manifest.write_text(json.dumps(payload))

    provenance = validate_collection_matrix(manifest, [shard])
    assert provenance["schema_version"] == payload["schema_version"]
    assert provenance["independent_realization_blocks"] == 1
    assert provenance["environment_cells"] == 1

    payload["environment_design"] = {"demand_scales": []}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="lacks declared demand scales"):
        validate_collection_matrix(manifest, [shard])


def test_v7_manifest_resolves_shards_across_roots(tmp_path):
    first = tmp_path / "zero-root" / "zero.npz"
    second = tmp_path / "behavior-root" / "behavior.npz"
    manifest = tmp_path / "combined.json"
    payload = {
        "schema_version": "transit-smdp-v7-collection-matrix-v2",
        "jobs": [{"output": str(first)}, {"output": str(second)}],
    }
    manifest.write_text(json.dumps(payload))

    assert shard_paths_from_manifest(manifest) == sorted(
        [first.resolve(), second.resolve()]
    )
    payload["jobs"].append({"output": str(first)})
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="repeats an output path"):
        shard_paths_from_manifest(manifest)


def test_v7_behavior_mixture_covers_full_positive_hold_range():
    policies = {policy["policy_name"]: policy for policy in POLICIES}
    broad = policies["random-hurdle-p0.5-uniform"]
    assert broad["random_zero_probability"] == 0.5
    assert broad["random_beta_alpha"] == 1.0
    assert broad["random_beta_beta"] == 1.0
    seed = policy_seed(7, broad["policy_name"], "scenario", 11)
    assert seed == policy_seed(7, broad["policy_name"], "scenario", 11)
    assert seed != policy_seed(7, broad["policy_name"], "scenario", 12)


def test_v7_pilot_statistics_disclose_single_realization():
    report = clustered_paired_summary(
        {0: (0, -3.0)}, allow_single_realization=True
    )
    assert report["inferential_statistics_estimable"] is False
    assert report["exact_two_sided_sign_flip_pvalue"] is None
    with pytest.raises(ValueError, match="at least two realizations"):
        clustered_paired_summary({0: (0, -3.0)})


def test_v7_design_reuses_validated_physical_scenarios(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "training_environment_design_v6_formal_v1.json"
    )
    output = tmp_path / "training_environment_design_v7.json"
    payload = build_design(
        Namespace(source_design=source, output=output, force=False)
    )
    assert payload["schema_version"] == TRAINING_DESIGN_SCHEMA
    assert payload["independent_realization_blocks"] == 10
    assert payload["environment_cells"] == 30
    assert payload["physical_scenario_reuse"]["path"] == str(source.resolve())
