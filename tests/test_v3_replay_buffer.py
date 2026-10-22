import json

import h5py
import numpy as np
import pytest
import torch

from SimpleSAC.transit_v3.replay_buffer import TransitReplayBuffer
from bus_h2o.mdp_contract import CANONICAL_STATE_DIM, SCHEMA_VERSION


def make_dataset(path):
    n = 6
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=np.arange(n * CANONICAL_STATE_DIM, dtype=np.float32).reshape(n, -1))
        handle.create_dataset("actions", data=np.linspace(-1, 1, n, dtype=np.float32)[:, None])
        handle.create_dataset("rewards", data=-np.arange(n, dtype=np.float32))
        handle.create_dataset("next_observations", data=np.ones((n, CANONICAL_STATE_DIM), np.float32))
        handle.create_dataset("discounts", data=np.linspace(0.9, 0.99, n, dtype=np.float32))
        handle.create_dataset(
            "delta_t_seconds", data=np.arange(n, dtype=np.float32) * 10.0
        )
        handle.create_dataset("terminals", data=np.zeros(n, np.uint8))
        handle.create_dataset("split_id", data=np.zeros(n, np.uint8))
        handle.create_dataset("policy_id", data=np.asarray([0, 1, 2, 3, 4, 0], np.int8))
        handle.create_dataset("policy_seed", data=np.asarray([42, 42, 123, 789, 789, 1024], np.int32))
        handle.create_dataset("sumo_seed", data=np.asarray([11, 12, 13, 11, 12, 13], np.int32))
        handle.create_dataset("scenario_id", data=np.asarray([0, 1, 2, 0, 1, 2], np.int16))
        handle.create_dataset("demand_realization_seed", data=np.asarray([101, 102, 103, 101, 102, 103], np.int32))
        handle.create_dataset("episode_id", data=np.asarray([0, 0, 0, 1, 1, 1], np.int64))
        handle.create_dataset("trajectory_id", data=np.arange(n, dtype=np.int64))
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["station_max_by_line"] = json.dumps([10.0] * 12)
        handle.attrs["config"] = json.dumps({"discount_gamma_per_minute": 0.99})
        handle.attrs["policy_names"] = json.dumps({"a": 0})


def test_replay_buffer_respects_split_and_never_changes_reward(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(path, split="train", online_capacity=3)
    assert buffer.offline_size == 6
    np.testing.assert_array_equal(buffer.rewards, [0, -1, -2, -3, -4, -5])
    assert set(buffer.policy_seeds.tolist()) == {42, 123, 789, 1024}
    assert set(buffer.sumo_seeds.tolist()) == {11, 12, 13}
    batch = buffer.sample(8, scope="offline", rng=np.random.default_rng(0))
    assert batch["observations"].shape == (8, CANONICAL_STATE_DIM)
    assert batch["actions"].shape == (8, 1)


def test_online_ring_buffer_and_mixed_sampling(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(path, split="train", online_capacity=2)
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    for trajectory_id, reward in enumerate((1.0, 2.0, 3.0)):
        buffer.append(
            state,
            np.asarray([-1.0]),
            reward,
            state,
            0.95,
            episode_id=0,
            trajectory_id=trajectory_id,
        )
    assert buffer.online_size == 2
    offline, online = buffer.sample_mixed(
        10, 0.4, rng=np.random.default_rng(0)
    )
    assert len(offline["rewards"]) == 6
    assert len(online["rewards"]) == 4
    assert set(online["rewards"].cpu().numpy().tolist()).issubset({2.0, 3.0})


def test_duration_enabled_replay_preserves_offline_online_and_checkpoint_values(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    source = TransitReplayBuffer(
        path, split="train", online_capacity=2, include_delta_t=True
    )
    offline = source.sample(8, scope="offline", rng=np.random.default_rng(0))
    assert "delta_t_seconds" in offline
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    source.append(
        state,
        np.asarray([-1.0]),
        1.0,
        state,
        1.0,
        episode_id=0,
        trajectory_id=0,
        delta_t_seconds=37.0,
    )
    source.materialize_device_cache()
    online = source.sample(1, scope="online", rng=np.random.default_rng(1))
    assert online["delta_t_seconds"].item() == 37.0

    restored = TransitReplayBuffer(
        path, split="train", online_capacity=2, include_delta_t=True
    )
    restored.load_online_state_dict(source.online_state_dict())
    assert restored._online_delta_t_seconds[0] == 37.0


def test_half_storage_cache_returns_float32_for_both_replay_domains(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(
        path,
        split="train",
        online_capacity=2,
        device_cache_observation_dtype="float16",
    )
    state = np.full(CANONICAL_STATE_DIM, 0.123456, np.float32)
    buffer.append(
        state,
        np.asarray([-1.0]),
        1.0,
        state,
        1.0,
        episode_id=0,
        trajectory_id=0,
    )
    stats = buffer.materialize_device_cache()
    assert stats["observation_storage_dtype"] == "float16"
    assert buffer._offline_device_cache["observations"].dtype is torch.float16
    assert buffer._online_device_cache["observations"].dtype is torch.float16
    for scope in ("offline", "online"):
        batch = buffer.sample(2, scope=scope, rng=np.random.default_rng(0))
        assert batch["observations"].dtype is torch.float32
        assert batch["next_observations"].dtype is torch.float32

    with pytest.raises(ValueError, match="observation dtype"):
        TransitReplayBuffer(
            path,
            split="train",
            online_capacity=2,
            device_cache_observation_dtype="int8",
        )


def test_float32_device_cache_and_host_sampling_are_bit_exact(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    uncached = TransitReplayBuffer(
        path, split="train", online_capacity=2, include_delta_t=True
    )
    cached = TransitReplayBuffer(
        path, split="train", online_capacity=2, include_delta_t=True
    )
    state = np.linspace(-0.75, 0.75, CANONICAL_STATE_DIM, dtype=np.float32)
    for replay in (uncached, cached):
        replay.append(
            state,
            np.asarray([0.375], np.float32),
            -123.5,
            state[::-1].copy(),
            0.9875,
            episode_id=7,
            trajectory_id=11,
            delta_t_seconds=37.25,
        )
    cached.materialize_device_cache()

    for scope in ("offline", "online"):
        host_batch = uncached.sample(
            32, scope=scope, rng=np.random.default_rng(20260809)
        )
        cache_batch = cached.sample(
            32, scope=scope, rng=np.random.default_rng(20260809)
        )
        assert host_batch.keys() == cache_batch.keys()
        for name in host_batch:
            assert torch.equal(host_batch[name], cache_batch[name]), name


def test_explicit_device_cache_is_updated_in_place_by_online_append(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(path, split="train", online_capacity=3)
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    buffer.append(
        state, np.asarray([-1.0]), 1.0, state, 0.95,
        episode_id=0, trajectory_id=0,
    )
    cache = buffer.materialize_device_cache()
    assert cache["offline_rows"] == 6
    assert cache["online_rows"] == 1
    assert cache["bytes"] > 0
    online_cache = buffer._online_device_cache
    assert online_cache is not None
    buffer.append(
        state, np.asarray([-1.0]), 2.0, state, 0.95,
        episode_id=0, trajectory_id=1,
    )
    assert buffer._online_device_cache is online_cache
    assert buffer._online_cache_size == 2
    batch = buffer.sample(8, scope="online", rng=np.random.default_rng(1))
    assert set(batch["rewards"].tolist()).issubset({1.0, 2.0})


def test_online_device_cache_grows_geometrically_without_becoming_stale(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(path, split="train", online_capacity=5000)
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    buffer.append(
        state,
        np.asarray([-1.0]),
        1.0,
        state,
        0.95,
        episode_id=0,
        trajectory_id=0,
    )
    buffer.materialize_device_cache()
    original_cache = buffer._online_device_cache
    assert buffer._online_cache_capacity == 1024
    for index in range(1, 1025):
        buffer.append(
            state,
            np.asarray([-1.0]),
            float(index + 1),
            state,
            0.95,
            episode_id=0,
            trajectory_id=0,
        )
    assert buffer._online_device_cache is not original_cache
    assert buffer._online_cache_capacity == 2048
    assert buffer._online_cache_size == buffer.online_size == 1025
    assert buffer._online_device_cache is not None
    assert buffer._online_device_cache["rewards"][0].item() == 1.0
    assert buffer._online_device_cache["rewards"][1024].item() == 1025.0


def test_online_replay_checkpoint_round_trip(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    source = TransitReplayBuffer(path, split="train", online_capacity=2)
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    for trajectory_id, reward in enumerate((1.0, 2.0, 3.0)):
        source.append(
            state,
            np.asarray([-1.0]),
            reward,
            state,
            0.95,
            episode_id=0,
            trajectory_id=trajectory_id,
        )
    restored = TransitReplayBuffer(path, split="train", online_capacity=2)
    restored.load_online_state_dict(source.online_state_dict())
    assert restored.online_size == 2
    assert restored._online_pointer == source._online_pointer
    np.testing.assert_array_equal(
        restored._online_rewards[:2], source._online_rewards[:2]
    )
    np.testing.assert_array_equal(
        restored._online_ratio_validation[:2],
        source._online_ratio_validation[:2],
    )
    np.testing.assert_array_equal(
        restored._online_trajectory_ids[:2], source._online_trajectory_ids[:2]
    )
    assert restored._online_total_appended == source._online_total_appended


def test_ratio_train_and_validation_partitions_are_disjoint(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(path, split="train", online_capacity=10)
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    for reward in range(10):
        buffer.append(
            state,
            np.asarray([-1.0]),
            reward,
            state,
            0.95,
            episode_id=0,
            trajectory_id=reward,
        )
    for scope in ("offline", "online"):
        train = set(buffer._eligible_indices(scope, "ratio_train").tolist())
        validation = set(
            buffer._eligible_indices(scope, "ratio_validation").tolist()
        )
        assert train
        assert validation
        assert train.isdisjoint(validation)
        assert train | validation == set(buffer._eligible_indices(scope, "all"))
        trajectory_ids = (
            buffer.trajectory_ids
            if scope == "offline"
            else buffer._online_trajectory_ids[: buffer.online_size]
        )
        assert set(trajectory_ids[list(train)]).isdisjoint(
            trajectory_ids[list(validation)]
        )

    offline_train = set(
        zip(
            buffer.demand_realization_seeds[
                ~buffer._offline_ratio_validation
            ].tolist(),
            buffer.sumo_seeds[~buffer._offline_ratio_validation].tolist(),
        )
    )
    offline_validation = set(
        zip(
            buffer.demand_realization_seeds[
                buffer._offline_ratio_validation
            ].tolist(),
            buffer.sumo_seeds[buffer._offline_ratio_validation].tolist(),
        )
    )
    assert offline_train.isdisjoint(offline_validation)
    assert len(offline_validation) == 1


def test_ratio_split_keeps_correlated_demand_scales_together(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    with h5py.File(path, "r+") as handle:
        handle["scenario_id"][:] = np.arange(6, dtype=np.int16)
        handle["demand_realization_seed"][:] = np.asarray(
            [101, 101, 101, 102, 102, 102], np.int32
        )
        handle["sumo_seed"][:] = np.asarray(
            [11, 11, 11, 12, 12, 12], np.int32
        )
    buffer = TransitReplayBuffer(path, split="train", online_capacity=10)
    train_realizations = set(
        buffer.demand_realization_seeds[
            ~buffer._offline_ratio_validation
        ].tolist()
    )
    validation_realizations = set(
        buffer.demand_realization_seeds[
            buffer._offline_ratio_validation
        ].tolist()
    )
    assert train_realizations.isdisjoint(validation_realizations)
    assert train_realizations | validation_realizations == {101, 102}


def test_online_ratio_split_starts_with_training_then_adds_holdout(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    buffer = TransitReplayBuffer(path, split="train", online_capacity=10)
    state = np.zeros(CANONICAL_STATE_DIM, np.float32)
    for _ in range(3):
        buffer.append(
            state,
            np.asarray([-1.0]),
            0.0,
            state,
            1.0,
            episode_id=0,
            trajectory_id=0,
        )
    assert buffer.partition_size("online", "ratio_train") == 3
    assert buffer.partition_size("online", "ratio_validation") == 0
    buffer.append(
        state,
        np.asarray([-1.0]),
        0.0,
        state,
        1.0,
        episode_id=1,
        trajectory_id=1,
    )
    assert buffer.partition_size("online", "ratio_train") == 3
    assert buffer.partition_size("online", "ratio_validation") == 1


def test_legacy_dataset_cannot_be_used_as_validation_or_test(tmp_path):
    path = tmp_path / "data.h5"
    make_dataset(path)
    with pytest.raises(ValueError, match="contains no rows"):
        TransitReplayBuffer(path, split="validation", online_capacity=1)
    with pytest.raises(ValueError, match="contains no rows"):
        TransitReplayBuffer(path, split="test", online_capacity=1)
    with pytest.raises(ValueError, match="unknown split"):
        TransitReplayBuffer(path, split="submission", online_capacity=1)
