"""Replay buffer for the immutable transit-smdp-v3 dataset contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

from bus_h2o.mdp_contract import ACTION_DIM, CANONICAL_STATE_DIM, SCHEMA_VERSION


SPLIT_NAME_TO_ID = {"train": 0, "validation": 1, "test": 2}
RATIO_VALIDATION_FRACTION = 0.2
ONLINE_RATIO_VALIDATION_PERIOD = 5
ONLINE_RATIO_VALIDATION_OFFSET = 1


def _stable_unit_score(unit: tuple[int, ...]) -> int:
    payload = ":".join(str(int(value)) for value in unit).encode("ascii")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"transit-ratio-v1",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _ranked_group_validation_mask(
    *columns: np.ndarray,
    validation_fraction: float = RATIO_VALIDATION_FRACTION,
) -> np.ndarray:
    """Select complete groups for a deterministic, approximately sized holdout."""

    if not columns:
        raise ValueError("at least one grouping column is required")
    arrays = [np.asarray(column).reshape(-1) for column in columns]
    if any(len(column) != len(arrays[0]) for column in arrays[1:]):
        raise ValueError("ratio grouping columns must have equal lengths")
    if not len(arrays[0]):
        raise ValueError("ratio grouping columns cannot be empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("ratio validation fraction must lie in (0, 1)")

    row_units = [
        tuple(int(column[index]) for column in arrays)
        for index in range(len(arrays[0]))
    ]
    unique_units = sorted(set(row_units))
    if len(unique_units) < 2:
        raise ValueError(
            "offline trajectories are too few for a ratio train/validation split"
        )
    validation_units = min(
        len(unique_units) - 1,
        max(1, int(round(validation_fraction * len(unique_units)))),
    )
    ranked = sorted(unique_units, key=lambda unit: (_stable_unit_score(unit), unit))
    selected = set(ranked[:validation_units])
    return np.asarray([unit in selected for unit in row_units], dtype=bool)


def _read_rows(dataset: h5py.Dataset, selected: np.ndarray, chunk_rows: int) -> np.ndarray:
    """Read a boolean-selected HDF5 dataset without a multi-million-index query."""

    output_shape = (int(selected.sum()),) + dataset.shape[1:]
    output = np.empty(output_shape, dtype=dataset.dtype)
    write_at = 0
    for start in range(0, dataset.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dataset.shape[0])
        local_mask = selected[start:stop]
        count = int(local_mask.sum())
        if count:
            output[write_at : write_at + count] = dataset[start:stop][local_mask]
            write_at += count
    if write_at != len(output):
        raise RuntimeError("filtered HDF5 read produced an inconsistent row count")
    return output


class TransitReplayBuffer:
    """Offline split plus an append-only online partition.

    Rewards and discounts are consumed exactly as stored.  This class has no
    reward-correction or normalization side effects.
    """

    REQUIRED_DATASETS = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "discounts",
        "terminals",
        "split_id",
        "policy_id",
        "policy_seed",
        "sumo_seed",
        "episode_id",
        "trajectory_id",
    )

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str = "train",
        device: str | torch.device = "cpu",
        online_capacity: int = 500_000,
        max_offline_rows: int = 0,
        subsample_seed: int = 0,
        policy_ids: Iterable[int] | None = None,
        chunk_rows: int = 250_000,
        schema_version: str = SCHEMA_VERSION,
        state_dim: int = CANONICAL_STATE_DIM,
        action_dim: int = ACTION_DIM,
        include_delta_t: bool = False,
        device_cache_observation_dtype: str = "float32",
    ) -> None:
        if split not in SPLIT_NAME_TO_ID:
            raise ValueError(f"unknown split {split!r}; expected one of {tuple(SPLIT_NAME_TO_ID)}")
        self.dataset_path = Path(dataset_path).resolve()
        self.device = torch.device(device)
        self.split = split
        self.schema_version = str(schema_version)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.include_delta_t = bool(include_delta_t)
        if device_cache_observation_dtype not in {"float32", "float16"}:
            raise ValueError(
                "device cache observation dtype must be 'float32' or 'float16'"
            )
        self.device_cache_observation_dtype = device_cache_observation_dtype
        self._device_cache_observation_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
        }[device_cache_observation_dtype]
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")

        with h5py.File(self.dataset_path, "r") as handle:
            schema = handle.attrs.get("schema_version")
            if schema != self.schema_version:
                raise ValueError(
                    f"expected schema {self.schema_version!r}, got {schema!r}"
                )
            required_datasets = set(self.REQUIRED_DATASETS)
            if self.include_delta_t:
                required_datasets.add("delta_t_seconds")
            missing = sorted(required_datasets.difference(handle.keys()))
            if missing:
                raise KeyError(f"dataset is missing required arrays: {missing}")
            if handle["observations"].shape[1] != self.state_dim:
                raise ValueError("canonical state width mismatch")
            if handle["actions"].shape[1] != self.action_dim:
                raise ValueError("canonical action width mismatch")

            selected = handle["split_id"][:] == SPLIT_NAME_TO_ID[split]
            if policy_ids is not None:
                allowed = np.asarray(tuple(policy_ids), dtype=np.int8)
                selected &= np.isin(handle["policy_id"][:], allowed)
            selected_indices = np.flatnonzero(selected)
            if max_offline_rows > 0 and len(selected_indices) > max_offline_rows:
                rng = np.random.default_rng(subsample_seed)
                keep = np.sort(
                    rng.choice(selected_indices, size=max_offline_rows, replace=False)
                )
                selected = np.zeros_like(selected, dtype=bool)
                selected[keep] = True

            self.observations = _read_rows(handle["observations"], selected, chunk_rows).astype(
                np.float32, copy=False
            )
            self.actions = _read_rows(handle["actions"], selected, chunk_rows).astype(
                np.float32, copy=False
            )
            self.rewards = _read_rows(handle["rewards"], selected, chunk_rows).astype(
                np.float32, copy=False
            )
            self.next_observations = _read_rows(
                handle["next_observations"], selected, chunk_rows
            ).astype(np.float32, copy=False)
            self.discounts = _read_rows(handle["discounts"], selected, chunk_rows).astype(
                np.float32, copy=False
            )
            self.delta_t_seconds = (
                _read_rows(
                    handle["delta_t_seconds"], selected, chunk_rows
                ).astype(np.float32, copy=False)
                if self.include_delta_t
                else None
            )
            self.terminals = _read_rows(handle["terminals"], selected, chunk_rows).astype(
                np.float32, copy=False
            )
            self.policy_ids = _read_rows(handle["policy_id"], selected, chunk_rows).astype(
                np.int8, copy=False
            )
            self.policy_seeds = _read_rows(
                handle["policy_seed"], selected, chunk_rows
            ).astype(
                np.int32, copy=False
            )
            self.sumo_seeds = _read_rows(
                handle["sumo_seed"], selected, chunk_rows
            ).astype(np.int32, copy=False)
            self.scenario_ids = (
                _read_rows(handle["scenario_id"], selected, chunk_rows).astype(
                    np.int32, copy=False
                )
                if "scenario_id" in handle
                else None
            )
            self.demand_realization_seeds = (
                _read_rows(
                    handle["demand_realization_seed"], selected, chunk_rows
                ).astype(np.int32, copy=False)
                if "demand_realization_seed" in handle
                else None
            )
            self.episode_ids = _read_rows(
                handle["episode_id"], selected, chunk_rows
            ).astype(np.int64, copy=False)
            self.trajectory_ids = _read_rows(
                handle["trajectory_id"], selected, chunk_rows
            ).astype(np.int64, copy=False)
            self.station_max_by_line = np.asarray(
                json.loads(handle.attrs["station_max_by_line"]), dtype=np.float32
            )
            self.contract_config = json.loads(handle.attrs["config"])
            self.policy_names = json.loads(handle.attrs["policy_names"])

        self.offline_size = len(self.observations)
        if self.offline_size == 0:
            raise ValueError(f"split {split!r} contains no rows")
        if not np.isfinite(self.rewards).all() or not np.isfinite(self.discounts).all():
            raise ValueError("offline reward or discount contains non-finite values")
        if np.any(self.discounts < 0) or np.any(self.discounts > 1):
            raise ValueError("offline discounts must lie in [0, 1]")
        if self.delta_t_seconds is not None and (
            not np.isfinite(self.delta_t_seconds).all()
            or np.any(self.delta_t_seconds < 0.0)
        ):
            raise ValueError("offline transition durations must be finite and nonnegative")
        # Paired behavior policies and correlated demand-scale variants from
        # one realization must stay on the same side. This prevents the same
        # jittered source passengers from leaking across ratio partitions.
        self._offline_ratio_validation = (
            _ranked_group_validation_mask(
                self.demand_realization_seeds, self.sumo_seeds
            )
            if self.demand_realization_seeds is not None
            else _ranked_group_validation_mask(
                self.scenario_ids, self.sumo_seeds
            )
            if self.scenario_ids is not None
            else self._trajectory_validation_mask(self.trajectory_ids)
        )
        if not self._offline_ratio_validation.any() or self._offline_ratio_validation.all():
            raise ValueError(
                "offline trajectories are too few for a ratio train/validation split"
            )
        self._offline_partition_indices = {
            "ratio_train": np.flatnonzero(~self._offline_ratio_validation),
            "ratio_validation": np.flatnonzero(self._offline_ratio_validation),
        }

        self.online_capacity = int(online_capacity)
        if self.online_capacity <= 0:
            raise ValueError("online_capacity must be positive")
        self._online_observations = np.empty(
            (self.online_capacity, self.state_dim), dtype=np.float32
        )
        self._online_actions = np.empty(
            (self.online_capacity, self.action_dim), dtype=np.float32
        )
        self._online_rewards = np.empty(self.online_capacity, dtype=np.float32)
        self._online_next_observations = np.empty(
            (self.online_capacity, self.state_dim), dtype=np.float32
        )
        self._online_discounts = np.empty(self.online_capacity, dtype=np.float32)
        self._online_delta_t_seconds = (
            np.empty(self.online_capacity, dtype=np.float32)
            if self.include_delta_t
            else None
        )
        self._online_terminals = np.empty(self.online_capacity, dtype=np.float32)
        self._online_episode_ids = np.empty(self.online_capacity, dtype=np.int64)
        self._online_trajectory_ids = np.empty(self.online_capacity, dtype=np.int64)
        self._online_ratio_validation = np.zeros(self.online_capacity, dtype=bool)
        self._online_partition_indices: dict[str, np.ndarray] = {}
        self._online_size = 0
        self._online_pointer = 0
        self._online_total_appended = 0
        self._offline_device_cache: dict[str, torch.Tensor] | None = None
        self._online_device_cache: dict[str, torch.Tensor] | None = None
        self._online_cache_size = -1
        self._online_cache_capacity = 0

    @property
    def online_size(self) -> int:
        return self._online_size

    @staticmethod
    def _trajectory_validation_mask(trajectory_ids: np.ndarray) -> np.ndarray:
        """Assign complete offline trajectories to a deterministic holdout."""

        return _ranked_group_validation_mask(trajectory_ids)

    @staticmethod
    def _online_trajectory_validation_mask(
        trajectory_ids: np.ndarray,
    ) -> np.ndarray:
        """Reserve every fifth online episode, after one train episode.

        Unlike a ranked split, this assignment is stable while an append-only
        replay grows. Trajectory zero is deliberately training data so ratio
        learning can start from a partial first warmup episode; trajectory one
        provides the first environment-disjoint diagnostic episode.
        """

        values = np.asarray(trajectory_ids, dtype=np.int64)
        if np.any(values < 0):
            raise ValueError("online trajectory ids must be nonnegative")
        return (
            values % ONLINE_RATIO_VALIDATION_PERIOD
            == ONLINE_RATIO_VALIDATION_OFFSET
        )

    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        discount: float,
        terminal: float = 0.0,
        *,
        episode_id: int,
        trajectory_id: int,
        delta_t_seconds: float | None = None,
    ) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if observation.shape != (self.state_dim,):
            raise ValueError(f"observation must have shape ({self.state_dim},)")
        if next_observation.shape != (self.state_dim,):
            raise ValueError(f"next_observation must have shape ({self.state_dim},)")
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        if not np.isfinite(observation).all() or not np.isfinite(next_observation).all():
            raise ValueError("online observation contains non-finite values")
        if not np.isfinite(action).all() or np.any(np.abs(action) > 1.00001):
            raise ValueError("online action must be finite and lie in [-1, 1]")
        if not np.isfinite(reward) or not 0.0 <= discount <= 1.0:
            raise ValueError("online reward/discount is invalid")
        if self.include_delta_t:
            if (
                delta_t_seconds is None
                or not np.isfinite(delta_t_seconds)
                or delta_t_seconds < 0.0
            ):
                raise ValueError(
                    "online transition duration must be finite and nonnegative"
                )

        index = self._online_pointer
        self._online_observations[index] = observation
        self._online_actions[index] = action
        self._online_rewards[index] = reward
        self._online_next_observations[index] = next_observation
        self._online_discounts[index] = discount
        if self._online_delta_t_seconds is not None:
            self._online_delta_t_seconds[index] = float(delta_t_seconds)
        self._online_terminals[index] = terminal
        self._online_episode_ids[index] = int(episode_id)
        self._online_trajectory_ids[index] = int(trajectory_id)
        # Entire online episodes stay on one side of the ratio split.
        # All rows remain available to the actor and critics.
        self._online_ratio_validation[index] = self._online_trajectory_validation_mask(
            np.asarray([trajectory_id], dtype=np.int64)
        )[0]
        self._online_pointer = (index + 1) % self.online_capacity
        self._online_size = min(self._online_size + 1, self.online_capacity)
        self._online_total_appended += 1
        self._online_partition_indices.clear()
        if self._online_device_cache is not None:
            self._ensure_online_device_cache_capacity(self.online_size)
            values = self._partition_dict(
                observation,
                action,
                np.float32(reward),
                next_observation,
                np.float32(discount),
                np.float32(terminal),
                (
                    np.float32(delta_t_seconds)
                    if self.include_delta_t
                    else None
                ),
            )
            for key, value in values.items():
                self._online_device_cache[key][index].copy_(
                    torch.as_tensor(value, device=self.device)
                )
            self._online_cache_size = self.online_size

    def _ensure_online_device_cache_capacity(self, required: int) -> None:
        if required < 0 or required > self.online_capacity:
            raise ValueError("online cache requirement is outside replay capacity")
        if required <= self._online_cache_capacity:
            return
        target = min(
            self.online_capacity,
            max(1024, 1 << max(required - 1, 0).bit_length()),
        )
        source_shapes = {
            "observations": (target, self.state_dim),
            "actions": (target, self.action_dim),
            "rewards": (target,),
            "next_observations": (target, self.state_dim),
            "discounts": (target,),
            "terminals": (target,),
        }
        if self.include_delta_t:
            source_shapes["delta_t_seconds"] = (target,)
        grown = {
            key: torch.empty(
                shape,
                dtype=(
                    self._device_cache_observation_dtype
                    if key in {"observations", "next_observations"}
                    else torch.float32
                ),
                device=self.device,
            )
            for key, shape in source_shapes.items()
        }
        if self._online_device_cache is not None and self._online_cache_size > 0:
            retained = min(self._online_cache_size, self._online_cache_capacity)
            for key in grown:
                grown[key][:retained].copy_(
                    self._online_device_cache[key][:retained]
                )
        self._online_device_cache = grown
        self._online_cache_capacity = target

    def _partition_dict(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        discounts,
        terminals,
        delta_t_seconds=None,
    ):
        values = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "discounts": discounts,
            "terminals": terminals,
        }
        if self.include_delta_t:
            if delta_t_seconds is None:
                raise ValueError("transition durations are required by this replay")
            values["delta_t_seconds"] = delta_t_seconds
        return values

    def materialize_device_cache(
        self, *, include_online: bool = True
    ) -> dict[str, int | str]:
        """Copy replay partitions to the training device once per collection phase."""

        if self._offline_device_cache is None:
            offline = self._partition_dict(
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.discounts,
                self.terminals,
                self.delta_t_seconds,
            )
            self._offline_device_cache = {
                key: torch.as_tensor(
                    value,
                    device=self.device,
                    dtype=(
                        self._device_cache_observation_dtype
                        if key in {"observations", "next_observations"}
                        else None
                    ),
                )
                for key, value in offline.items()
            }
        if include_online and self.online_size:
            valid = slice(0, self.online_size)
            source = self._partition_dict(
                self._online_observations,
                self._online_actions,
                self._online_rewards,
                self._online_next_observations,
                self._online_discounts,
                self._online_terminals,
                self._online_delta_t_seconds,
            )
            self._ensure_online_device_cache_capacity(self.online_size)
            assert self._online_device_cache is not None
            for key, value in source.items():
                self._online_device_cache[key][valid].copy_(
                    torch.as_tensor(value[valid], device=self.device)
                )
            self._online_cache_size = self.online_size
        cache_tensors = list(self._offline_device_cache.values())
        if self._online_device_cache is not None:
            cache_tensors.extend(self._online_device_cache.values())
        return {
            "offline_rows": self.offline_size,
            "online_rows": self._online_cache_size if self._online_cache_size >= 0 else 0,
            "observation_storage_dtype": self.device_cache_observation_dtype,
            "bytes": int(
                sum(tensor.numel() * tensor.element_size() for tensor in cache_tensors)
            ),
        }

    def clear_device_cache(self) -> None:
        self._offline_device_cache = None
        self._online_device_cache = None
        self._online_cache_size = -1
        self._online_cache_capacity = 0

    def online_state_dict(self) -> dict[str, np.ndarray | int]:
        valid = slice(0, self.online_size)
        state = {
            "capacity": self.online_capacity,
            "size": self.online_size,
            "pointer": self._online_pointer,
            "observations": self._online_observations[valid].copy(),
            "actions": self._online_actions[valid].copy(),
            "rewards": self._online_rewards[valid].copy(),
            "next_observations": self._online_next_observations[valid].copy(),
            "discounts": self._online_discounts[valid].copy(),
            "terminals": self._online_terminals[valid].copy(),
            "episode_ids": self._online_episode_ids[valid].copy(),
            "trajectory_ids": self._online_trajectory_ids[valid].copy(),
            "ratio_validation": self._online_ratio_validation[valid].copy(),
            "total_appended": self._online_total_appended,
        }
        if self._online_delta_t_seconds is not None:
            state["delta_t_seconds"] = self._online_delta_t_seconds[valid].copy()
        return state

    def load_online_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.online_capacity:
            raise ValueError("checkpoint online replay capacity does not match")
        size = int(state["size"])
        pointer = int(state["pointer"])
        if not 0 <= size <= self.online_capacity:
            raise ValueError("checkpoint online replay size is invalid")
        if not 0 <= pointer < self.online_capacity:
            raise ValueError("checkpoint online replay pointer is invalid")
        expected = {
            "observations": (size, self.state_dim),
            "actions": (size, self.action_dim),
            "rewards": (size,),
            "next_observations": (size, self.state_dim),
            "discounts": (size,),
            "terminals": (size,),
            "episode_ids": (size,),
            "trajectory_ids": (size,),
        }
        if self.include_delta_t:
            expected["delta_t_seconds"] = (size,)
        targets = {
            "observations": self._online_observations,
            "actions": self._online_actions,
            "rewards": self._online_rewards,
            "next_observations": self._online_next_observations,
            "discounts": self._online_discounts,
            "terminals": self._online_terminals,
            "episode_ids": self._online_episode_ids,
            "trajectory_ids": self._online_trajectory_ids,
        }
        if self._online_delta_t_seconds is not None:
            targets["delta_t_seconds"] = self._online_delta_t_seconds
        for key, shape in expected.items():
            values = np.asarray(state[key])
            if values.shape != shape:
                raise ValueError(
                    f"checkpoint replay {key} has shape {values.shape}; expected {shape}"
                )
            targets[key][:size] = values
        ratio_validation = np.asarray(
            state.get("ratio_validation", np.zeros(size, dtype=bool)), dtype=bool
        )
        if ratio_validation.shape != (size,):
            raise ValueError("checkpoint ratio-validation mask has an invalid shape")
        self._online_ratio_validation[:size] = ratio_validation
        self._online_size = size
        self._online_pointer = pointer
        self._online_total_appended = int(state.get("total_appended", size))
        self._online_partition_indices.clear()
        self._online_device_cache = None
        self._online_cache_size = -1
        self._online_cache_capacity = 0

    def _eligible_indices(self, scope: str, partition: str) -> np.ndarray:
        if partition not in {"all", "ratio_train", "ratio_validation"}:
            raise ValueError(
                "partition must be 'all', 'ratio_train' or 'ratio_validation'"
            )
        if scope == "offline":
            size = self.offline_size
            validation = self._offline_ratio_validation
        elif scope == "online":
            size = self.online_size
            if size == 0:
                raise RuntimeError("online partition is empty")
            validation = self._online_ratio_validation[:size]
        else:
            raise ValueError("scope must be 'offline' or 'online'")
        if partition == "all":
            return np.arange(size, dtype=np.int64)
        if scope == "offline":
            indices = self._offline_partition_indices[partition]
        else:
            indices = self._online_partition_indices.get(partition)
            if indices is None:
                selected = validation if partition == "ratio_validation" else ~validation
                indices = np.flatnonzero(selected)
                self._online_partition_indices[partition] = indices
        if not len(indices):
            raise RuntimeError(f"{scope} {partition} partition is empty")
        return indices

    def partition_size(self, scope: str, partition: str = "all") -> int:
        """Return a replay-partition size without requiring it to be nonempty."""

        if partition not in {"all", "ratio_train", "ratio_validation"}:
            raise ValueError(
                "partition must be 'all', 'ratio_train' or 'ratio_validation'"
            )
        if scope == "offline":
            size = self.offline_size
            validation = self._offline_ratio_validation
        elif scope == "online":
            size = self.online_size
            validation = self._online_ratio_validation[:size]
        else:
            raise ValueError("scope must be 'offline' or 'online'")
        if partition == "all":
            return int(size)
        if partition == "ratio_validation":
            return int(validation.sum())
        return int(size - validation.sum())

    def _sample_indices(
        self,
        batch_size: int,
        scope: str,
        partition: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample replay rows without materializing ``arange(size)`` for all rows."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if partition == "all":
            if scope == "offline":
                size = self.offline_size
            elif scope == "online":
                size = self.online_size
                if size == 0:
                    raise RuntimeError("online partition is empty")
            else:
                raise ValueError("scope must be 'offline' or 'online'")
            return rng.integers(0, size, size=batch_size)
        eligible = self._eligible_indices(scope, partition)
        return eligible[rng.integers(0, len(eligible), size=batch_size)]

    def _sample_arrays(
        self,
        batch_size: int,
        scope: str,
        partition: str,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(batch_size, scope, partition, rng)
        if scope == "offline":
            values = {
                "observations": self.observations[indices],
                "actions": self.actions[indices],
                "rewards": self.rewards[indices],
                "next_observations": self.next_observations[indices],
                "discounts": self.discounts[indices],
                "terminals": self.terminals[indices],
            }
            if self.delta_t_seconds is not None:
                values["delta_t_seconds"] = self.delta_t_seconds[indices]
            return values
        if scope == "online":
            values = {
                "observations": self._online_observations[indices],
                "actions": self._online_actions[indices],
                "rewards": self._online_rewards[indices],
                "next_observations": self._online_next_observations[indices],
                "discounts": self._online_discounts[indices],
                "terminals": self._online_terminals[indices],
            }
            if self._online_delta_t_seconds is not None:
                values["delta_t_seconds"] = self._online_delta_t_seconds[indices]
            return values
        raise ValueError("scope must be 'offline' or 'online'")

    def sample(
        self,
        batch_size: int,
        *,
        scope: str,
        partition: str = "all",
        rng: np.random.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        generator = rng if rng is not None else np.random.default_rng()
        cache = None
        size = 0
        if scope == "offline" and self._offline_device_cache is not None:
            cache = self._offline_device_cache
            size = self.offline_size
        elif (
            scope == "online"
            and self._online_device_cache is not None
            and self._online_cache_size == self.online_size
        ):
            cache = self._online_device_cache
            size = self.online_size
        if cache is not None:
            selected = self._sample_indices(
                batch_size, scope, partition, generator
            )
            indices = torch.as_tensor(
                selected,
                device=self.device,
            )
            selected = {
                key: values.index_select(0, indices) for key, values in cache.items()
            }
            for key in ("observations", "next_observations"):
                if selected[key].dtype != torch.float32:
                    selected[key] = selected[key].float()
            return selected
        arrays = self._sample_arrays(batch_size, scope, partition, generator)
        return {
            key: torch.as_tensor(value, device=self.device)
            for key, value in arrays.items()
        }

    def sample_mixed(
        self,
        batch_size: int,
        online_fraction: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        if not 0.0 <= online_fraction <= 1.0:
            raise ValueError("online_fraction must lie in [0, 1]")
        online_n = int(round(batch_size * online_fraction)) if self.online_size else 0
        offline_n = batch_size - online_n
        if offline_n <= 0:
            raise ValueError("mixed batches must retain at least one offline sample")
        generator = rng if rng is not None else np.random.default_rng()
        offline = self.sample(offline_n, scope="offline", rng=generator)
        online = self.sample(online_n, scope="online", rng=generator) if online_n else None
        return offline, online
