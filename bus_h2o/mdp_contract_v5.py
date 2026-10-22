"""True-welfare event-driven transit SMDP contract.

Version 5 retains the audited v4 physical action and local event features, adds
two broadcast system-occupancy features, and replaces the analytical headway
proxy with accrued passenger-time differences from the simulator ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from bus_h2o.mdp_contract_v4 import (
    ACTION_DIM,
    CANONICAL_STATE_NAMES as CANONICAL_STATE_NAMES_V4,
    RAW_EVENT_STATE_NAMES as RAW_EVENT_STATE_NAMES_V4,
    TransitMDPConfigV4,
    continuous_time_discount_v4,
    hold_seconds_to_policy_action_v4,
    policy_action_to_hold_seconds_v4,
    raw_event_to_canonical_v4,
    transition_duration_seconds_v4,
)
from bus_h2o.passenger_time import (
    PassengerTimeSnapshot,
    generalized_passenger_time_delta,
)


SCHEMA_VERSION = "transit-smdp-v5"
RAW_EVENT_STATE_NAMES = RAW_EVENT_STATE_NAMES_V4 + (
    "system_waiting_passengers",
    "system_in_vehicle_passengers",
)
RAW_EVENT_STATE_DIM = len(RAW_EVENT_STATE_NAMES)
RAW_EVENT_STATE_INDEX = {
    name: index for index, name in enumerate(RAW_EVENT_STATE_NAMES)
}

CANONICAL_STATE_NAMES = CANONICAL_STATE_NAMES_V4 + (
    "system_waiting_fraction",
    "system_in_vehicle_fraction",
)
CANONICAL_STATE_DIM = len(CANONICAL_STATE_NAMES)
CANONICAL_STATE_INDEX = {
    name: index for index, name in enumerate(CANONICAL_STATE_NAMES)
}


@dataclass(frozen=True)
class TransitMDPConfigV5(TransitMDPConfigV4):
    """Physical, welfare, and normalization constants for v5."""

    reward_unit_passenger_seconds: float = 100_000.0
    nominal_system_capacity: float = 20_000.0
    max_system_capacity_fraction: float = 1.25

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("nominal_system_capacity", "max_system_capacity_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


DEFAULT_CONFIG_V5 = TransitMDPConfigV5()


def _raw_batch(raw_observations: np.ndarray) -> tuple[np.ndarray, bool]:
    observations = np.asarray(raw_observations, dtype=np.float32)
    was_vector = observations.ndim == 1
    if was_vector:
        observations = observations[None, :]
    if observations.ndim != 2 or observations.shape[1] != RAW_EVENT_STATE_DIM:
        raise ValueError(
            "raw_observations must have shape "
            f"({RAW_EVENT_STATE_DIM},) or (N, {RAW_EVENT_STATE_DIM}); "
            f"got {observations.shape}"
        )
    if not np.isfinite(observations).all():
        raise ValueError("raw_observations contains non-finite values")
    return observations, was_vector


def raw_event_to_canonical_v5(
    raw_observations: np.ndarray,
    station_max_by_line: Iterable[float],
    config: TransitMDPConfigV5 = DEFAULT_CONFIG_V5,
) -> np.ndarray:
    """Normalize local physical features and broadcast system occupancies."""

    observations, was_vector = _raw_batch(raw_observations)
    local = raw_event_to_canonical_v4(
        observations[:, : len(RAW_EVENT_STATE_NAMES_V4)],
        station_max_by_line,
        config,
    )
    system_counts = observations[:, -2:]
    if np.any(system_counts < 0.0):
        raise ValueError("system passenger counts must be non-negative")
    system_fractions = np.clip(
        system_counts / config.nominal_system_capacity,
        0.0,
        config.max_system_capacity_fraction,
    )
    canonical = np.column_stack((local, system_fractions)).astype(
        np.float32, copy=False
    )
    return canonical[0] if was_vector else canonical


def policy_action_to_hold_seconds_v5(
    policy_action: np.ndarray | float,
    config: TransitMDPConfigV5 = DEFAULT_CONFIG_V5,
) -> np.ndarray | float:
    return policy_action_to_hold_seconds_v4(policy_action, config)


def hold_seconds_to_policy_action_v5(
    hold_seconds: np.ndarray | float,
    config: TransitMDPConfigV5 = DEFAULT_CONFIG_V5,
) -> np.ndarray | float:
    return hold_seconds_to_policy_action_v4(hold_seconds, config)


def transition_duration_seconds_v5(
    raw_observations: np.ndarray, raw_next_observations: np.ndarray
) -> np.ndarray | float:
    observations, was_vector = _raw_batch(raw_observations)
    next_observations, next_was_vector = _raw_batch(raw_next_observations)
    if was_vector != next_was_vector or observations.shape != next_observations.shape:
        raise ValueError("raw observation batches must have matching shapes")
    local_observations = observations[:, : len(RAW_EVENT_STATE_NAMES_V4)]
    local_next_observations = next_observations[
        :, : len(RAW_EVENT_STATE_NAMES_V4)
    ]
    if was_vector:
        local_observations = local_observations[0]
        local_next_observations = local_next_observations[0]
    return transition_duration_seconds_v4(
        local_observations,
        local_next_observations,
    )


def continuous_time_discount_v5(
    delta_t_seconds: np.ndarray | float,
    config: TransitMDPConfigV5 = DEFAULT_CONFIG_V5,
) -> np.ndarray | float:
    return continuous_time_discount_v4(delta_t_seconds, config)


def passenger_time_snapshot_v5(
    value: PassengerTimeSnapshot | Mapping[str, float | int],
) -> PassengerTimeSnapshot:
    if isinstance(value, PassengerTimeSnapshot):
        return value
    return PassengerTimeSnapshot(
        time_seconds=float(value["time_seconds"]),
        waiting_passengers=int(value["waiting_passengers"]),
        in_vehicle_passengers=int(value["in_vehicle_passengers"]),
        accrued_wait_seconds=float(value["accrued_wait_seconds"]),
        accrued_in_vehicle_seconds=float(value["accrued_in_vehicle_seconds"]),
    )


def transition_cost_components_v5(
    passenger_time_start: PassengerTimeSnapshot | Mapping[str, float | int],
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
    config: TransitMDPConfigV5 = DEFAULT_CONFIG_V5,
) -> dict[str, float]:
    """Return exact team passenger-time accrued between two decision epochs."""

    return generalized_passenger_time_delta(
        passenger_time_snapshot_v5(passenger_time_start),
        passenger_time_snapshot_v5(passenger_time_end),
        waiting_time_value=config.waiting_time_value,
        in_vehicle_time_value=config.in_vehicle_time_value,
        reward_unit_passenger_seconds=config.reward_unit_passenger_seconds,
    )


def transition_reward_v5(
    passenger_time_start: PassengerTimeSnapshot | Mapping[str, float | int],
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
    config: TransitMDPConfigV5 = DEFAULT_CONFIG_V5,
) -> float:
    """Return negative accrued generalized passenger time in training units."""

    return -transition_cost_components_v5(
        passenger_time_start, passenger_time_end, config
    )["total_cost"]


__all__ = [
    "ACTION_DIM",
    "CANONICAL_STATE_DIM",
    "CANONICAL_STATE_INDEX",
    "CANONICAL_STATE_NAMES",
    "DEFAULT_CONFIG_V5",
    "RAW_EVENT_STATE_DIM",
    "RAW_EVENT_STATE_INDEX",
    "RAW_EVENT_STATE_NAMES",
    "SCHEMA_VERSION",
    "TransitMDPConfigV5",
    "continuous_time_discount_v5",
    "hold_seconds_to_policy_action_v5",
    "passenger_time_snapshot_v5",
    "policy_action_to_hold_seconds_v5",
    "raw_event_to_canonical_v5",
    "transition_cost_components_v5",
    "transition_duration_seconds_v5",
    "transition_reward_v5",
]
