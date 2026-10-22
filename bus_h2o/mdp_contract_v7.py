"""Deduplicated physical transit SMDP contract for the formal v7 protocol.

The v7 state stores physical time and system passenger counts once per
simultaneous decision batch.  Slot features contain only local event quantities
and the actions already selected in that batch.  Dataset rewards remain in
physical generalized passenger-seconds; method-specific numerical scaling is a
training configuration rather than part of the scientific data contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np

from bus_h2o.mdp_contract_v5 import (
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    passenger_time_snapshot_v5,
)
from bus_h2o.passenger_time import (
    PassengerTimeSnapshot,
    generalized_passenger_time_delta,
)


SCHEMA_VERSION = "transit-smdp-v7"
ACTION_DIM = 1
MAX_SIMULTANEOUS_DECISIONS = 16
HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS = 1.0

GLOBAL_EVENT_STATE_NAMES = (
    "sim_time_seconds",
    "system_waiting_passengers",
    "system_in_vehicle_passengers",
)
LOCAL_EVENT_STATE_NAMES = tuple(
    name for name in RAW_EVENT_STATE_NAMES if name not in GLOBAL_EVENT_STATE_NAMES
)
LOCAL_EVENT_STATE_DIM = len(LOCAL_EVENT_STATE_NAMES)
LOCAL_EVENT_STATE_INDEX = {
    name: index for index, name in enumerate(LOCAL_EVENT_STATE_NAMES)
}

RAW_PREFIX_NAMES = (
    "decision_time_seconds",
    "simultaneous_decision_count",
    "current_decision_index",
    "system_waiting_passengers",
    "system_in_vehicle_passengers",
)
RAW_SLOT_NAMES = (
    "active",
    *LOCAL_EVENT_STATE_NAMES,
    "prior_policy_action",
)
RAW_SLOT_WIDTH = len(RAW_SLOT_NAMES)
RAW_STATE_NAMES = RAW_PREFIX_NAMES + tuple(
    f"slot_{slot:02d}_{name}"
    for slot in range(MAX_SIMULTANEOUS_DECISIONS)
    for name in RAW_SLOT_NAMES
)
RAW_STATE_DIM = len(RAW_STATE_NAMES)
RAW_STATE_INDEX = {name: index for index, name in enumerate(RAW_STATE_NAMES)}

LOCAL_CANONICAL_STATE_NAMES = (
    "line_id",
    "station_progress",
    "forward_headway_ratio",
    "backward_headway_ratio",
    "target_forward_headway_scaled",
    "target_backward_headway_scaled",
    "eligible_stranded_capacity_ratio",
    "onboard_capacity_ratio",
    "eligible_demand_per_target_headway_capacity_ratio",
    "base_dwell_scaled",
    "forward_bus_present",
    "backward_bus_present",
)
LOCAL_CANONICAL_STATE_DIM = len(LOCAL_CANONICAL_STATE_NAMES)
LOCAL_CANONICAL_STATE_INDEX = {
    name: index for index, name in enumerate(LOCAL_CANONICAL_STATE_NAMES)
}
CANONICAL_PREFIX_NAMES = (
    "horizon_progress",
    "simultaneous_decision_count_scaled",
    "current_decision_index_scaled",
    "system_waiting_passengers_scaled",
    "system_in_vehicle_passengers_scaled",
)
CANONICAL_SLOT_NAMES = (
    "active",
    *LOCAL_CANONICAL_STATE_NAMES,
    "prior_policy_action",
)
CANONICAL_SLOT_WIDTH = len(CANONICAL_SLOT_NAMES)
CANONICAL_STATE_NAMES = CANONICAL_PREFIX_NAMES + tuple(
    f"slot_{slot:02d}_{name}"
    for slot in range(MAX_SIMULTANEOUS_DECISIONS)
    for name in CANONICAL_SLOT_NAMES
)
CANONICAL_STATE_DIM = len(CANONICAL_STATE_NAMES)
CANONICAL_STATE_INDEX = {
    name: index for index, name in enumerate(CANONICAL_STATE_NAMES)
}


@dataclass(frozen=True)
class TransitMDPConfigV7:
    """Physical constants and explicitly named normalization scales."""

    max_hold_seconds: float = 60.0
    horizon_seconds: float = 24_000.0
    scheduled_headway_scale_seconds: float = 720.0
    base_dwell_scale_seconds: float = 130.0
    system_passenger_count_scale: float = 20_000.0
    max_headway_ratio: float = 5.0
    max_queue_capacity_ratio: float = 2.0
    max_demand_capacity_ratio: float = 2.0
    max_system_passenger_count_scaled: float = 1.25
    waiting_time_value: float = 2.0
    in_vehicle_time_value: float = 1.0
    reward_unit_passenger_seconds: float = 1.0
    discount_gamma_per_minute: float = 1.0
    max_simultaneous_decisions: int = MAX_SIMULTANEOUS_DECISIONS

    def __post_init__(self) -> None:
        positive = (
            "max_hold_seconds",
            "horizon_seconds",
            "scheduled_headway_scale_seconds",
            "base_dwell_scale_seconds",
            "system_passenger_count_scale",
            "max_headway_ratio",
            "max_queue_capacity_ratio",
            "max_demand_capacity_ratio",
            "max_system_passenger_count_scaled",
            "waiting_time_value",
            "in_vehicle_time_value",
            "reward_unit_passenger_seconds",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.reward_unit_passenger_seconds) != 1.0:
            raise ValueError(
                "v7 stores reward in physical generalized passenger-seconds"
            )
        if float(self.discount_gamma_per_minute) != 1.0:
            raise ValueError("the finite-horizon v7 objective is undiscounted")
        if int(self.max_simultaneous_decisions) != MAX_SIMULTANEOUS_DECISIONS:
            raise ValueError(
                "max_simultaneous_decisions is schema-fixed at "
                f"{MAX_SIMULTANEOUS_DECISIONS}"
            )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_CONFIG_V7 = TransitMDPConfigV7()


def _as_event_batch(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != RAW_EVENT_STATE_DIM:
        raise ValueError(
            "event observations must have shape "
            f"({RAW_EVENT_STATE_DIM},) or (N, {RAW_EVENT_STATE_DIM}); "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("event observations contain non-finite values")
    return array, was_vector


def _as_raw_batch(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != RAW_STATE_DIM:
        raise ValueError(
            f"raw states must have width {RAW_STATE_DIM}; got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("raw states contain non-finite values")
    return array, was_vector


def _slot_start(slot: int) -> int:
    if not 0 <= int(slot) < MAX_SIMULTANEOUS_DECISIONS:
        raise IndexError(f"batch slot {slot} is outside the v7 schema")
    return len(RAW_PREFIX_NAMES) + int(slot) * RAW_SLOT_WIDTH


def _canonical_slot_start(slot: int) -> int:
    return len(CANONICAL_PREFIX_NAMES) + int(slot) * CANONICAL_SLOT_WIDTH


def _event_column(observations: np.ndarray, name: str) -> np.ndarray:
    return observations[:, RAW_EVENT_STATE_INDEX[name]]


def _validated_policy_actions(values: np.ndarray | float) -> np.ndarray:
    actions = np.asarray(values, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("policy actions must be finite")
    if np.any(actions < -1.000001) or np.any(actions > 1.000001):
        raise ValueError("policy actions must lie in [-1, 1]")
    return np.clip(actions, -1.0, 1.0)


def infer_station_max_v7(
    raw_event_observations: np.ndarray, n_lines: int = 12
) -> np.ndarray:
    """Infer each directional service's largest zero-based stop index."""

    observations, _ = _as_event_batch(raw_event_observations)
    maxima = np.ones(int(n_lines), dtype=np.float32)
    line_ids = np.rint(_event_column(observations, "line_id")).astype(np.int64)
    stations = _event_column(observations, "station_index")
    for line_id in range(int(n_lines)):
        selected = stations[line_ids == line_id]
        if selected.size:
            maxima[line_id] = max(float(np.max(selected)), 1.0)
    return maxima


def raw_event_to_local_canonical_v7(
    raw_event_observations: np.ndarray,
    station_max_by_line: Iterable[float],
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> np.ndarray:
    """Normalize one or more physical events without duplicating global state."""

    observations, was_vector = _as_event_batch(raw_event_observations)
    observations = observations.copy()
    for name in ("forward_headway_seconds", "backward_headway_seconds"):
        index = RAW_EVENT_STATE_INDEX[name]
        values = observations[:, index]
        if np.any(values < -HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS):
            raise ValueError(f"{name} is below the measurement-jitter tolerance")
        observations[values < 0.0, index] = 0.0

    station_max = np.asarray(tuple(station_max_by_line), dtype=np.float32)
    if station_max.ndim != 1 or station_max.size == 0:
        raise ValueError("station_max_by_line must be a non-empty vector")
    if not np.isfinite(station_max).all() or np.any(station_max <= 0.0):
        raise ValueError("station_max_by_line must contain finite positive values")
    line_ids = np.rint(_event_column(observations, "line_id")).astype(np.int64)
    if np.any(line_ids < 0) or np.any(line_ids >= station_max.size):
        raise ValueError("event observation contains an out-of-range line_id")

    target_forward = _event_column(
        observations, "target_forward_headway_seconds"
    )
    target_backward = _event_column(
        observations, "target_backward_headway_seconds"
    )
    if np.any(target_forward <= 0.0) or np.any(target_backward <= 0.0):
        raise ValueError("scheduled target headways must be strictly positive")
    capacity = _event_column(observations, "bus_capacity")
    if np.any(capacity <= 0.0):
        raise ValueError("bus capacity must be strictly positive")

    forward_present = np.clip(
        np.rint(_event_column(observations, "forward_bus_present")), 0.0, 1.0
    )
    backward_present = np.clip(
        np.rint(_event_column(observations, "backward_bus_present")), 0.0, 1.0
    )
    forward_ratio = np.where(
        forward_present > 0.5,
        _event_column(observations, "forward_headway_seconds") / target_forward,
        1.0,
    )
    backward_ratio = np.where(
        backward_present > 0.5,
        _event_column(observations, "backward_headway_seconds") / target_backward,
        1.0,
    )
    reference_headway = 0.5 * (target_forward + target_backward)
    demand_per_target_headway = (
        np.maximum(
            _event_column(observations, "eligible_arrival_rate_per_second"),
            0.0,
        )
        * reference_headway
        / capacity
    )

    canonical = np.column_stack(
        (
            line_ids.astype(np.float32),
            np.clip(
                _event_column(observations, "station_index")
                / station_max[line_ids],
                0.0,
                1.0,
            ),
            np.clip(forward_ratio, 0.0, config.max_headway_ratio),
            np.clip(backward_ratio, 0.0, config.max_headway_ratio),
            target_forward / config.scheduled_headway_scale_seconds,
            target_backward / config.scheduled_headway_scale_seconds,
            np.clip(
                np.maximum(
                    _event_column(
                        observations, "eligible_stranded_passengers"
                    ),
                    0.0,
                )
                / capacity,
                0.0,
                config.max_queue_capacity_ratio,
            ),
            np.clip(
                np.maximum(
                    _event_column(observations, "onboard_passengers"), 0.0
                )
                / capacity,
                0.0,
                1.0,
            ),
            np.clip(
                demand_per_target_headway,
                0.0,
                config.max_demand_capacity_ratio,
            ),
            np.clip(
                np.maximum(
                    _event_column(observations, "base_dwell_seconds"), 0.0
                )
                / config.base_dwell_scale_seconds,
                0.0,
                2.0,
            ),
            forward_present,
            backward_present,
        )
    ).astype(np.float32, copy=False)
    return canonical[0] if was_vector else canonical


def pack_event_batch_v7(
    event_observations: np.ndarray,
    current_decision_index: int,
    prior_policy_actions: np.ndarray | None = None,
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> np.ndarray:
    """Pack a simultaneous physical event batch without repeated global fields."""

    events, was_vector = _as_event_batch(event_observations)
    if was_vector:
        raise ValueError("event_observations must be a non-empty event batch")
    count = len(events)
    if not 1 <= count <= config.max_simultaneous_decisions:
        raise ValueError(
            f"simultaneous decision count {count} is outside [1, "
            f"{config.max_simultaneous_decisions}]"
        )
    current = int(current_decision_index)
    if not 0 <= current < count:
        raise ValueError("current_decision_index is outside the active batch")
    for name in GLOBAL_EVENT_STATE_NAMES:
        values = _event_column(events, name)
        if not np.all(values == values[0]):
            raise ValueError(f"simultaneous events disagree on {name}")

    prior = np.full(MAX_SIMULTANEOUS_DECISIONS, -1.0, dtype=np.float32)
    if prior_policy_actions is not None:
        supplied = _validated_policy_actions(prior_policy_actions).reshape(-1)
        if supplied.size not in (count, MAX_SIMULTANEOUS_DECISIONS):
            raise ValueError("prior action buffer has an invalid width")
        prior[: supplied.size] = supplied
    if np.any(prior[current:count] != -1.0):
        raise ValueError("future batch actions must remain at the no-hold sentinel")

    state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    state[RAW_STATE_INDEX["decision_time_seconds"]] = _event_column(
        events, "sim_time_seconds"
    )[0]
    state[RAW_STATE_INDEX["simultaneous_decision_count"]] = count
    state[RAW_STATE_INDEX["current_decision_index"]] = current
    state[RAW_STATE_INDEX["system_waiting_passengers"]] = _event_column(
        events, "system_waiting_passengers"
    )[0]
    state[RAW_STATE_INDEX["system_in_vehicle_passengers"]] = _event_column(
        events, "system_in_vehicle_passengers"
    )[0]
    local_indices = [RAW_EVENT_STATE_INDEX[name] for name in LOCAL_EVENT_STATE_NAMES]
    for slot, event in enumerate(events):
        start = _slot_start(slot)
        state[start] = 1.0
        state[start + 1 : start + 1 + LOCAL_EVENT_STATE_DIM] = event[
            local_indices
        ]
        state[start + RAW_SLOT_WIDTH - 1] = prior[slot]
    for slot in range(count, MAX_SIMULTANEOUS_DECISIONS):
        state[_slot_start(slot) + RAW_SLOT_WIDTH - 1] = -1.0
    return state


def event_batch_v7(raw_state: np.ndarray) -> np.ndarray:
    """Reconstruct active physical event observations from one packed state."""

    states, was_vector = _as_raw_batch(raw_state)
    if not was_vector:
        raise ValueError("event_batch_v7 accepts one state at a time")
    state = states[0]
    count = int(round(float(state[RAW_STATE_INDEX["simultaneous_decision_count"]])))
    if not 1 <= count <= MAX_SIMULTANEOUS_DECISIONS:
        raise ValueError("packed state has an invalid simultaneous decision count")
    events = np.zeros((count, RAW_EVENT_STATE_DIM), dtype=np.float32)
    events[:, RAW_EVENT_STATE_INDEX["sim_time_seconds"]] = state[
        RAW_STATE_INDEX["decision_time_seconds"]
    ]
    events[:, RAW_EVENT_STATE_INDEX["system_waiting_passengers"]] = state[
        RAW_STATE_INDEX["system_waiting_passengers"]
    ]
    events[:, RAW_EVENT_STATE_INDEX["system_in_vehicle_passengers"]] = state[
        RAW_STATE_INDEX["system_in_vehicle_passengers"]
    ]
    local_indices = [RAW_EVENT_STATE_INDEX[name] for name in LOCAL_EVENT_STATE_NAMES]
    for slot in range(count):
        start = _slot_start(slot)
        if state[start] != 1.0:
            raise ValueError("packed state has a missing active batch slot")
        events[slot, local_indices] = state[
            start + 1 : start + 1 + LOCAL_EVENT_STATE_DIM
        ]
    return events


def current_event_v7(raw_state: np.ndarray) -> np.ndarray:
    states, was_vector = _as_raw_batch(raw_state)
    if not was_vector:
        raise ValueError("current_event_v7 accepts one state at a time")
    state = states[0]
    current = int(round(float(state[RAW_STATE_INDEX["current_decision_index"]])))
    events = event_batch_v7(state)
    if not 0 <= current < len(events):
        raise ValueError("packed state has an invalid current decision index")
    return events[current].copy()


def raw_state_to_canonical_v7(
    raw_states: np.ndarray,
    station_max_by_line: Iterable[float],
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> np.ndarray:
    """Normalize deduplicated batch states while masking inactive slots."""

    states, was_vector = _as_raw_batch(raw_states)
    canonical = np.zeros((len(states), CANONICAL_STATE_DIM), dtype=np.float32)
    canonical[:, 0] = np.clip(
        states[:, RAW_STATE_INDEX["decision_time_seconds"]]
        / config.horizon_seconds,
        0.0,
        1.0,
    )
    canonical[:, 1] = np.clip(
        states[:, RAW_STATE_INDEX["simultaneous_decision_count"]]
        / config.max_simultaneous_decisions,
        0.0,
        1.0,
    )
    canonical[:, 2] = np.clip(
        states[:, RAW_STATE_INDEX["current_decision_index"]]
        / max(config.max_simultaneous_decisions - 1, 1),
        0.0,
        1.0,
    )
    canonical[:, 3] = np.clip(
        states[:, RAW_STATE_INDEX["system_waiting_passengers"]]
        / config.system_passenger_count_scale,
        0.0,
        config.max_system_passenger_count_scaled,
    )
    canonical[:, 4] = np.clip(
        states[:, RAW_STATE_INDEX["system_in_vehicle_passengers"]]
        / config.system_passenger_count_scale,
        0.0,
        config.max_system_passenger_count_scaled,
    )

    local_event_indices = [
        RAW_EVENT_STATE_INDEX[name] for name in LOCAL_EVENT_STATE_NAMES
    ]
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        raw_start = _slot_start(slot)
        canonical_start = _canonical_slot_start(slot)
        active = states[:, raw_start] > 0.5
        canonical[:, canonical_start] = active.astype(np.float32)
        if np.any(active):
            events = np.zeros((int(np.sum(active)), RAW_EVENT_STATE_DIM), np.float32)
            events[:, RAW_EVENT_STATE_INDEX["sim_time_seconds"]] = states[
                active, RAW_STATE_INDEX["decision_time_seconds"]
            ]
            events[:, RAW_EVENT_STATE_INDEX["system_waiting_passengers"]] = states[
                active, RAW_STATE_INDEX["system_waiting_passengers"]
            ]
            events[
                :, RAW_EVENT_STATE_INDEX["system_in_vehicle_passengers"]
            ] = states[
                active, RAW_STATE_INDEX["system_in_vehicle_passengers"]
            ]
            events[:, local_event_indices] = states[
                active,
                raw_start + 1 : raw_start + 1 + LOCAL_EVENT_STATE_DIM,
            ]
            canonical[
                active,
                canonical_start
                + 1 : canonical_start
                + 1
                + LOCAL_CANONICAL_STATE_DIM,
            ] = raw_event_to_local_canonical_v7(
                events, station_max_by_line, config
            )
        canonical[:, canonical_start + CANONICAL_SLOT_WIDTH - 1] = np.clip(
            states[:, raw_start + RAW_SLOT_WIDTH - 1], -1.0, 1.0
        )
    return canonical[0] if was_vector else canonical


def terminal_raw_state_v7(
    previous_raw_state: np.ndarray,
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
) -> np.ndarray:
    """Construct an absorbing endpoint with one terminal clock and count pair."""

    state = np.asarray(previous_raw_state, dtype=np.float32).reshape(-1).copy()
    if state.shape != (RAW_STATE_DIM,):
        raise ValueError("previous_raw_state has the wrong v7 width")
    end = passenger_time_snapshot_v5(passenger_time_end)
    state[RAW_STATE_INDEX["decision_time_seconds"]] = end.time_seconds
    state[RAW_STATE_INDEX["system_waiting_passengers"]] = end.waiting_passengers
    state[
        RAW_STATE_INDEX["system_in_vehicle_passengers"]
    ] = end.in_vehicle_passengers
    return state


def transition_duration_seconds_v7(
    raw_states: np.ndarray, raw_next_states: np.ndarray
) -> np.ndarray | float:
    states, was_vector = _as_raw_batch(raw_states)
    next_states, next_was_vector = _as_raw_batch(raw_next_states)
    if was_vector != next_was_vector or states.shape != next_states.shape:
        raise ValueError("v7 raw state batches must have matching shapes")
    duration = (
        next_states[:, RAW_STATE_INDEX["decision_time_seconds"]]
        - states[:, RAW_STATE_INDEX["decision_time_seconds"]]
    )
    if np.any(duration < 0.0):
        raise ValueError("chronological v7 transitions cannot move backwards")
    duration = duration.astype(np.float32, copy=False)
    return float(duration[0]) if was_vector else duration


def continuous_time_discount_v7(
    delta_t_seconds: np.ndarray | float,
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> np.ndarray | float:
    duration = np.asarray(delta_t_seconds, dtype=np.float32)
    if not np.isfinite(duration).all() or np.any(duration < 0.0):
        raise ValueError("delta_t_seconds must be finite and non-negative")
    discount = np.power(
        config.discount_gamma_per_minute,
        duration / 60.0,
        dtype=np.float32,
    )
    return float(discount) if discount.ndim == 0 else discount


def policy_action_to_hold_seconds_v7(
    policy_action: np.ndarray | float,
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> np.ndarray | float:
    action = _validated_policy_actions(policy_action)
    hold = 0.5 * (action + 1.0) * config.max_hold_seconds
    return float(hold) if hold.ndim == 0 else hold


def hold_seconds_to_policy_action_v7(
    hold_seconds: np.ndarray | float,
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> np.ndarray | float:
    hold = np.asarray(hold_seconds, dtype=np.float32)
    if not np.isfinite(hold).all():
        raise ValueError("hold_seconds must be finite")
    if np.any(hold < -1e-6) or np.any(hold > config.max_hold_seconds + 1e-6):
        raise ValueError(
            f"hold_seconds must lie in [0, {config.max_hold_seconds}]"
        )
    action = 2.0 * np.clip(hold, 0.0, config.max_hold_seconds) / config.max_hold_seconds - 1.0
    return float(action) if action.ndim == 0 else action


def transition_cost_components_v7(
    passenger_time_start: PassengerTimeSnapshot | Mapping[str, float | int],
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> dict[str, float]:
    """Return exact physical passenger-time accrued between decision epochs."""

    return generalized_passenger_time_delta(
        passenger_time_snapshot_v5(passenger_time_start),
        passenger_time_snapshot_v5(passenger_time_end),
        waiting_time_value=config.waiting_time_value,
        in_vehicle_time_value=config.in_vehicle_time_value,
        reward_unit_passenger_seconds=config.reward_unit_passenger_seconds,
    )


def transition_reward_v7(
    passenger_time_start: PassengerTimeSnapshot | Mapping[str, float | int],
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
    config: TransitMDPConfigV7 = DEFAULT_CONFIG_V7,
) -> float:
    """Return negative generalized passenger-seconds without training scaling."""

    return -transition_cost_components_v7(
        passenger_time_start, passenger_time_end, config
    )["total_cost"]


__all__ = [
    "ACTION_DIM",
    "CANONICAL_PREFIX_NAMES",
    "CANONICAL_SLOT_NAMES",
    "CANONICAL_SLOT_WIDTH",
    "CANONICAL_STATE_DIM",
    "CANONICAL_STATE_INDEX",
    "CANONICAL_STATE_NAMES",
    "DEFAULT_CONFIG_V7",
    "GLOBAL_EVENT_STATE_NAMES",
    "HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS",
    "LOCAL_CANONICAL_STATE_DIM",
    "LOCAL_CANONICAL_STATE_INDEX",
    "LOCAL_CANONICAL_STATE_NAMES",
    "LOCAL_EVENT_STATE_DIM",
    "LOCAL_EVENT_STATE_INDEX",
    "LOCAL_EVENT_STATE_NAMES",
    "MAX_SIMULTANEOUS_DECISIONS",
    "RAW_EVENT_STATE_DIM",
    "RAW_EVENT_STATE_INDEX",
    "RAW_EVENT_STATE_NAMES",
    "RAW_PREFIX_NAMES",
    "RAW_SLOT_NAMES",
    "RAW_SLOT_WIDTH",
    "RAW_STATE_DIM",
    "RAW_STATE_INDEX",
    "RAW_STATE_NAMES",
    "SCHEMA_VERSION",
    "TransitMDPConfigV7",
    "continuous_time_discount_v7",
    "current_event_v7",
    "event_batch_v7",
    "hold_seconds_to_policy_action_v7",
    "infer_station_max_v7",
    "pack_event_batch_v7",
    "policy_action_to_hold_seconds_v7",
    "raw_event_to_local_canonical_v7",
    "raw_state_to_canonical_v7",
    "terminal_raw_state_v7",
    "transition_cost_components_v7",
    "transition_duration_seconds_v7",
    "transition_reward_v7",
]
