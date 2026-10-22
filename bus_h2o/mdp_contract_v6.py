"""Chronological, non-overlapping transit welfare SMDP contract.

Version 6 serializes each simultaneous SUMO decision batch as an
agent-environment cycle. Every token observes the complete fixed-width batch
and all actions already selected in that batch. Physical time advances only
after the final token, so system passenger time appears in exactly one row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from bus_h2o.mdp_contract_v4 import ACTION_DIM
from bus_h2o.mdp_contract_v5 import (
    CANONICAL_STATE_NAMES as EVENT_CANONICAL_STATE_NAMES,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV5,
    continuous_time_discount_v5,
    hold_seconds_to_policy_action_v5,
    passenger_time_snapshot_v5,
    policy_action_to_hold_seconds_v5,
    raw_event_to_canonical_v5,
    transition_cost_components_v5,
    transition_reward_v5,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot


SCHEMA_VERSION = "transit-smdp-v6"
MAX_SIMULTANEOUS_DECISIONS = 16
RAW_PREFIX_NAMES = (
    "decision_time_seconds",
    "simultaneous_decision_count",
    "current_decision_index",
)
RAW_SLOT_NAMES = (
    "active",
    *RAW_EVENT_STATE_NAMES,
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

CANONICAL_PREFIX_NAMES = (
    "episode_progress",
    "simultaneous_decision_fraction",
    "current_decision_fraction",
)
CANONICAL_SLOT_NAMES = (
    "active",
    *EVENT_CANONICAL_STATE_NAMES,
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
class TransitMDPConfigV6(TransitMDPConfigV5):
    """Physical, welfare, and fixed-batch constants for v6."""

    # The operational objective is finite-horizon, undiscounted generalized
    # passenger time. The larger unit changes only numerical scale.
    reward_unit_passenger_seconds: float = 1_000_000.0
    discount_gamma_per_minute: float = 1.0
    max_simultaneous_decisions: int = MAX_SIMULTANEOUS_DECISIONS

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.reward_unit_passenger_seconds) != 1_000_000.0:
            raise ValueError(
                "v6 reward_unit_passenger_seconds is objective-fixed at 1,000,000"
            )
        if float(self.discount_gamma_per_minute) != 1.0:
            raise ValueError(
                "v6 discount_gamma_per_minute is objective-fixed at 1"
            )
        if int(self.max_simultaneous_decisions) != MAX_SIMULTANEOUS_DECISIONS:
            raise ValueError(
                "max_simultaneous_decisions is schema-fixed at "
                f"{MAX_SIMULTANEOUS_DECISIONS}"
            )


DEFAULT_CONFIG_V6 = TransitMDPConfigV6()


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
        raise IndexError(f"batch slot {slot} is outside the v6 schema")
    return len(RAW_PREFIX_NAMES) + int(slot) * RAW_SLOT_WIDTH


def _canonical_slot_start(slot: int) -> int:
    return len(CANONICAL_PREFIX_NAMES) + int(slot) * CANONICAL_SLOT_WIDTH


def pack_event_batch_v6(
    event_observations: np.ndarray,
    current_decision_index: int,
    prior_policy_actions: np.ndarray | None = None,
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> np.ndarray:
    """Pack one ordered simultaneous event batch into an AEC decision state."""

    events = np.asarray(event_observations, dtype=np.float32)
    if events.ndim != 2 or events.shape[1] != RAW_EVENT_STATE_DIM:
        raise ValueError(
            "event_observations must have shape "
            f"(N, {RAW_EVENT_STATE_DIM}); got {events.shape}"
        )
    count = len(events)
    if not 1 <= count <= config.max_simultaneous_decisions:
        raise ValueError(
            f"simultaneous decision count {count} is outside [1, "
            f"{config.max_simultaneous_decisions}]"
        )
    current = int(current_decision_index)
    if not 0 <= current < count:
        raise ValueError("current_decision_index is outside the active batch")
    if not np.isfinite(events).all():
        raise ValueError("event batch contains non-finite values")
    times = events[:, RAW_EVENT_STATE_INDEX["sim_time_seconds"]]
    if not np.all(times == times[0]):
        raise ValueError("a v6 simultaneous batch must share one SUMO time")
    for name in ("system_waiting_passengers", "system_in_vehicle_passengers"):
        values = events[:, RAW_EVENT_STATE_INDEX[name]]
        if not np.all(values == values[0]):
            raise ValueError(f"simultaneous events disagree on {name}")

    prior = np.full(MAX_SIMULTANEOUS_DECISIONS, -1.0, dtype=np.float32)
    if prior_policy_actions is not None:
        supplied = np.asarray(prior_policy_actions, dtype=np.float32).reshape(-1)
        if supplied.size not in (count, MAX_SIMULTANEOUS_DECISIONS):
            raise ValueError("prior action buffer has an invalid width")
        if not np.isfinite(supplied).all() or np.any(np.abs(supplied) > 1.00001):
            raise ValueError("prior action buffer contains an invalid action")
        prior[: supplied.size] = supplied
    if np.any(prior[current:count] != -1.0):
        raise ValueError("future batch actions must remain at the no-hold sentinel")

    state = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    state[RAW_STATE_INDEX["decision_time_seconds"]] = times[0]
    state[RAW_STATE_INDEX["simultaneous_decision_count"]] = count
    state[RAW_STATE_INDEX["current_decision_index"]] = current
    for slot, event in enumerate(events):
        start = _slot_start(slot)
        state[start] = 1.0
        state[start + 1 : start + 1 + RAW_EVENT_STATE_DIM] = event
        state[start + RAW_SLOT_WIDTH - 1] = prior[slot]
    for slot in range(count, MAX_SIMULTANEOUS_DECISIONS):
        state[_slot_start(slot) + RAW_SLOT_WIDTH - 1] = -1.0
    return state


def event_batch_v6(raw_state: np.ndarray) -> np.ndarray:
    """Return the active local event observations from one packed state."""

    states, was_vector = _as_raw_batch(raw_state)
    if not was_vector:
        raise ValueError("event_batch_v6 accepts one state at a time")
    state = states[0]
    count = int(round(float(state[RAW_STATE_INDEX["simultaneous_decision_count"]])))
    if not 1 <= count <= MAX_SIMULTANEOUS_DECISIONS:
        raise ValueError("packed state has an invalid simultaneous decision count")
    events = []
    for slot in range(count):
        start = _slot_start(slot)
        if state[start] != 1.0:
            raise ValueError("packed state has a missing active batch slot")
        events.append(state[start + 1 : start + 1 + RAW_EVENT_STATE_DIM])
    return np.stack(events)


def current_event_v6(raw_state: np.ndarray) -> np.ndarray:
    states, was_vector = _as_raw_batch(raw_state)
    if not was_vector:
        raise ValueError("current_event_v6 accepts one state at a time")
    state = states[0]
    current = int(round(float(state[RAW_STATE_INDEX["current_decision_index"]])))
    start = _slot_start(current)
    if state[start] != 1.0:
        raise ValueError("current v6 decision slot is inactive")
    return state[start + 1 : start + 1 + RAW_EVENT_STATE_DIM].copy()


def raw_state_to_canonical_v6(
    raw_states: np.ndarray,
    station_max_by_line: Iterable[float],
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> np.ndarray:
    """Normalize packed AEC states while preserving inactive zero slots."""

    states, was_vector = _as_raw_batch(raw_states)
    canonical = np.zeros((len(states), CANONICAL_STATE_DIM), dtype=np.float32)
    canonical[:, 0] = np.clip(states[:, 0] / config.episode_seconds, 0.0, 1.0)
    canonical[:, 1] = np.clip(
        states[:, 1] / config.max_simultaneous_decisions, 0.0, 1.0
    )
    canonical[:, 2] = np.clip(
        states[:, 2] / max(config.max_simultaneous_decisions - 1, 1),
        0.0,
        1.0,
    )
    for slot in range(MAX_SIMULTANEOUS_DECISIONS):
        raw_start = _slot_start(slot)
        canonical_start = _canonical_slot_start(slot)
        active = states[:, raw_start] > 0.5
        canonical[:, canonical_start] = active.astype(np.float32)
        if np.any(active):
            local = states[
                active,
                raw_start + 1 : raw_start + 1 + RAW_EVENT_STATE_DIM,
            ]
            canonical[
                active,
                canonical_start + 1 : canonical_start + 1 + len(
                    EVENT_CANONICAL_STATE_NAMES
                ),
            ] = raw_event_to_canonical_v5(local, station_max_by_line, config)
        canonical[:, canonical_start + CANONICAL_SLOT_WIDTH - 1] = np.clip(
            states[:, raw_start + RAW_SLOT_WIDTH - 1], -1.0, 1.0
        )
    return canonical[0] if was_vector else canonical


def terminal_raw_state_v6(
    previous_raw_state: np.ndarray,
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
) -> np.ndarray:
    """Construct an absorbing endpoint with an auditable terminal clock."""

    state = np.asarray(previous_raw_state, dtype=np.float32).reshape(-1).copy()
    if state.shape != (RAW_STATE_DIM,):
        raise ValueError("previous_raw_state has the wrong v6 width")
    end = passenger_time_snapshot_v5(passenger_time_end)
    state[RAW_STATE_INDEX["decision_time_seconds"]] = end.time_seconds
    count = int(round(float(state[RAW_STATE_INDEX["simultaneous_decision_count"]])))
    for slot in range(count):
        start = _slot_start(slot) + 1
        state[start + RAW_EVENT_STATE_INDEX["sim_time_seconds"]] = end.time_seconds
        state[
            start + RAW_EVENT_STATE_INDEX["system_waiting_passengers"]
        ] = end.waiting_passengers
        state[
            start + RAW_EVENT_STATE_INDEX["system_in_vehicle_passengers"]
        ] = end.in_vehicle_passengers
    return state


def transition_duration_seconds_v6(
    raw_states: np.ndarray, raw_next_states: np.ndarray
) -> np.ndarray | float:
    states, was_vector = _as_raw_batch(raw_states)
    next_states, next_was_vector = _as_raw_batch(raw_next_states)
    if was_vector != next_was_vector or states.shape != next_states.shape:
        raise ValueError("v6 raw state batches must have matching shapes")
    duration = next_states[:, 0] - states[:, 0]
    if np.any(duration < 0.0):
        raise ValueError("chronological v6 transitions cannot move backwards")
    duration = duration.astype(np.float32, copy=False)
    return float(duration[0]) if was_vector else duration


def continuous_time_discount_v6(
    delta_t_seconds: np.ndarray | float,
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> np.ndarray | float:
    return continuous_time_discount_v5(delta_t_seconds, config)


def policy_action_to_hold_seconds_v6(
    policy_action: np.ndarray | float,
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> np.ndarray | float:
    return policy_action_to_hold_seconds_v5(policy_action, config)


def hold_seconds_to_policy_action_v6(
    hold_seconds: np.ndarray | float,
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> np.ndarray | float:
    return hold_seconds_to_policy_action_v5(hold_seconds, config)


def transition_cost_components_v6(
    passenger_time_start: PassengerTimeSnapshot | Mapping[str, float | int],
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> dict[str, float]:
    return transition_cost_components_v5(
        passenger_time_start, passenger_time_end, config
    )


def transition_reward_v6(
    passenger_time_start: PassengerTimeSnapshot | Mapping[str, float | int],
    passenger_time_end: PassengerTimeSnapshot | Mapping[str, float | int],
    config: TransitMDPConfigV6 = DEFAULT_CONFIG_V6,
) -> float:
    return transition_reward_v5(passenger_time_start, passenger_time_end, config)


__all__ = [
    "ACTION_DIM",
    "CANONICAL_STATE_DIM",
    "CANONICAL_STATE_INDEX",
    "CANONICAL_STATE_NAMES",
    "DEFAULT_CONFIG_V6",
    "MAX_SIMULTANEOUS_DECISIONS",
    "RAW_EVENT_STATE_DIM",
    "RAW_EVENT_STATE_INDEX",
    "RAW_EVENT_STATE_NAMES",
    "RAW_STATE_DIM",
    "RAW_STATE_INDEX",
    "RAW_STATE_NAMES",
    "SCHEMA_VERSION",
    "TransitMDPConfigV6",
    "continuous_time_discount_v6",
    "current_event_v6",
    "event_batch_v6",
    "hold_seconds_to_policy_action_v6",
    "pack_event_batch_v6",
    "passenger_time_snapshot_v5",
    "policy_action_to_hold_seconds_v6",
    "raw_state_to_canonical_v6",
    "terminal_raw_state_v6",
    "transition_cost_components_v6",
    "transition_duration_seconds_v6",
    "transition_reward_v6",
]
