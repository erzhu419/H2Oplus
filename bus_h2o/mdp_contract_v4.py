"""Passenger-welfare transit semi-Markov decision-process contract.

Version 4 replaces the proxy headway reward used by v3 with an executable
generalized passenger-time objective.  Raw observations retain physical units
for auditability; canonical observations are normalized exactly once here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "transit-smdp-v4"
RAW_EVENT_STATE_NAMES = (
    "line_id",
    "station_index",
    "sim_time_seconds",
    "forward_headway_seconds",
    "backward_headway_seconds",
    "target_forward_headway_seconds",
    "target_backward_headway_seconds",
    "eligible_stranded_passengers",
    "onboard_passengers",
    "eligible_arrival_rate_per_second",
    "base_dwell_seconds",
    "forward_bus_present",
    "backward_bus_present",
    "bus_capacity",
)
RAW_EVENT_STATE_DIM = len(RAW_EVENT_STATE_NAMES)
RAW_EVENT_STATE_INDEX = {
    name: index for index, name in enumerate(RAW_EVENT_STATE_NAMES)
}

CANONICAL_STATE_NAMES = (
    "line_id",
    "station_progress",
    "episode_progress",
    "forward_headway_ratio",
    "backward_headway_ratio",
    "target_forward_headway_fraction",
    "target_backward_headway_fraction",
    "eligible_stranded_fraction",
    "onboard_fraction",
    "eligible_demand_per_headway_fraction",
    "base_dwell_fraction",
    "forward_bus_present",
    "backward_bus_present",
)
CANONICAL_STATE_DIM = len(CANONICAL_STATE_NAMES)
CANONICAL_STATE_INDEX = {
    name: index for index, name in enumerate(CANONICAL_STATE_NAMES)
}
ACTION_DIM = 1
HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class TransitMDPConfigV4:
    """Physical scales and generalized passenger-time values for v4."""

    max_hold_seconds: float = 60.0
    episode_seconds: float = 24_000.0
    nominal_bus_capacity: float = 50.0
    max_scheduled_headway_seconds: float = 720.0
    max_base_dwell_seconds: float = 130.0
    max_headway_ratio: float = 5.0
    max_queue_capacity_ratio: float = 2.0
    max_demand_capacity_ratio: float = 2.0
    waiting_time_value: float = 2.0
    in_vehicle_time_value: float = 1.0
    reward_unit_passenger_seconds: float = 60.0
    discount_gamma_per_minute: float = 0.99

    def __post_init__(self) -> None:
        positive = (
            "max_hold_seconds",
            "episode_seconds",
            "nominal_bus_capacity",
            "max_scheduled_headway_seconds",
            "max_base_dwell_seconds",
            "max_headway_ratio",
            "max_queue_capacity_ratio",
            "max_demand_capacity_ratio",
            "waiting_time_value",
            "in_vehicle_time_value",
            "reward_unit_passenger_seconds",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.discount_gamma_per_minute <= 1.0:
            raise ValueError("discount_gamma_per_minute must lie in (0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_CONFIG_V4 = TransitMDPConfigV4()


def _as_batch(values: np.ndarray, width: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{name} must have shape ({width},) or (N, {width}); got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array, was_vector


def _column(observations: np.ndarray, name: str) -> np.ndarray:
    return observations[:, RAW_EVENT_STATE_INDEX[name]]


def correct_headway_measurement_jitter_v4(
    raw_observations: np.ndarray,
    tolerance_seconds: float = HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS,
) -> np.ndarray:
    """Correct sub-step negative headways caused by event ordering.

    SUMO advances this scenario in one-second steps.  Two stop events in the
    same step can therefore produce a headway just below zero.  Values within
    one simulation step are simultaneous at the model's measurement
    resolution and are mapped to zero.  Larger negative values indicate a
    broken event ordering or state extraction and are rejected.
    """

    observations, was_vector = _as_batch(
        raw_observations, RAW_EVENT_STATE_DIM, "raw_observations"
    )
    tolerance = float(tolerance_seconds)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_seconds must be finite and non-negative")

    corrected = observations
    for name in ("forward_headway_seconds", "backward_headway_seconds"):
        index = RAW_EVENT_STATE_INDEX[name]
        values = corrected[:, index]
        if np.any(values < -tolerance):
            minimum = float(np.min(values))
            raise ValueError(
                f"{name} contains {minimum} s, below the {-tolerance} s "
                "measurement-jitter tolerance"
            )
        negative = values < 0.0
        if np.any(negative):
            if corrected is observations:
                corrected = observations.copy()
            corrected[negative, index] = 0.0
    return corrected[0] if was_vector else corrected


def infer_station_max_v4(
    raw_observations: np.ndarray, n_lines: int = 12
) -> np.ndarray:
    """Infer each directional line's largest zero-based station index."""

    observations, _ = _as_batch(
        raw_observations, RAW_EVENT_STATE_DIM, "raw_observations"
    )
    maxima = np.ones(n_lines, dtype=np.float32)
    line_ids = np.rint(_column(observations, "line_id")).astype(np.int64)
    stations = _column(observations, "station_index")
    for line_id in range(n_lines):
        selected = stations[line_ids == line_id]
        if selected.size:
            maxima[line_id] = max(float(np.max(selected)), 1.0)
    return maxima


def raw_event_to_canonical_v4(
    raw_observations: np.ndarray,
    station_max_by_line: Iterable[float],
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray:
    """Normalize physical event observations into the documented v4 state."""

    observations, was_vector = _as_batch(
        raw_observations, RAW_EVENT_STATE_DIM, "raw_observations"
    )
    observations = np.asarray(
        correct_headway_measurement_jitter_v4(observations), dtype=np.float32
    )
    station_max = np.asarray(tuple(station_max_by_line), dtype=np.float32)
    if station_max.ndim != 1 or station_max.size == 0 or np.any(station_max <= 0):
        raise ValueError("station_max_by_line must be a non-empty positive vector")

    line_ids = np.rint(_column(observations, "line_id")).astype(np.int64)
    if np.any(line_ids < 0) or np.any(line_ids >= station_max.size):
        raise ValueError("raw observation contains an out-of-range line_id")

    target_forward = np.maximum(
        _column(observations, "target_forward_headway_seconds"), 1.0
    )
    target_backward = np.maximum(
        _column(observations, "target_backward_headway_seconds"), 1.0
    )
    reference_headway = 0.5 * (target_forward + target_backward)
    capacity = np.maximum(_column(observations, "bus_capacity"), 1.0)
    forward_present = np.clip(
        np.rint(_column(observations, "forward_bus_present")), 0.0, 1.0
    )
    backward_present = np.clip(
        np.rint(_column(observations, "backward_bus_present")), 0.0, 1.0
    )

    forward_ratio = np.where(
        forward_present > 0.5,
        _column(observations, "forward_headway_seconds") / target_forward,
        1.0,
    )
    backward_ratio = np.where(
        backward_present > 0.5,
        _column(observations, "backward_headway_seconds") / target_backward,
        1.0,
    )
    demand_per_headway = (
        np.maximum(_column(observations, "eligible_arrival_rate_per_second"), 0.0)
        * reference_headway
        / capacity
    )

    state = np.column_stack(
        (
            line_ids.astype(np.float32),
            np.clip(
                _column(observations, "station_index") / station_max[line_ids],
                0.0,
                1.0,
            ),
            np.clip(
                _column(observations, "sim_time_seconds") / config.episode_seconds,
                0.0,
                1.0,
            ),
            np.clip(forward_ratio, 0.0, config.max_headway_ratio),
            np.clip(backward_ratio, 0.0, config.max_headway_ratio),
            target_forward / config.max_scheduled_headway_seconds,
            target_backward / config.max_scheduled_headway_seconds,
            np.clip(
                np.maximum(_column(observations, "eligible_stranded_passengers"), 0.0)
                / capacity,
                0.0,
                config.max_queue_capacity_ratio,
            ),
            np.clip(
                np.maximum(_column(observations, "onboard_passengers"), 0.0)
                / capacity,
                0.0,
                1.0,
            ),
            np.clip(
                demand_per_headway, 0.0, config.max_demand_capacity_ratio
            ),
            np.clip(
                np.maximum(_column(observations, "base_dwell_seconds"), 0.0)
                / config.max_base_dwell_seconds,
                0.0,
                2.0,
            ),
            forward_present,
            backward_present,
        )
    ).astype(np.float32, copy=False)
    return state[0] if was_vector else state


def policy_action_to_hold_seconds_v4(
    policy_action: np.ndarray | float,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray | float:
    """Map the one-dimensional action in [-1, 1] to holding seconds."""

    action = np.asarray(policy_action, dtype=np.float32)
    hold = 0.5 * (np.clip(action, -1.0, 1.0) + 1.0) * config.max_hold_seconds
    return float(hold) if hold.ndim == 0 else hold


def hold_seconds_to_policy_action_v4(
    hold_seconds: np.ndarray | float,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray | float:
    """Map physical holding seconds to the one-dimensional action domain."""

    hold = np.asarray(hold_seconds, dtype=np.float32)
    action = 2.0 * np.clip(hold, 0.0, config.max_hold_seconds) / config.max_hold_seconds - 1.0
    return float(action) if action.ndim == 0 else action


def transition_cost_components_v4(
    raw_observations: np.ndarray,
    policy_actions: np.ndarray,
    raw_next_observations: np.ndarray,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
    *,
    correct_measurement_jitter: bool = True,
) -> dict[str, np.ndarray | float]:
    """Return separately auditable waiting and in-vehicle holding costs.

    For random passenger arrivals, excess waiting over regular headways is
    proportional to squared headway deviations.  Each interval appears in two
    adjacent bus events, hence the factor 1/4.  The holding term is the added
    in-vehicle passenger time.  Both physical passenger-second terms are
    divided by one global unit; the default therefore reports generalized
    passenger-minutes without changing relative welfare weights.
    """

    observations, was_vector = _as_batch(
        raw_observations, RAW_EVENT_STATE_DIM, "raw_observations"
    )
    next_observations, next_was_vector = _as_batch(
        raw_next_observations, RAW_EVENT_STATE_DIM, "raw_next_observations"
    )
    if was_vector != next_was_vector or observations.shape != next_observations.shape:
        raise ValueError("raw observation batches must have matching shapes")
    if correct_measurement_jitter:
        observations = np.asarray(
            correct_headway_measurement_jitter_v4(observations), dtype=np.float32
        )
        next_observations = np.asarray(
            correct_headway_measurement_jitter_v4(next_observations), dtype=np.float32
        )

    actions = np.asarray(policy_actions, dtype=np.float32)
    if actions.ndim == 0:
        actions = actions.reshape(1, 1)
    elif actions.ndim == 1:
        actions = actions[:, None] if actions.size == len(observations) else actions[None, :]
    if actions.shape != (len(observations), ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError(
            f"policy_actions must be finite with shape ({len(observations)}, {ACTION_DIM})"
        )

    next_target_forward = np.maximum(
        _column(next_observations, "target_forward_headway_seconds"), 1.0
    )
    next_target_backward = np.maximum(
        _column(next_observations, "target_backward_headway_seconds"), 1.0
    )
    next_arrival_rate = np.maximum(
        _column(next_observations, "eligible_arrival_rate_per_second"), 0.0
    )
    forward_present = np.clip(
        np.rint(_column(next_observations, "forward_bus_present")), 0.0, 1.0
    )
    backward_present = np.clip(
        np.rint(_column(next_observations, "backward_bus_present")), 0.0, 1.0
    )
    forward_error = (
        _column(next_observations, "forward_headway_seconds")
        - next_target_forward
    )
    backward_error = (
        _column(next_observations, "backward_headway_seconds")
        - next_target_backward
    )
    irregularity = (
        forward_present * np.square(forward_error)
        + backward_present * np.square(backward_error)
    )
    waiting_cost = (
        config.waiting_time_value
        * next_arrival_rate
        * irregularity
        / (4.0 * config.reward_unit_passenger_seconds)
    )

    onboard = np.maximum(_column(observations, "onboard_passengers"), 0.0)
    hold_seconds = np.asarray(
        policy_action_to_hold_seconds_v4(actions[:, 0], config), dtype=np.float32
    )
    holding_cost = (
        config.in_vehicle_time_value
        * onboard
        * hold_seconds
        / config.reward_unit_passenger_seconds
    )
    total_cost = waiting_cost + holding_cost

    result = {
        "waiting_cost": waiting_cost.astype(np.float32, copy=False),
        "holding_cost": holding_cost.astype(np.float32, copy=False),
        "total_cost": total_cost.astype(np.float32, copy=False),
    }
    if was_vector:
        return {name: float(values[0]) for name, values in result.items()}
    return result


def transition_reward_v4(
    raw_observations: np.ndarray,
    policy_actions: np.ndarray,
    raw_next_observations: np.ndarray,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
    *,
    correct_measurement_jitter: bool = True,
) -> np.ndarray | float:
    """Return negative generalized passenger-time cost for a transition."""

    components = transition_cost_components_v4(
        raw_observations,
        policy_actions,
        raw_next_observations,
        config,
        correct_measurement_jitter=correct_measurement_jitter,
    )
    total = components["total_cost"]
    if np.isscalar(total):
        return -float(total)
    return -np.asarray(total, dtype=np.float32)


def transition_duration_seconds_v4(
    raw_observations: np.ndarray, raw_next_observations: np.ndarray
) -> np.ndarray | float:
    """Return the same-bus inter-event duration from physical event states."""

    observations, was_vector = _as_batch(
        raw_observations, RAW_EVENT_STATE_DIM, "raw_observations"
    )
    next_observations, next_was_vector = _as_batch(
        raw_next_observations, RAW_EVENT_STATE_DIM, "raw_next_observations"
    )
    if was_vector != next_was_vector or observations.shape != next_observations.shape:
        raise ValueError("raw observation batches must have matching shapes")
    duration = (
        _column(next_observations, "sim_time_seconds")
        - _column(observations, "sim_time_seconds")
    )
    if np.any(duration <= 0.0):
        raise ValueError("same-bus transition duration must be strictly positive")
    duration = duration.astype(np.float32, copy=False)
    return float(duration[0]) if was_vector else duration


def continuous_time_discount_v4(
    duration_seconds: np.ndarray | float,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray | float:
    """Discount a variable-duration event with a per-minute base gamma."""

    duration = np.asarray(duration_seconds, dtype=np.float32)
    if not np.isfinite(duration).all() or np.any(duration < 0.0):
        raise ValueError("duration_seconds must be finite and non-negative")
    discount = np.power(
        config.discount_gamma_per_minute, duration / 60.0, dtype=np.float32
    )
    return float(discount) if discount.ndim == 0 else discount


__all__ = [
    "ACTION_DIM",
    "CANONICAL_STATE_DIM",
    "CANONICAL_STATE_INDEX",
    "CANONICAL_STATE_NAMES",
    "DEFAULT_CONFIG_V4",
    "HEADWAY_MEASUREMENT_JITTER_TOLERANCE_SECONDS",
    "RAW_EVENT_STATE_DIM",
    "RAW_EVENT_STATE_INDEX",
    "RAW_EVENT_STATE_NAMES",
    "SCHEMA_VERSION",
    "TransitMDPConfigV4",
    "continuous_time_discount_v4",
    "correct_headway_measurement_jitter_v4",
    "hold_seconds_to_policy_action_v4",
    "infer_station_max_v4",
    "policy_action_to_hold_seconds_v4",
    "raw_event_to_canonical_v4",
    "transition_cost_components_v4",
    "transition_duration_seconds_v4",
    "transition_reward_v4",
]
