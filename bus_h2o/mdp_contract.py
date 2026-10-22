"""Canonical transit semi-Markov decision-process contract.

The v2 project mixed several incompatible meanings for state, action and
reward.  This module is the single executable definition used by conversion,
training, online rollout and evaluation in the rebuilt pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "transit-smdp-v3"
RAW_V2_OBS_DIM = 17
CANONICAL_STATE_NAMES = (
    "line_id",
    "station_progress",
    "episode_progress",
    "forward_headway_ratio",
    "backward_headway_ratio",
    "target_forward_headway_ratio",
    "waiting_passenger_fraction",
    "base_dwell_fraction",
    "co_line_forward_headway_ratio",
    "co_line_backward_headway_ratio",
    "scheduled_headway_fraction",
)
CANONICAL_STATE_DIM = len(CANONICAL_STATE_NAMES)
CANONICAL_STATE_INDEX = {
    name: index for index, name in enumerate(CANONICAL_STATE_NAMES)
}
ACTION_DIM = 1


@dataclass(frozen=True)
class TransitMDPConfig:
    """Physical scales and objective weights for the canonical SMDP."""

    max_hold_seconds: float = 60.0
    episode_seconds: float = 24_000.0
    max_waiting_passengers: float = 50.0
    max_scheduled_headway_seconds: float = 720.0
    max_headway_ratio: float = 5.0
    reward_hold_weight: float = 0.05
    reward_balance_weight: float = 0.10
    reward_huber_delta: float = 1.0
    discount_gamma_per_minute: float = 0.99

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_CONFIG = TransitMDPConfig()


def _as_batch(values: np.ndarray, width: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float32)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (N, {width}); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array, was_vector


def infer_station_max(raw_observations: np.ndarray, n_lines: int = 12) -> np.ndarray:
    """Infer each directional line's largest zero-based station index."""

    observations, _ = _as_batch(raw_observations, RAW_V2_OBS_DIM, "raw_observations")
    maxima = np.ones(n_lines, dtype=np.float32)
    line_ids = np.rint(observations[:, 0]).astype(np.int64)
    for line_id in range(n_lines):
        selected = observations[line_ids == line_id, 2]
        if selected.size:
            maxima[line_id] = max(float(np.max(selected)), 1.0)
    return maxima


def raw_v2_to_canonical(
    raw_observations: np.ndarray,
    station_max_by_line: Iterable[float],
    config: TransitMDPConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Convert legacy 17-D event observations to the documented v3 state."""

    observations, was_vector = _as_batch(
        raw_observations, RAW_V2_OBS_DIM, "raw_observations"
    )
    station_max = np.asarray(tuple(station_max_by_line), dtype=np.float32)
    if station_max.ndim != 1 or station_max.size == 0 or np.any(station_max <= 0):
        raise ValueError("station_max_by_line must be a non-empty positive vector")

    line_ids = np.rint(observations[:, 0]).astype(np.int64)
    if np.any(line_ids < 0) or np.any(line_ids >= station_max.size):
        raise ValueError("raw observation contains an out-of-range line_id")

    target = np.maximum(observations[:, 8], 1.0)
    forward = observations[:, 5]
    backward = observations[:, 6]
    target_forward = forward + observations[:, 11]
    max_ratio = float(config.max_headway_ratio)

    state = np.column_stack(
        (
            line_ids.astype(np.float32),
            np.clip(observations[:, 2] / station_max[line_ids], 0.0, 1.0),
            np.clip(observations[:, 10] / config.episode_seconds, 0.0, 1.0),
            np.clip(forward / target, 0.0, max_ratio),
            np.clip(backward / target, 0.0, max_ratio),
            np.clip(target_forward / target, 0.0, max_ratio),
            np.clip(
                np.log1p(np.maximum(observations[:, 7], 0.0))
                / np.log1p(config.max_waiting_passengers),
                0.0,
                1.5,
            ),
            np.clip(observations[:, 9] / config.max_hold_seconds, 0.0, 2.0),
            np.clip(observations[:, 12] / target, 0.0, max_ratio),
            np.clip(observations[:, 13] / target, 0.0, max_ratio),
            np.clip(
                target / config.max_scheduled_headway_seconds,
                0.0,
                2.0,
            ),
        )
    ).astype(np.float32, copy=False)
    return state[0] if was_vector else state


def policy_action_to_hold_seconds(
    policy_action: np.ndarray | float,
    config: TransitMDPConfig = DEFAULT_CONFIG,
) -> np.ndarray | float:
    """Map the single SAC action in [-1, 1] to physical holding seconds."""

    action = np.asarray(policy_action, dtype=np.float32)
    hold = 0.5 * (np.clip(action, -1.0, 1.0) + 1.0) * config.max_hold_seconds
    return float(hold) if hold.ndim == 0 else hold


def hold_seconds_to_policy_action(
    hold_seconds: np.ndarray | float,
    config: TransitMDPConfig = DEFAULT_CONFIG,
) -> np.ndarray | float:
    """Map physical holding seconds back to the SAC action domain."""

    hold = np.asarray(hold_seconds, dtype=np.float32)
    action = 2.0 * np.clip(hold, 0.0, config.max_hold_seconds) / config.max_hold_seconds - 1.0
    return float(action) if action.ndim == 0 else action


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


def transition_reward_from_v2(
    raw_next_observations: np.ndarray,
    policy_actions: np.ndarray,
    config: TransitMDPConfig = DEFAULT_CONFIG,
) -> np.ndarray | float:
    """Compute the v3 transition reward from next-event headways and hold action.

    The headway term is dimensionless and route-scale invariant.  The explicit
    holding term prevents a controller from improving regularity by imposing
    unbounded passenger delay.  No hidden reward correction is applied later.
    """

    next_observations, was_vector = _as_batch(
        raw_next_observations, RAW_V2_OBS_DIM, "raw_next_observations"
    )
    actions = np.asarray(policy_actions, dtype=np.float32)
    if actions.ndim == 0:
        actions = actions.reshape(1, 1)
    elif actions.ndim == 1:
        actions = actions[:, None] if actions.size == len(next_observations) else actions[None, :]
    if actions.ndim != 2 or actions.shape != (len(next_observations), ACTION_DIM):
        raise ValueError(
            f"policy_actions must have shape ({len(next_observations)}, {ACTION_DIM}); "
            f"got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("policy_actions contains non-finite values")

    target = np.maximum(next_observations[:, 8], 1.0)
    max_ratio = float(config.max_headway_ratio)
    forward_ratio = np.clip(next_observations[:, 5] / target, 0.0, max_ratio)
    backward_ratio = np.clip(next_observations[:, 6] / target, 0.0, max_ratio)
    forward_error = forward_ratio - 1.0
    backward_error = backward_ratio - 1.0
    headway_cost = 0.5 * (
        _huber(forward_error, config.reward_huber_delta)
        + _huber(backward_error, config.reward_huber_delta)
    )
    balance_cost = np.abs(forward_ratio - backward_ratio)
    hold_fraction = np.clip((actions[:, 0] + 1.0) * 0.5, 0.0, 1.0)
    reward = -(
        headway_cost
        + config.reward_balance_weight * balance_cost
        + config.reward_hold_weight * hold_fraction
    )
    reward = reward.astype(np.float32, copy=False)
    return float(reward[0]) if was_vector else reward


def transition_duration_seconds(
    raw_observations: np.ndarray, raw_next_observations: np.ndarray
) -> np.ndarray | float:
    """Return the same-bus inter-event duration encoded in v2 observations."""

    observations, was_vector = _as_batch(raw_observations, RAW_V2_OBS_DIM, "raw_observations")
    next_observations, next_was_vector = _as_batch(
        raw_next_observations, RAW_V2_OBS_DIM, "raw_next_observations"
    )
    if was_vector != next_was_vector or observations.shape != next_observations.shape:
        raise ValueError("raw observation batches must have matching shapes")
    duration = next_observations[:, 10] - observations[:, 10]
    if np.any(duration <= 0):
        raise ValueError("same-bus transition duration must be strictly positive")
    duration = duration.astype(np.float32, copy=False)
    return float(duration[0]) if was_vector else duration


def continuous_time_discount(
    duration_seconds: np.ndarray | float,
    config: TransitMDPConfig = DEFAULT_CONFIG,
) -> np.ndarray | float:
    """Discount a variable-duration event using a per-minute base gamma."""

    duration = np.asarray(duration_seconds, dtype=np.float32)
    if np.any(duration < 0):
        raise ValueError("duration_seconds must be non-negative")
    if not 0.0 < config.discount_gamma_per_minute <= 1.0:
        raise ValueError("discount_gamma_per_minute must lie in (0, 1]")
    discount = np.power(
        config.discount_gamma_per_minute, duration / 60.0, dtype=np.float32
    )
    return float(discount) if discount.ndim == 0 else discount


def validate_v2_transition_batch(
    raw_observations: np.ndarray,
    raw_actions: np.ndarray,
    raw_next_observations: np.ndarray,
) -> dict[str, int]:
    """Validate the salvage assumptions required to convert v2 transitions."""

    observations, _ = _as_batch(raw_observations, RAW_V2_OBS_DIM, "raw_observations")
    next_observations, _ = _as_batch(
        raw_next_observations, RAW_V2_OBS_DIM, "raw_next_observations"
    )
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if observations.shape != next_observations.shape or actions.shape[0] != observations.shape[0]:
        raise ValueError("transition arrays must contain the same number of rows")
    if actions.ndim != 2 or actions.shape[1] < 1 or not np.isfinite(actions).all():
        raise ValueError("raw_actions must be a finite (N, >=1) array")

    same_line = np.rint(observations[:, 0]) == np.rint(next_observations[:, 0])
    same_bus = np.rint(observations[:, 1]) == np.rint(next_observations[:, 1])
    positive_duration = next_observations[:, 10] > observations[:, 10]
    station_delta = np.rint(next_observations[:, 2]) - np.rint(observations[:, 2])
    changed_station = station_delta != 0
    forward_station = station_delta > 0
    action_in_range = np.abs(actions[:, 0]) <= 1.00001
    return {
        "rows": int(len(observations)),
        "different_line": int((~same_line).sum()),
        "different_bus": int((~same_bus).sum()),
        "nonpositive_duration": int((~positive_duration).sum()),
        "unchanged_station": int((~changed_station).sum()),
        "nonforward_station": int((~forward_station).sum()),
        "action_out_of_range": int((~action_in_range).sum()),
    }
