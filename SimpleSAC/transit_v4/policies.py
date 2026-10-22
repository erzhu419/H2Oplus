"""Auditable non-learning policies for v4 data collection and baselines."""

from __future__ import annotations

import numpy as np

from bus_h2o.mdp_contract_v4 import (
    DEFAULT_CONFIG_V4,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    TransitMDPConfigV4,
    hold_seconds_to_policy_action_v4,
)


def _batch(raw_observations: np.ndarray) -> tuple[np.ndarray, bool]:
    observations = np.asarray(raw_observations, dtype=np.float32)
    was_vector = observations.ndim == 1
    if was_vector:
        observations = observations[None, :]
    if observations.ndim != 2 or observations.shape[1] != RAW_EVENT_STATE_DIM:
        raise ValueError("raw v4 observations have an invalid shape")
    if not np.isfinite(observations).all():
        raise ValueError("raw v4 observations contain non-finite values")
    return observations, was_vector


def _column(observations: np.ndarray, name: str) -> np.ndarray:
    return observations[:, RAW_EVENT_STATE_INDEX[name]]


def headway_balance_hold_seconds_v4(
    raw_observations: np.ndarray,
    gain: float,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray | float:
    """Correct half the scheduled-headway-adjusted gap imbalance."""

    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be finite and non-negative")
    observations, was_vector = _batch(raw_observations)
    forward_error = (
        _column(observations, "forward_headway_seconds")
        - _column(observations, "target_forward_headway_seconds")
    ) * np.rint(_column(observations, "forward_bus_present"))
    backward_error = (
        _column(observations, "backward_headway_seconds")
        - _column(observations, "target_backward_headway_seconds")
    ) * np.rint(_column(observations, "backward_bus_present"))
    hold = np.clip(
        0.5 * gain * (backward_error - forward_error),
        0.0,
        config.max_hold_seconds,
    ).astype(np.float32)
    return float(hold[0]) if was_vector else hold


def myopic_welfare_hold_seconds_v4(
    raw_observations: np.ndarray,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray | float:
    """Minimize the local quadratic waiting cost plus onboard holding cost.

    This closed-form policy is a transparent behavior-policy baseline, not the
    proposed learning method.  It uses the same generalized passenger-time
    values as the v4 MDP contract and returns zero whenever holding has no
    positive local welfare derivative.
    """

    observations, was_vector = _batch(raw_observations)
    capacity = np.maximum(_column(observations, "bus_capacity"), 1.0)
    arrival_rate = np.maximum(
        _column(observations, "eligible_arrival_rate_per_second"), 0.0
    )
    onboard = np.clip(_column(observations, "onboard_passengers"), 0.0, capacity)
    forward_present = np.clip(
        np.rint(_column(observations, "forward_bus_present")), 0.0, 1.0
    )
    backward_present = np.clip(
        np.rint(_column(observations, "backward_bus_present")), 0.0, 1.0
    )
    forward_error = (
        _column(observations, "forward_headway_seconds")
        - _column(observations, "target_forward_headway_seconds")
    )
    backward_error = (
        _column(observations, "backward_headway_seconds")
        - _column(observations, "target_backward_headway_seconds")
    )

    waiting_coefficient = (
        config.waiting_time_value
        * arrival_rate
        / (4.0 * config.reward_unit_passenger_seconds)
    )
    holding_derivative = (
        config.in_vehicle_time_value
        * onboard
        / config.reward_unit_passenger_seconds
    )
    curvature = waiting_coefficient * (
        forward_present + backward_present
    )
    improvement_derivative = waiting_coefficient * (
        backward_present * backward_error
        - forward_present * forward_error
    )
    numerator = improvement_derivative - 0.5 * holding_derivative
    hold = np.divide(
        numerator,
        curvature,
        out=np.zeros_like(numerator),
        where=curvature > 0.0,
    )
    hold = np.clip(hold, 0.0, config.max_hold_seconds).astype(np.float32)
    return float(hold[0]) if was_vector else hold


def hurdle_random_actions_v4(
    n: int,
    rng: np.random.Generator,
    *,
    zero_probability: float = 0.5,
    beta_alpha: float = 1.0,
    beta_beta: float = 3.0,
    config: TransitMDPConfigV4 = DEFAULT_CONFIG_V4,
) -> np.ndarray:
    """Sample exact zeros plus a right-skewed positive holding distribution."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= zero_probability <= 1.0:
        raise ValueError("zero_probability must lie in [0, 1]")
    if beta_alpha <= 0.0 or beta_beta <= 0.0:
        raise ValueError("beta parameters must be positive")
    zero = rng.random(n) < zero_probability
    positive_fraction = rng.beta(beta_alpha, beta_beta, size=n)
    positive_fraction = np.maximum(positive_fraction, np.finfo(np.float32).eps)
    holds = positive_fraction * config.max_hold_seconds
    holds[zero] = 0.0
    return np.asarray(hold_seconds_to_policy_action_v4(holds, config), dtype=np.float32)[
        :, None
    ]


__all__ = [
    "headway_balance_hold_seconds_v4",
    "hurdle_random_actions_v4",
    "myopic_welfare_hold_seconds_v4",
]
