from __future__ import annotations

import numpy as np
import pytest

from bus_h2o.mdp_contract_v5 import (
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_INDEX,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V5,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_NAMES,
    raw_event_to_canonical_v5,
    transition_cost_components_v5,
    transition_duration_seconds_v5,
    transition_reward_v5,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot


def _raw_event(time_seconds: float = 3600.0) -> np.ndarray:
    values = {
        "line_id": 2.0,
        "station_index": 5.0,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 300.0,
        "target_forward_headway_seconds": 300.0,
        "target_backward_headway_seconds": 300.0,
        "eligible_stranded_passengers": 10.0,
        "onboard_passengers": 25.0,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 12.0,
        "forward_bus_present": 1.0,
        "backward_bus_present": 1.0,
        "bus_capacity": 50.0,
        "system_waiting_passengers": 500.0,
        "system_in_vehicle_passengers": 4_000.0,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], dtype=np.float32)


def test_v5_names_and_dimensions_are_unique() -> None:
    assert RAW_EVENT_STATE_DIM == 16
    assert CANONICAL_STATE_DIM == 15
    assert len(RAW_EVENT_STATE_NAMES) == len(set(RAW_EVENT_STATE_NAMES))
    assert len(CANONICAL_STATE_NAMES) == len(set(CANONICAL_STATE_NAMES))


def test_v5_adds_broadcast_system_counts_without_changing_local_features() -> None:
    canonical = raw_event_to_canonical_v5(_raw_event(), np.full(12, 10.0))
    assert canonical.shape == (CANONICAL_STATE_DIM,)
    assert canonical[CANONICAL_STATE_INDEX["system_waiting_fraction"]] == pytest.approx(
        0.025
    )
    assert canonical[
        CANONICAL_STATE_INDEX["system_in_vehicle_fraction"]
    ] == pytest.approx(0.2)


def test_v5_reward_is_exact_scaled_passenger_time_difference() -> None:
    start = PassengerTimeSnapshot(100.0, 5, 7, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(110.0, 6, 8, 1_050.0, 2_070.0)
    components = transition_cost_components_v5(start, end)
    expected = (2.0 * 50.0 + 70.0) / 100_000.0
    assert components["wait_passenger_seconds"] == 50.0
    assert components["in_vehicle_passenger_seconds"] == 70.0
    assert components["total_cost"] == pytest.approx(expected)
    assert transition_reward_v5(start, end) == pytest.approx(-expected)
    assert DEFAULT_CONFIG_V5.reward_unit_passenger_seconds == 100_000.0


def test_v5_duration_uses_physical_event_times() -> None:
    assert transition_duration_seconds_v5(
        _raw_event(100.0), _raw_event(175.0)
    ) == 75.0
