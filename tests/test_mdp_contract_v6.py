from __future__ import annotations

import numpy as np
import pytest

from bus_h2o.mdp_contract_v6 import (
    CANONICAL_STATE_DIM,
    DEFAULT_CONFIG_V6,
    RAW_EVENT_STATE_NAMES,
    RAW_STATE_DIM,
    continuous_time_discount_v6,
    current_event_v6,
    event_batch_v6,
    pack_event_batch_v6,
    raw_state_to_canonical_v6,
    terminal_raw_state_v6,
    transition_duration_seconds_v6,
    transition_reward_v6,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot


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


def test_v6_serializes_a_simultaneous_joint_action_without_losing_history():
    events = np.stack((_event(0, 2, 10), _event(3, 5, 10)))
    first = pack_event_batch_v6(events, 0)
    second = pack_event_batch_v6(
        events, 1, np.asarray([0.25, -1.0], np.float32)
    )

    assert first.shape == (RAW_STATE_DIM,)
    np.testing.assert_array_equal(event_batch_v6(first), events)
    np.testing.assert_array_equal(current_event_v6(first), events[0])
    np.testing.assert_array_equal(current_event_v6(second), events[1])
    assert transition_duration_seconds_v6(first, second) == 0.0
    assert continuous_time_discount_v6(0.0) == 1.0

    canonical = raw_state_to_canonical_v6(
        np.stack((first, second)), np.full(12, 20.0)
    )
    assert canonical.shape == (2, CANONICAL_STATE_DIM)
    assert np.isfinite(canonical).all()
    assert not np.array_equal(canonical[0], canonical[1])


def test_v6_terminal_clock_and_nonoverlapping_welfare_identity():
    events = np.stack((_event(0, 2, 10), _event(3, 5, 10)))
    first = pack_event_batch_v6(events, 0)
    second = pack_event_batch_v6(
        events, 1, np.asarray([-1.0, -1.0], np.float32)
    )
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    same_time = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(70.0, 90, 180, 7_001.0, 14_003.0)
    terminal = terminal_raw_state_v6(second, end)

    first_reward = transition_reward_v6(start, same_time)
    final_reward = transition_reward_v6(same_time, end)
    direct_reward = transition_reward_v6(start, end)
    assert first_reward == 0.0
    assert first_reward + final_reward == direct_reward
    assert transition_duration_seconds_v6(second, terminal) == 60.0
    assert current_event_v6(terminal)[2] == 70.0
    assert current_event_v6(terminal)[-2] == 90.0
    assert current_event_v6(terminal)[-1] == 180.0


def test_v6_rejects_future_actions_and_mixed_event_times():
    events = np.stack((_event(0, 2, 10), _event(3, 5, 10)))
    with pytest.raises(ValueError, match="future batch actions"):
        pack_event_batch_v6(events, 0, np.asarray([-1.0, 0.0]))
    events[1, 2] = 11.0
    with pytest.raises(ValueError, match="one SUMO time"):
        pack_event_batch_v6(events, 0)


def test_v6_configuration_keeps_the_batch_width_schema_fixed():
    assert DEFAULT_CONFIG_V6.reward_unit_passenger_seconds == 1_000_000.0
    assert DEFAULT_CONFIG_V6.discount_gamma_per_minute == 1.0
    assert continuous_time_discount_v6(3_600.0) == 1.0
    with pytest.raises(ValueError, match="schema-fixed"):
        type(DEFAULT_CONFIG_V6)(max_simultaneous_decisions=8)
    with pytest.raises(ValueError, match="objective-fixed"):
        type(DEFAULT_CONFIG_V6)(discount_gamma_per_minute=0.99)
    with pytest.raises(ValueError, match="objective-fixed"):
        type(DEFAULT_CONFIG_V6)(reward_unit_passenger_seconds=100_000.0)
