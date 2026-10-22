from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from bus_h2o.mdp_contract_v7 import (
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_INDEX,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V7,
    RAW_EVENT_STATE_NAMES,
    RAW_STATE_DIM,
    RAW_STATE_NAMES,
    TransitMDPConfigV7,
    continuous_time_discount_v7,
    current_event_v7,
    event_batch_v7,
    hold_seconds_to_policy_action_v7,
    pack_event_batch_v7,
    policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7,
    terminal_raw_state_v7,
    transition_duration_seconds_v7,
    transition_reward_v7,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot


def _event(line: int, station: int, time_seconds: float) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time_seconds,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 360.0,
        "target_forward_headway_seconds": 1_800.0,
        "target_backward_headway_seconds": 360.0,
        "eligible_stranded_passengers": 4.0,
        "onboard_passengers": 20.0,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 260.0,
        "forward_bus_present": 1.0,
        "backward_bus_present": 1.0,
        "bus_capacity": 50.0,
        "system_waiting_passengers": 100.0,
        "system_in_vehicle_passengers": 200.0,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def test_v7_deduplicates_global_event_quantities_and_round_trips_events():
    events = np.stack((_event(0, 2, 10.0), _event(3, 5, 10.0)))
    first = pack_event_batch_v7(events, 0)
    second = pack_event_batch_v7(
        events, 1, np.asarray([0.25, -1.0], np.float32)
    )

    assert first.shape == (RAW_STATE_DIM,)
    assert RAW_STATE_DIM == 245
    assert CANONICAL_STATE_DIM == 229
    np.testing.assert_array_equal(event_batch_v7(first), events)
    np.testing.assert_array_equal(current_event_v7(first), events[0])
    np.testing.assert_array_equal(current_event_v7(second), events[1])
    assert sum(name == "decision_time_seconds" for name in RAW_STATE_NAMES) == 1
    assert sum("system_waiting_passengers" in name for name in RAW_STATE_NAMES) == 1
    assert sum("system_in_vehicle_passengers" in name for name in RAW_STATE_NAMES) == 1
    assert sum(name == "horizon_progress" for name in CANONICAL_STATE_NAMES) == 1
    assert sum(
        "system_waiting_passengers_scaled" in name
        for name in CANONICAL_STATE_NAMES
    ) == 1


def test_v7_names_scales_as_scales_instead_of_false_maxima_or_fractions():
    config = asdict(DEFAULT_CONFIG_V7)
    assert "scheduled_headway_scale_seconds" in config
    assert "base_dwell_scale_seconds" in config
    assert "max_scheduled_headway_seconds" not in config
    assert "max_base_dwell_seconds" not in config
    assert not any("target_forward_headway_fraction" in name for name in CANONICAL_STATE_NAMES)

    state = pack_event_batch_v7(np.stack((_event(0, 2, 10.0),)), 0)
    canonical = raw_state_to_canonical_v7(state, np.full(12, 20.0))
    assert canonical[
        CANONICAL_STATE_INDEX["slot_00_target_forward_headway_scaled"]
    ] == pytest.approx(2.5)
    assert canonical[
        CANONICAL_STATE_INDEX["slot_00_base_dwell_scaled"]
    ] == pytest.approx(2.0)
    assert canonical[
        CANONICAL_STATE_INDEX["system_waiting_passengers_scaled"]
    ] == pytest.approx(100.0 / 20_000.0)

    batch = np.stack((state, state.copy()))
    batch[1, 0] = 20.0
    vectorized = raw_state_to_canonical_v7(batch, np.full(12, 20.0))
    np.testing.assert_array_equal(vectorized[0], canonical)
    assert vectorized[1, CANONICAL_STATE_INDEX["horizon_progress"]] == pytest.approx(
        20.0 / 24_000.0
    )


def test_v7_stores_physical_welfare_and_keeps_training_scaling_external():
    start = PassengerTimeSnapshot(10.0, 100, 200, 1_000.0, 2_000.0)
    end = PassengerTimeSnapshot(70.0, 90, 180, 7_001.0, 14_003.0)
    expected_cost = 2.0 * 6_001.0 + 12_003.0

    assert transition_reward_v7(start, end) == -expected_cost
    assert DEFAULT_CONFIG_V7.reward_unit_passenger_seconds == 1.0
    with pytest.raises(ValueError, match="physical generalized"):
        TransitMDPConfigV7(reward_unit_passenger_seconds=100_000.0)


def test_v7_terminal_clock_duration_and_undiscounted_continuation():
    events = np.stack((_event(0, 2, 10.0), _event(3, 5, 10.0)))
    state = pack_event_batch_v7(events, 1, np.asarray([-1.0, -1.0]))
    end = PassengerTimeSnapshot(70.0, 90, 180, 7_001.0, 14_003.0)
    terminal = terminal_raw_state_v7(state, end)

    assert transition_duration_seconds_v7(state, terminal) == 60.0
    assert current_event_v7(terminal)[2] == 70.0
    assert current_event_v7(terminal)[-2] == 90.0
    assert current_event_v7(terminal)[-1] == 180.0
    assert continuous_time_discount_v7(3_600.0) == 1.0


def test_v7_action_mapping_is_one_dimensional_and_strict():
    assert policy_action_to_hold_seconds_v7(-1.0) == 0.0
    assert policy_action_to_hold_seconds_v7(1.0) == 60.0
    assert hold_seconds_to_policy_action_v7(0.0) == -1.0
    assert hold_seconds_to_policy_action_v7(60.0) == 1.0
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        policy_action_to_hold_seconds_v7(1.1)
    with pytest.raises(ValueError, match=r"\[0, 60.0\]"):
        hold_seconds_to_policy_action_v7(61.0)
    with pytest.raises(ValueError, match="schema-fixed"):
        TransitMDPConfigV7(max_simultaneous_decisions=8)


def test_v7_rejects_inconsistent_batch_globals_and_future_actions():
    events = np.stack((_event(0, 2, 10.0), _event(3, 5, 10.0)))
    with pytest.raises(ValueError, match="future batch actions"):
        pack_event_batch_v7(events, 0, np.asarray([-1.0, 0.0]))
    events[1, 2] = 11.0
    with pytest.raises(ValueError, match="sim_time_seconds"):
        pack_event_batch_v7(events, 0)
