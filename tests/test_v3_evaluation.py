from types import SimpleNamespace

import numpy as np

from SimpleSAC.transit_v3.evaluation import (
    headway_balance_actions,
    summarize_operations,
    summarize_passengers,
    summarize_service,
)
from bus_h2o.mdp_contract import (
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_INDEX,
    TransitMDPConfig,
)


def test_headway_thresholds_are_normalized_per_line():
    events = [
        {"line_id": "A", "forward_ratio": 0.4, "backward_ratio": 1.0, "hold_seconds": 10},
        {"line_id": "B", "forward_ratio": 1.6, "backward_ratio": 1.0, "hold_seconds": 0},
    ]
    transitions = [
        {"reward": -0.2, "delta_t_seconds": 60, "discount": 0.99},
        {"reward": -0.4, "delta_t_seconds": 120, "discount": 0.98},
    ]
    summary = summarize_operations(events, transitions)
    assert summary["headway"]["bunching_rate"] == 0.5
    assert summary["headway"]["large_gap_rate"] == 0.5
    assert np.isclose(summary["smdp_reward"]["mean"], -0.3)
    assert summary["total_hold_seconds"] == 10.0


def test_two_sided_headway_rule_has_correct_physical_direction():
    states = np.zeros((3, CANONICAL_STATE_DIM), dtype=np.float32)
    states[:, CANONICAL_STATE_INDEX["scheduled_headway_fraction"]] = 0.5
    states[:, 3] = [0.5, 1.0, 1.5]
    states[:, 4] = [1.5, 1.0, 0.5]
    actions = headway_balance_actions(
        states, gain=1.0, config=TransitMDPConfig()
    )
    assert actions.shape == (3, 1)
    assert actions[0, 0] == 1.0
    assert actions[1, 0] == -1.0
    assert actions[2, 0] == -1.0


def test_passenger_summary_keeps_unfinished_waiting_burden():
    completed = SimpleNamespace(
        start_time_n=0,
        passenger_state_s="Bus",
        arriver_time_n=10,
        boarding_time_n=30,
        travel_data_l=[[10, 30, 80, 20, 70, "bus"]],
    )
    waiting = SimpleNamespace(
        start_time_n=20,
        passenger_state_s="Stop",
        arriver_time_n=40,
        boarding_time_n=0,
        travel_data_l=[],
    )
    bridge = SimpleNamespace(
        departed_passenger_ids={"done", "wait"},
        completed_passenger_times={"done": 80},
        passenger_obj_dic={"done": completed, "wait": waiting},
    )
    summary = summarize_passengers(bridge, horizon_seconds=100)
    assert summary["completion_rate"] == 0.5
    assert summary["state_counts_at_horizon"] == {"Stop": 1}
    assert summary["completed_leg_wait_seconds"]["mean"] == 20.0
    assert summary["accrued_wait_seconds_total"] == 80.0
    assert summary["accrued_wait_seconds_per_departed_passenger"] == 40.0
    assert summary["accrued_in_vehicle_seconds_total"] == 50.0
    assert summary["accrued_in_vehicle_seconds_per_departed_passenger"] == 25.0


def test_passenger_summary_accrues_unfinished_in_vehicle_time_to_horizon():
    onboard = SimpleNamespace(
        start_time_n=0,
        passenger_state_s="Bus",
        arriver_time_n=10,
        boarding_time_n=30,
        travel_data_l=[],
    )
    bridge = SimpleNamespace(
        departed_passenger_ids={"onboard"},
        completed_passenger_times={},
        passenger_obj_dic={"onboard": onboard},
    )
    summary = summarize_passengers(bridge, horizon_seconds=100)
    assert summary["accrued_wait_seconds_total"] == 20.0
    assert summary["accrued_in_vehicle_seconds_total"] == 70.0
    assert summary["accrued_wait_plus_in_vehicle_seconds_total"] == 90.0


def test_trip_completion_uses_arrival_events_not_nonempty_trajectory():
    bridge = SimpleNamespace(
        departed_trip_ids={"a", "b"},
        completed_trip_times={"a": 100},
        active_bus_ids={"b"},
        bus_obj_dic={"a": SimpleNamespace(start_time_n=10)},
    )
    summary = summarize_service(bridge)
    assert summary["completed_trips"] == 1
    assert summary["completion_rate"] == 0.5
    assert summary["completed_trip_duration_seconds"]["mean"] == 90.0
