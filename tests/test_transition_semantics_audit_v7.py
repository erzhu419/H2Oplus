from copy import deepcopy

from reproducibility.audit_transition_semantics_v7 import (
    evaluate_boarding_probe,
    evaluate_pair,
)


def _trajectory(*, held: bool):
    event_time = 100.0
    departure = 161.0 if held else 102.0
    next_time = 262.0 if held else 200.0
    return {
        "intervention": {
            "line_id": "line",
            "bus_id": "bus",
            "stop_id": "stop",
            "stop_index": 3,
            "event_time_seconds": event_time,
            "arrival_time_seconds": 96.0,
            "service_completion_time_seconds": 100.0,
            "base_dwell_seconds": 4.0,
            "predicted_departure_time_seconds": 160.0 if held else 100.0,
        },
        "actual_departure_time_seconds": departure,
        "actual_postdecision_stop_seconds": departure - 100.0,
        "actual_total_stop_seconds": departure - 96.0,
        "next_event": {
            "event_time_seconds": next_time,
            "forward_headway_seconds": 362.0 if held else 300.0,
        },
        "semantic_event_audit": {"violations": []},
    }


def test_paired_transition_semantics_accepts_step_discretization():
    evaluation = evaluate_pair(
        _trajectory(held=False),
        _trajectory(held=True),
        requested_hold_seconds=60.0,
        step_tolerance_seconds=2.0,
    )

    assert evaluation["status"] == "passed"
    assert evaluation["derived_effects"][
        "actual_postdecision_stop_delta_seconds"
    ] == 59.0


def test_paired_transition_semantics_rejects_repeated_base_dwell():
    zero = _trajectory(held=False)
    held = _trajectory(held=True)
    zero["actual_departure_time_seconds"] = 107.0
    zero["actual_postdecision_stop_seconds"] = 7.0
    zero["actual_total_stop_seconds"] = 11.0

    evaluation = evaluate_pair(
        zero,
        held,
        requested_hold_seconds=60.0,
        step_tolerance_seconds=2.0,
    )

    assert evaluation["status"] == "failed"
    assert not evaluation["checks"][
        "zero_action_does_not_repeat_base_dwell"
    ]["passed"]


def _boarding_probe():
    return {
        "actual_postdecision_stop_seconds": 65.0,
        "late_candidate_passenger_ids": ["late-1", "late-2"],
        "late_candidates_boarded": ["late-1", "late-2"],
        "boarding_service_seconds_per_passenger": 2.5,
        "onboard_before": ["existing"],
        "onboard_after": ["existing", "late-1", "late-2"],
        "semantic_event_audit": {
            "checked_events": 20,
            "violations": [],
        },
    }


def test_boarding_probe_requires_at_least_one_selected_late_passenger():
    passed = evaluate_boarding_probe(
        _boarding_probe(),
        requested_hold_seconds=60.0,
        step_tolerance_seconds=2.0,
    )
    assert passed["status"] == "passed"

    incomplete = deepcopy(_boarding_probe())
    incomplete["late_candidates_boarded"] = ["late-1"]
    incomplete["actual_postdecision_stop_seconds"] = 61.0
    partial = evaluate_boarding_probe(
        incomplete,
        requested_hold_seconds=60.0,
        step_tolerance_seconds=2.0,
    )
    assert partial["status"] == "passed"

    missing = deepcopy(_boarding_probe())
    missing["late_candidates_boarded"] = []
    failed = evaluate_boarding_probe(
        missing,
        requested_hold_seconds=60.0,
        step_tolerance_seconds=2.0,
    )
    assert failed["status"] == "failed"
    assert not failed["checks"]["open_door_boarding_during_hold"]["passed"]
