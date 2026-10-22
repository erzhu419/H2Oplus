from reproducibility.audit_simulator_fidelity_v6 import compare_fidelity


def _sim_episode(seed, generated=100):
    return {
        "seed": seed,
        "episode": {
            "observed_end_time_seconds": 100.0,
            "ledger_end_time_seconds": 100.0,
            "control_decisions": 200,
        },
        "passengers": {
            "generated_journeys": generated,
            "generated_ride_legs": generated,
            "completion_rate": 0.95,
            "journey_state_balance_error": 0,
            "ledger_count_errors": {"waiting": 0, "in_vehicle": 0},
            "accrued_generalized_passenger_seconds": 1_000 * generated,
            "accrued_wait_seconds_total": 300 * generated,
            "accrued_in_vehicle_seconds_total": 400 * generated,
        },
        "service": {
            "departed_trips": 389,
            "completed_trips": 389,
            "completed_trip_duration_seconds": {"mean": 500.0},
        },
    }


def _sumo_summary():
    return {
        "episode": {"control_decisions": 200},
        "passenger_welfare": {
            "accrued_generalized_seconds_per_departed_passenger": 1_000.0
        },
        "passengers": {
            "departed_passengers": 100,
            "accrued_wait_seconds_per_departed_passenger": 300.0,
            "accrued_in_vehicle_seconds_per_departed_passenger": 400.0,
            "completion_rate": 0.95,
        },
        "service": {"completed_trip_duration_seconds": {"mean": 500.0}},
    }


def test_fidelity_comparison_normalizes_stochastic_journey_count():
    thresholds = {
        "demand_relative_error": 0.05,
        "decision_relative_gap": 0.10,
        "generalized_welfare_relative_gap": 0.20,
        "wait_time_relative_gap": 0.25,
        "in_vehicle_time_relative_gap": 0.25,
        "completion_rate_absolute_gap": 0.05,
        "trip_duration_relative_gap": 0.15,
    }
    comparison = compare_fidelity(
        [_sim_episode(1, 98), _sim_episode(2, 102)],
        [_sumo_summary(), _sumo_summary()],
        expected_source_persons=100,
        episode_seconds=100.0,
        thresholds=thresholds,
    )
    assert comparison["passed"]
    assert comparison["gaps"]["generalized_welfare_relative_gap"] == 0.0
    assert comparison["checks"]["all_invariants"]


def test_fidelity_comparison_rejects_ledger_mismatch():
    episode = _sim_episode(1)
    episode["passengers"]["ledger_count_errors"]["waiting"] = -3
    thresholds = {name: 1.0 for name in (
        "demand_relative_error",
        "decision_relative_gap",
        "generalized_welfare_relative_gap",
        "wait_time_relative_gap",
        "in_vehicle_time_relative_gap",
        "completion_rate_absolute_gap",
        "trip_duration_relative_gap",
    )}
    comparison = compare_fidelity(
        [episode],
        [_sumo_summary(), _sumo_summary()],
        expected_source_persons=100,
        episode_seconds=100.0,
        thresholds=thresholds,
    )
    assert not comparison["passed"]
    assert not comparison["checks"]["all_invariants"]


def test_fidelity_comparison_requires_bootstrap_interval_inside_tolerance():
    thresholds = {
        "demand_relative_error": 0.05,
        "decision_relative_gap": 1.0,
        "generalized_welfare_relative_gap": 1.0,
        "wait_time_relative_gap": 1.0,
        "in_vehicle_time_relative_gap": 1.0,
        "completion_rate_absolute_gap": 1.0,
        "trip_duration_relative_gap": 1.0,
    }
    comparison = compare_fidelity(
        [_sim_episode(1, 80), _sim_episode(2, 120)],
        [_sumo_summary(), _sumo_summary()],
        expected_source_persons=100,
        episode_seconds=100.0,
        thresholds=thresholds,
        bootstrap_samples=2_000,
    )
    assert comparison["gaps"]["demand_relative_error"] == 0.0
    assert not comparison["gap_bootstrap_interval_checks"][
        "demand_relative_error"
    ]
    assert not comparison["checks"]["all_gap_confidence_intervals"]
    assert not comparison["passed"]
