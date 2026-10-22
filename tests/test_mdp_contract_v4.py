import numpy as np
import pytest

from bus_h2o.mdp_contract_v4 import (
    CANONICAL_STATE_INDEX,
    CANONICAL_STATE_NAMES,
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV4,
    correct_headway_measurement_jitter_v4,
    hold_seconds_to_policy_action_v4,
    policy_action_to_hold_seconds_v4,
    raw_event_to_canonical_v4,
    transition_cost_components_v4,
    transition_duration_seconds_v4,
    transition_reward_v4,
)


def _event(**overrides):
    values = {
        "line_id": 2.0,
        "station_index": 5.0,
        "sim_time_seconds": 3600.0,
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
    }
    values.update(overrides)
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], dtype=np.float32)


def test_v4_symbol_names_are_unique_and_dimensions_are_explicit():
    assert len(RAW_EVENT_STATE_NAMES) == len(set(RAW_EVENT_STATE_NAMES))
    assert len(CANONICAL_STATE_NAMES) == len(set(CANONICAL_STATE_NAMES))
    assert RAW_EVENT_STATE_DIM == len(RAW_EVENT_STATE_INDEX)
    assert len(CANONICAL_STATE_NAMES) == len(CANONICAL_STATE_INDEX)


def test_v4_canonical_state_has_one_physical_meaning_per_feature():
    state = raw_event_to_canonical_v4(_event(), np.full(12, 10.0))
    assert state[CANONICAL_STATE_INDEX["line_id"]] == 2.0
    assert state[CANONICAL_STATE_INDEX["station_progress"]] == 0.5
    assert state[CANONICAL_STATE_INDEX["episode_progress"]] == pytest.approx(0.15)
    assert state[CANONICAL_STATE_INDEX["forward_headway_ratio"]] == 1.0
    assert state[CANONICAL_STATE_INDEX["backward_headway_ratio"]] == 1.0
    assert state[
        CANONICAL_STATE_INDEX["eligible_stranded_fraction"]
    ] == pytest.approx(0.2)
    assert state[CANONICAL_STATE_INDEX["onboard_fraction"]] == 0.5
    assert state[CANONICAL_STATE_INDEX["eligible_demand_per_headway_fraction"]] == pytest.approx(0.12)


def test_absent_neighbor_is_masked_and_explicitly_flagged():
    raw = _event(
        forward_headway_seconds=0.0,
        forward_bus_present=0.0,
    )
    state = raw_event_to_canonical_v4(raw, np.full(12, 10.0))
    assert state[CANONICAL_STATE_INDEX["forward_headway_ratio"]] == 1.0
    assert state[CANONICAL_STATE_INDEX["forward_bus_present"]] == 0.0


def test_target_headway_reference_fraction_is_not_silently_truncated():
    state = raw_event_to_canonical_v4(
        _event(
            target_forward_headway_seconds=1800.0,
            target_backward_headway_seconds=1800.0,
        ),
        np.full(12, 10.0),
    )
    assert state[CANONICAL_STATE_INDEX["target_forward_headway_fraction"]] == 2.5
    assert state[CANONICAL_STATE_INDEX["target_backward_headway_fraction"]] == 2.5


def test_substep_negative_headway_is_corrected_but_larger_error_is_rejected():
    corrected = correct_headway_measurement_jitter_v4(
        _event(forward_headway_seconds=-0.8)
    )
    assert corrected[RAW_EVENT_STATE_INDEX["forward_headway_seconds"]] == 0.0
    with pytest.raises(ValueError, match="measurement-jitter tolerance"):
        correct_headway_measurement_jitter_v4(
            _event(forward_headway_seconds=-1.01)
        )


def test_v4_reward_is_zero_for_equal_headways_and_no_hold():
    current = _event()
    following = _event(sim_time_seconds=3660.0)
    components = transition_cost_components_v4(current, [-1.0], following)
    assert components == {
        "waiting_cost": pytest.approx(0.0),
        "holding_cost": pytest.approx(0.0),
        "total_cost": pytest.approx(0.0),
    }
    assert transition_reward_v4(current, [-1.0], following) == pytest.approx(0.0)


def test_v4_waiting_cost_matches_expected_wait_derivation():
    current = _event()
    following = _event(
        sim_time_seconds=3660.0,
        forward_headway_seconds=450.0,
        backward_headway_seconds=150.0,
    )
    components = transition_cost_components_v4(current, [-1.0], following)
    # 2 * 0.02 * (150^2 + (-150)^2) / (4 * 60 passenger-seconds)
    assert components["waiting_cost"] == pytest.approx(7.5)
    assert components["holding_cost"] == pytest.approx(0.0)


def test_v4_holding_cost_uses_current_onboard_load_only():
    current = _event(onboard_passengers=25.0)
    following = _event(sim_time_seconds=3660.0)
    action_for_30_seconds = hold_seconds_to_policy_action_v4(30.0)
    components = transition_cost_components_v4(
        current, [action_for_30_seconds], following
    )
    # 1 * 25 onboard passengers * 30 seconds / 60 passenger-seconds
    assert components["holding_cost"] == pytest.approx(12.5)
    assert transition_reward_v4(
        current, [action_for_30_seconds], following
    ) == pytest.approx(-12.5)


def test_reward_uses_one_global_unit_without_line_specific_reweighting():
    current = _event(onboard_passengers=25.0, bus_capacity=50.0)
    following = _event(
        sim_time_seconds=3660.0,
        forward_headway_seconds=450.0,
        backward_headway_seconds=150.0,
        target_forward_headway_seconds=300.0,
        target_backward_headway_seconds=300.0,
        bus_capacity=50.0,
    )
    reference = transition_cost_components_v4(current, [0.0], following)

    changed_scale_current = _event(
        onboard_passengers=25.0,
        bus_capacity=100.0,
        target_forward_headway_seconds=600.0,
        target_backward_headway_seconds=600.0,
    )
    changed_scale_following = _event(
        sim_time_seconds=3660.0,
        forward_headway_seconds=750.0,
        backward_headway_seconds=450.0,
        target_forward_headway_seconds=600.0,
        target_backward_headway_seconds=600.0,
        bus_capacity=100.0,
    )
    changed = transition_cost_components_v4(
        changed_scale_current, [0.0], changed_scale_following
    )
    assert changed == pytest.approx(reference)


def test_v4_action_mapping_preserves_exact_no_hold_atom():
    assert policy_action_to_hold_seconds_v4(-1.0) == 0.0
    assert hold_seconds_to_policy_action_v4(0.0) == -1.0
    np.testing.assert_allclose(
        policy_action_to_hold_seconds_v4(
            hold_seconds_to_policy_action_v4(np.array([0.0, 7.0, 60.0]))
        ),
        [0.0, 7.0, 60.0],
        atol=1e-5,
    )


def test_v4_transition_duration_uses_physical_event_times():
    current = _event(sim_time_seconds=100.0)
    following = _event(sim_time_seconds=172.5)
    assert transition_duration_seconds_v4(current, following) == pytest.approx(72.5)


def test_v4_config_rejects_ambiguous_or_invalid_scales():
    with pytest.raises(ValueError, match="waiting_time_value"):
        TransitMDPConfigV4(waiting_time_value=0.0)
    with pytest.raises(ValueError, match="reward_unit_passenger_seconds"):
        TransitMDPConfigV4(reward_unit_passenger_seconds=0.0)
    with pytest.raises(ValueError, match="discount_gamma"):
        TransitMDPConfigV4(discount_gamma_per_minute=1.1)
