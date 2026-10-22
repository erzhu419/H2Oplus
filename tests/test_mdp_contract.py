import numpy as np
import pytest

from bus_h2o.mdp_contract import (
    CANONICAL_STATE_DIM,
    TransitMDPConfig,
    continuous_time_discount,
    hold_seconds_to_policy_action,
    policy_action_to_hold_seconds,
    raw_v2_to_canonical,
    transition_duration_seconds,
    transition_reward_from_v2,
    validate_v2_transition_batch,
)


def raw_obs(*, line=0, bus=1, station=2, time=100.0, fwd=360.0, bwd=360.0):
    return np.array(
        [
            line,
            bus,
            station,
            0,
            0,
            fwd,
            bwd,
            5,
            360,
            10,
            time,
            360 - fwd,
            360,
            360,
            13.89,
            -1,
            0,
        ],
        dtype=np.float32,
    )


def test_action_mapping_is_one_dimensional_and_invertible():
    actions = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    holds = policy_action_to_hold_seconds(actions)
    np.testing.assert_allclose(holds, [0.0, 30.0, 60.0])
    np.testing.assert_allclose(hold_seconds_to_policy_action(holds), actions)


def test_canonical_state_has_documented_width_and_physical_scaling():
    state = raw_v2_to_canonical(raw_obs(), np.ones(12) * 10)
    assert state.shape == (CANONICAL_STATE_DIM,)
    assert state[0] == 0
    assert state[1] == pytest.approx(0.2)
    assert state[3] == pytest.approx(1.0)
    assert state[4] == pytest.approx(1.0)
    assert np.isfinite(state).all()


def test_reward_is_zero_only_for_target_headways_and_zero_hold():
    target_event = raw_obs()
    assert transition_reward_from_v2(target_event, np.array([-1.0])) == pytest.approx(0.0)
    held = transition_reward_from_v2(target_event, np.array([1.0]))
    assert held == pytest.approx(-0.05)
    irregular = transition_reward_from_v2(
        raw_obs(fwd=180.0, bwd=540.0), np.array([-1.0])
    )
    assert irregular < held


def test_discount_respects_variable_event_duration():
    short = continuous_time_discount(30.0)
    long = continuous_time_discount(180.0)
    assert 0.0 < long < short < 1.0
    assert continuous_time_discount(60.0) == pytest.approx(0.99)


def test_transition_validation_enforces_same_bus_and_positive_time():
    obs = raw_obs(station=2, time=100)
    next_obs = raw_obs(station=3, time=180)
    report = validate_v2_transition_batch(obs, np.array([[-1.0, 0.5]]), next_obs)
    assert report == {
        "rows": 1,
        "different_line": 0,
        "different_bus": 0,
        "nonpositive_duration": 0,
        "unchanged_station": 0,
        "nonforward_station": 0,
        "action_out_of_range": 0,
    }
    backward = raw_obs(station=1, time=180)
    report = validate_v2_transition_batch(obs, np.array([[-1.0, 0.5]]), backward)
    assert report["nonforward_station"] == 1
    assert transition_duration_seconds(obs, next_obs) == pytest.approx(80.0)

    wrong_bus = raw_obs(bus=2, station=3, time=180)
    assert validate_v2_transition_batch(obs, np.array([[-1.0, 0.5]]), wrong_bus)[
        "different_bus"
    ] == 1


def test_nonpositive_transition_duration_is_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        transition_duration_seconds(raw_obs(time=100), raw_obs(station=3, time=100))
