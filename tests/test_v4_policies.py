import numpy as np
import pytest

from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_NAMES
from SimpleSAC.transit_v4.policies import (
    headway_balance_hold_seconds_v4,
    hurdle_random_actions_v4,
    myopic_welfare_hold_seconds_v4,
)


def _event(**overrides):
    values = {
        "line_id": 0,
        "station_index": 2,
        "sim_time_seconds": 1000,
        "forward_headway_seconds": 240,
        "backward_headway_seconds": 360,
        "target_forward_headway_seconds": 300,
        "target_backward_headway_seconds": 300,
        "eligible_stranded_passengers": 0,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.04,
        "base_dwell_seconds": 10,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
    }
    values.update(overrides)
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], np.float32)


def test_headway_policy_corrects_schedule_adjusted_errors():
    assert headway_balance_hold_seconds_v4(_event(), 0.5) == pytest.approx(30.0)
    assert headway_balance_hold_seconds_v4(
        _event(forward_headway_seconds=360, backward_headway_seconds=240), 0.5
    ) == 0.0


def test_myopic_welfare_policy_holds_less_as_onboard_load_increases():
    empty = myopic_welfare_hold_seconds_v4(_event(onboard_passengers=0))
    loaded = myopic_welfare_hold_seconds_v4(_event(onboard_passengers=40))
    assert 0.0 <= loaded < empty <= 60.0
    assert myopic_welfare_hold_seconds_v4(
        _event(eligible_arrival_rate_per_second=0.0)
    ) == 0.0


def test_hurdle_random_policy_has_reproducible_exact_zero_mass():
    first = hurdle_random_actions_v4(10_000, np.random.default_rng(8))
    second = hurdle_random_actions_v4(10_000, np.random.default_rng(8))
    np.testing.assert_array_equal(first, second)
    zero_rate = np.mean(first[:, 0] == -1.0)
    assert zero_rate == pytest.approx(0.5, abs=0.02)
    assert np.all(first[first[:, 0] > -1.0] < 1.0)
