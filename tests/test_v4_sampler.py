import numpy as np

from bus_h2o.mdp_contract_v4 import (
    CANONICAL_STATE_DIM,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV4,
)
from SimpleSAC.transit_v4.sampler import AllLinesTransitSamplerV4


def _event(station, sim_time, forward=300.0, backward=300.0):
    values = {
        "line_id": 11,
        "station_index": station,
        "sim_time_seconds": sim_time,
        "forward_headway_seconds": forward,
        "backward_headway_seconds": backward,
        "target_forward_headway_seconds": 300.0,
        "target_backward_headway_seconds": 300.0,
        "eligible_stranded_passengers": 0,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 8,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
    }
    return [values[name] for name in RAW_EVENT_STATE_NAMES]


class _Buffer:
    def __init__(self):
        self.rows = []

    @property
    def online_size(self):
        return len(self.rows)

    def append(self, *args, **kwargs):
        self.rows.append((args, kwargs))


class _Environment:
    def __init__(self):
        self.events = [
            {("7X", 1): [_event(0, 10)]},
            {("7X", 1): [_event(1, 70, forward=450, backward=150)]},
            {("7X", 1): [_event(2, 130)]},
        ]
        self.done = False
        self.state = {}
        self.index = 0

    def reset(self):
        self.index = 0
        self.done = False
        self.state = self.events[0]
        return self.state

    def step_to_event(self, actions):
        assert all(action[1] == 1.0 for action in actions.values())
        self.index += 1
        if self.index >= len(self.events):
            self.state = {}
            self.done = True
        else:
            self.state = self.events[self.index]
        return self.state, {}, self.done


def _zero_policy(observations, deterministic):
    assert deterministic
    assert observations.shape[1] == CANONICAL_STATE_DIM
    return np.full((len(observations), 1), -1.0, dtype=np.float32)


def test_v4_sampler_builds_welfare_reward_from_both_transition_endpoints():
    replay = _Buffer()
    sampler = AllLinesTransitSamplerV4(
        _Environment(),
        replay,
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV4(),
    )
    metrics = sampler.collect(_zero_policy, 3, deterministic=True)
    assert metrics["transitions"] == 2
    first_values, first_metadata = replay.rows[0]
    observation, action, reward, next_observation, discount = first_values
    assert observation.shape == (CANONICAL_STATE_DIM,)
    assert next_observation.shape == (CANONICAL_STATE_DIM,)
    np.testing.assert_array_equal(action, [-1.0])
    assert reward == np.float32(-7.5)
    assert 0.0 < discount < 1.0
    assert first_metadata["trajectory_id"] == 0
