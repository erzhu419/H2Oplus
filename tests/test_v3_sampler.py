import copy

import numpy as np

from SimpleSAC.transit_v3.sampler import AllLinesTransitSampler
from bus_h2o.mdp_contract import CANONICAL_STATE_DIM, TransitMDPConfig


def raw_event(line, bus, station, time):
    return [line, bus, station, 0, 0, 360, 360, 0, 360, 5, time, 0, 360, 360, 13.89]


class FakeBuffer:
    def __init__(self):
        self.rows = []

    @property
    def online_size(self):
        return len(self.rows)

    def append(self, *values, **kwargs):
        self.rows.append((values, kwargs))


class FakeAllLinesEnv:
    def __init__(self):
        self.done = False
        self.index = 0
        self.events = [
            {("7X", 1): [raw_event(11, 1, 0, 10)]},
            {("7X", 1): [raw_event(11, 1, 1, 70)], ("7S", 2): [raw_event(10, 2, 0, 70)]},
            {("7X", 1): [raw_event(11, 1, 2, 130)], ("7S", 2): [raw_event(10, 2, 1, 130)]},
        ]
        self.state = {}
        self.reward = {}

    def reset(self):
        self.done = False
        self.index = 0
        self.state = self.events[0]
        return self.state

    def step_to_event(self, actions):
        for action in actions.values():
            assert action[1] == 1.0
        self.index += 1
        if self.index >= len(self.events):
            self.done = True
            self.state = {}
        else:
            self.state = self.events[self.index]
        return self.state, {}, self.done


class ReusedBusEnv(FakeAllLinesEnv):
    def __init__(self):
        super().__init__()
        self.events = [
            {("7X", 1): [raw_event(11, 1, 9, 10)]},
            {("7X", 1): [raw_event(11, 1, 0, 70)]},
            {("7X", 1): [raw_event(11, 1, 1, 130)]},
        ]


def zero_hold_policy(observations, deterministic):
    assert observations.shape[1] == CANONICAL_STATE_DIM
    return np.full((len(observations), 1), -1.0, dtype=np.float32)


def test_sampler_persists_same_bus_transitions_and_fixes_speed():
    buffer = FakeBuffer()
    sampler = AllLinesTransitSampler(
        FakeAllLinesEnv(),
        buffer,
        station_max_by_line=np.ones(12) * 10,
        contract_config=TransitMDPConfig(),
    )
    stats = sampler.collect(zero_hold_policy, 4, deterministic=True)
    assert stats["events"] >= 4
    assert stats["transitions"] == 3
    assert stats["episodes_started"] == 1
    assert buffer.online_size == 3
    values, kwargs = buffer.rows[0]
    observation, action, reward, next_observation, discount = values
    assert observation.shape == (CANONICAL_STATE_DIM,)
    assert next_observation.shape == (CANONICAL_STATE_DIM,)
    np.testing.assert_array_equal(action, [-1.0])
    assert reward == 0.0
    assert 0.0 < discount < 1.0
    assert kwargs["terminal"] == 0.0
    assert kwargs["episode_id"] == 0
    assert isinstance(kwargs["trajectory_id"], int)


def test_sampler_drops_cross_trip_station_wraparound():
    buffer = FakeBuffer()
    sampler = AllLinesTransitSampler(
        ReusedBusEnv(),
        buffer,
        station_max_by_line=np.ones(12) * 10,
        contract_config=TransitMDPConfig(),
    )
    stats = sampler.collect(zero_hold_policy, 3, deterministic=True)
    assert stats["transitions"] == 1
    assert stats["trip_boundaries_total"] == 1
    assert buffer.rows[0][1]["trajectory_id"] == 1


def test_sampler_runtime_state_resumes_pending_transition_exactly():
    station_max = np.ones(12) * 10
    contract = TransitMDPConfig()
    continuous_buffer = FakeBuffer()
    continuous = AllLinesTransitSampler(
        FakeAllLinesEnv(),
        continuous_buffer,
        station_max_by_line=station_max,
        contract_config=contract,
    )
    continuous.collect(zero_hold_policy, 1, deterministic=True)
    runtime_state = copy.deepcopy(continuous.runtime_state_dict())
    expected_stats = continuous.collect(zero_hold_policy, 2, deterministic=True)

    restored_buffer = FakeBuffer()
    restored = AllLinesTransitSampler(
        FakeAllLinesEnv(),
        restored_buffer,
        station_max_by_line=station_max,
        contract_config=contract,
    )
    restored.load_runtime_state_dict(runtime_state)
    actual_stats = restored.collect(zero_hold_policy, 2, deterministic=True)

    assert actual_stats == expected_stats
    assert len(restored_buffer.rows) == len(continuous_buffer.rows)
    for restored_row, continuous_row in zip(
        restored_buffer.rows, continuous_buffer.rows
    ):
        restored_values, restored_kwargs = restored_row
        continuous_values, continuous_kwargs = continuous_row
        assert restored_kwargs == continuous_kwargs
        for actual, expected in zip(restored_values, continuous_values):
            np.testing.assert_array_equal(actual, expected)
