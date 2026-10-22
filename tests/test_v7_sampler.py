from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bus_h2o.mdp_contract_v7 import (
    CANONICAL_STATE_DIM,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV7,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot
from SimpleSAC.transit_v7.sampler import AllLinesTransitSamplerV7


def _raw_event(
    line: int, station: int, snapshot: PassengerTimeSnapshot
) -> list[float]:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": snapshot.time_seconds,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 300.0,
        "target_forward_headway_seconds": 300.0,
        "target_backward_headway_seconds": 300.0,
        "eligible_stranded_passengers": 4,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 8,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
        "system_waiting_passengers": snapshot.waiting_passengers,
        "system_in_vehicle_passengers": snapshot.in_vehicle_passengers,
    }
    return [values[name] for name in RAW_EVENT_STATE_NAMES]


class _Buffer:
    def __init__(self) -> None:
        self.rows = []

    @property
    def online_size(self) -> int:
        return len(self.rows)

    def append(self, *args, **kwargs) -> None:
        self.rows.append((args, kwargs))


class _SnapshotSource:
    def __init__(self, snapshot: PassengerTimeSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> PassengerTimeSnapshot:
        return self._snapshot


class _Environment:
    def __init__(self) -> None:
        self.demand_scale = 1.0
        self.reset_scales = []
        self.snapshots = (
            PassengerTimeSnapshot(10.0, 10, 20, 100.0, 200.0),
            PassengerTimeSnapshot(70.0, 12, 25, 400.0, 500.0),
        )
        self.terminal_snapshot = PassengerTimeSnapshot(
            130.0, 14, 30, 700.0, 900.0
        )
        self.events = [
            {
                ("7X", 2): [_raw_event(11, 0, self.snapshots[0])],
                ("102S", 1): [_raw_event(1, 0, self.snapshots[0])],
            },
            {("7X", 2): [_raw_event(11, 1, self.snapshots[1])]},
        ]
        self.line_map = {
            "7X": SimpleNamespace(
                passenger_time_ledger=_SnapshotSource(self.terminal_snapshot)
            )
        }
        self.state = {}
        self.done = False
        self.index = 0
        self.received_actions = []

    def reset(self):
        self.reset_scales.append(self.demand_scale)
        self.index = 0
        self.done = False
        self.state = self.events[0]
        return self.state

    def passenger_time_at_event(self, agent):
        assert agent in self.events[self.index]
        return self.snapshots[self.index]

    def step_to_event(self, actions):
        self.received_actions.append(dict(actions))
        self.index += 1
        if self.index >= len(self.events):
            self.state = {}
            self.done = True
        else:
            self.state = self.events[self.index]
        return self.state, {}, self.done


class _SequentialPolicy:
    def __init__(self) -> None:
        self.actions = iter((-1.0, 0.0, -1.0))
        self.states = []

    def __call__(self, observations, deterministic):
        assert deterministic
        assert observations.shape == (1, CANONICAL_STATE_DIM)
        self.states.append(observations.copy())
        return np.asarray([[next(self.actions)]], dtype=np.float32)


def _zero_policy(observations, deterministic):
    del deterministic
    return np.full((len(observations), 1), -1.0, dtype=np.float32)


def test_v7_sampler_serializes_batch_and_counts_global_welfare_once() -> None:
    replay = _Buffer()
    env = _Environment()
    policy = _SequentialPolicy()
    sampler = AllLinesTransitSamplerV7(
        env,
        replay,
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV7(),
    )
    metrics = sampler.collect(policy, 3, deterministic=True)

    assert metrics["events"] == 3
    assert metrics["transitions"] == 3
    assert metrics["maximum_simultaneous_decisions"] == 2
    assert len(replay.rows) == 3
    first, first_metadata = replay.rows[0]
    assert first[2] == 0.0
    assert first[4] == 1.0
    assert first_metadata["terminal"] == 0.0
    assert first_metadata["delta_t_seconds"] == 0.0
    second, second_metadata = replay.rows[1]
    assert second[2] == pytest.approx(-900.0)
    assert second[4] == 1.0
    assert second_metadata["terminal"] == 0.0
    assert second_metadata["delta_t_seconds"] == 60.0
    terminal, terminal_metadata = replay.rows[2]
    assert terminal[2] == pytest.approx(-1000.0)
    assert terminal[4] == 0.0
    assert terminal_metadata["terminal"] == 1.0
    assert terminal_metadata["delta_t_seconds"] == 60.0
    assert [row[1]["trajectory_id"] for row in replay.rows] == [0, 0, 0]

    first_batch_actions = env.received_actions[0]
    assert list(first_batch_actions) == [("102S", 1), ("7X", 2)]
    assert first_batch_actions[("102S", 1)][0] == 0.0
    assert first_batch_actions[("7X", 2)][0] == 30.0
    assert policy.states[0][0, 2] == 0.0
    assert policy.states[1][0, 2] == pytest.approx(1.0 / 15.0)


def test_v7_sampler_never_splits_a_simultaneous_batch() -> None:
    replay = _Buffer()
    policy = _SequentialPolicy()
    sampler = AllLinesTransitSamplerV7(
        _Environment(),
        replay,
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV7(),
    )
    metrics = sampler.collect(policy, 1, deterministic=True)
    assert metrics["events"] == 2
    assert metrics["transitions"] == 1
    assert replay.rows[0][0][2] == 0.0


def test_v7_sampler_hard_cap_defers_an_oversized_simultaneous_batch() -> None:
    replay = _Buffer()
    env = _Environment()
    policy = _SequentialPolicy()
    sampler = AllLinesTransitSamplerV7(
        env,
        replay,
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV7(),
    )

    blocked = sampler.collect(
        policy, 1, deterministic=True, hard_event_cap=True
    )
    assert blocked["events"] == 0
    assert blocked["transitions"] == 0
    assert blocked["blocked_simultaneous_batch_size"] == 2
    assert blocked["hard_event_cap_reached"] is True
    assert env.received_actions == []
    assert replay.online_size == 0

    completed = sampler.collect(
        policy, 2, deterministic=True, hard_event_cap=True
    )
    assert completed["events"] == 2
    assert completed["hard_event_cap_reached"] is False
    assert len(env.received_actions) == 1


def test_v7_sampler_cycles_registered_episode_demand_scales() -> None:
    replay = _Buffer()
    env = _Environment()
    sampler = AllLinesTransitSamplerV7(
        env,
        replay,
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV7(),
        demand_scale_schedule=(0.75, 1.25),
    )

    metrics = sampler.collect(_zero_policy, 6, deterministic=True)

    assert metrics["episodes_started"] == 2
    assert metrics["demand_scale_episode_starts"] == {"0.75": 1, "1.25": 1}
    assert metrics["current_demand_scale"] == 1.25
    assert env.reset_scales == [0.75, 1.25]
    state = sampler.runtime_state_dict()
    assert state["demand_scale_episode_counts"] == {0.75: 1, 1.25: 1}

    incompatible = AllLinesTransitSamplerV7(
        _Environment(),
        _Buffer(),
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV7(),
        demand_scale_schedule=(1.0,),
    )
    with pytest.raises(ValueError, match="demand-scale schedule changed"):
        incompatible.load_runtime_state_dict(state)
