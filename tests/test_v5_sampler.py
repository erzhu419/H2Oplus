from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bus_h2o.mdp_contract_v5 import (
    CANONICAL_STATE_DIM,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV5,
)
from bus_h2o.passenger_time import PassengerTimeSnapshot
from SimpleSAC.transit_v5.sampler import AllLinesTransitSamplerV5


def _raw_event(station: int, snapshot: PassengerTimeSnapshot) -> list[float]:
    values = {
        "line_id": 11,
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
        self.snapshots = (
            PassengerTimeSnapshot(10.0, 10, 20, 100.0, 200.0),
            PassengerTimeSnapshot(70.0, 12, 25, 400.0, 500.0),
        )
        self.terminal_snapshot = PassengerTimeSnapshot(
            130.0, 14, 30, 700.0, 900.0
        )
        self.events = [
            {("7X", 1): [_raw_event(0, self.snapshots[0])]},
            {("7X", 1): [_raw_event(1, self.snapshots[1])]},
        ]
        self.line_map = {
            "7X": SimpleNamespace(
                passenger_time_ledger=_SnapshotSource(self.terminal_snapshot)
            )
        }
        self.state = {}
        self.done = False
        self.index = 0

    def reset(self):
        self.index = 0
        self.done = False
        self.state = self.events[0]
        return self.state

    def passenger_time_at_event(self, agent):
        assert agent == ("7X", 1)
        return self.snapshots[self.index]

    def consume_completed_agent_passenger_times(self):
        return {}

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


def test_v5_sampler_uses_true_welfare_and_closes_horizon_transition() -> None:
    replay = _Buffer()
    sampler = AllLinesTransitSamplerV5(
        _Environment(),
        replay,
        station_max_by_line=np.full(12, 10.0),
        contract_config=TransitMDPConfigV5(),
    )
    metrics = sampler.collect(_zero_policy, 2, deterministic=True)

    assert metrics["transitions"] == 2
    assert len(replay.rows) == 2
    first_values, first_metadata = replay.rows[0]
    assert first_values[2] == pytest.approx(-0.009)
    assert 0.0 < first_values[4] < 1.0
    assert first_metadata["terminal"] == 0.0
    terminal_values, terminal_metadata = replay.rows[1]
    assert terminal_values[2] == pytest.approx(-0.01)
    assert terminal_values[4] == 0.0
    assert terminal_metadata["terminal"] == 1.0
