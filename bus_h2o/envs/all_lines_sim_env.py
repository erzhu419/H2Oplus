"""All-lines adapter for the low-fidelity transit simulator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

_BUS_H2O_ROOT = Path(__file__).resolve().parents[1]
if str(_BUS_H2O_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUS_H2O_ROOT))

from sim_core.sim import MultiLineEnv
from mdp_contract_v4 import (
    RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4,
    RAW_EVENT_STATE_NAMES as RAW_EVENT_STATE_NAMES_V4,
)
from mdp_contract_v5 import RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V5
from passenger_time import PassengerTimeSnapshot


AgentKey = Tuple[str, int]


class AllLinesSimEnv(MultiLineEnv):
    """Expose every line to one shared policy through flat `(line, bus)` keys."""

    def __init__(
        self,
        path: str,
        debug: bool = False,
        render: bool = False,
        episode_seconds: float = 24_000.0,
        passenger_arrival_end_seconds: float = 18_000.0,
        demand_scale: float = 1.0,
        observation_schema: str = "v3",
        observation_speed_seed: int | None = None,
    ):
        if observation_schema not in {"v3", "v4", "v5", "v6", "v7"}:
            raise ValueError(
                "observation_schema must be one of v3, v4, v5, v6, or v7"
            )
        runtime_calibration_filename = (
            "segment_runtime_calibration_v7_v4.json"
            if observation_schema == "v7"
            else "segment_runtime_calibration_v4.json"
        )
        if observation_schema == "v7" and not (
            Path(path) / runtime_calibration_filename
        ).is_file():
            raise FileNotFoundError(
                "v7 requires segment_runtime_calibration_v7_v4.json"
            )
        super().__init__(
            path,
            debug=debug,
            render=render,
            runtime_calibration_filename=runtime_calibration_filename,
            observation_speed_seed=observation_speed_seed,
        )
        if episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive")
        if passenger_arrival_end_seconds <= 0:
            raise ValueError("passenger_arrival_end_seconds must be positive")
        if demand_scale <= 0:
            raise ValueError("demand_scale must be positive")
        self.episode_seconds = float(episode_seconds)
        self.passenger_arrival_end_seconds = float(passenger_arrival_end_seconds)
        self.demand_scale = float(demand_scale)
        self.observation_schema = observation_schema
        for line_env in self.line_map.values():
            line_env.max_time = self.episode_seconds
            line_env.passenger_arrival_end_time = self.passenger_arrival_end_seconds
            line_env.demand_scale = self.demand_scale
        if observation_schema == "v4":
            self.state_dim = RAW_EVENT_STATE_DIM_V4
        elif observation_schema in {"v5", "v6", "v7"}:
            self.state_dim = RAW_EVENT_STATE_DIM_V5
        else:
            self.state_dim = 15
        self.state: dict[AgentKey, Any] = {}
        self.reward: dict[AgentKey, float] = {}
        self.done = False

    @staticmethod
    def _flatten(nested: dict[str, dict[int, Any]]) -> dict[AgentKey, Any]:
        return {
            (line_id, int(bus_id)): value
            for line_id, buses in nested.items()
            for bus_id, value in buses.items()
        }

    @staticmethod
    def _eligible_arrival_rate(line_env, station, sim_time: float) -> float:
        if sim_time >= line_env.passenger_arrival_end_time or station.od is None:
            return 0.0
        effective_hour = 6 + int(max(sim_time, 0.0) // 3600)
        period = station.od.get(f"{effective_hour:02d}:00:00", {})
        return float(
            line_env.demand_scale
            * sum(max(float(value), 0.0) for value in period.values())
            / 3600.0
        )

    def _v4_observation(self, line_id: str, bus_id: int, legacy_observation) -> list[float]:
        line_env = self.line_map[line_id]
        bus = next(
            (candidate for candidate in line_env.bus_all if candidate.bus_id == bus_id),
            None,
        )
        if bus is None:
            raise KeyError(f"simulator event has no bus object for {(line_id, bus_id)!r}")
        legacy = list(legacy_observation)
        if len(legacy) != 15:
            raise ValueError(f"legacy simulator observation has width {len(legacy)}, expected 15")
        station = bus.last_station
        values = {
            "line_id": float(line_env.line_idx),
            "station_index": float(legacy[2]),
            "sim_time_seconds": float(legacy[10]),
            "forward_headway_seconds": float(legacy[5]),
            "backward_headway_seconds": float(legacy[6]),
            "target_forward_headway_seconds": float(bus.target_forward_headway),
            "target_backward_headway_seconds": float(bus.target_backward_headway),
            "eligible_stranded_passengers": float(len(station.waiting_passengers)),
            "onboard_passengers": float(len(bus.passengers)),
            "eligible_arrival_rate_per_second": self._eligible_arrival_rate(
                line_env, station, float(legacy[10])
            ),
            "base_dwell_seconds": float(legacy[9]),
            "forward_bus_present": float(bool(bus.forward_bus_present)),
            "backward_bus_present": float(bool(bus.backward_bus_present)),
            "bus_capacity": float(bus.capacity),
        }
        return [values[name] for name in RAW_EVENT_STATE_NAMES_V4]

    def _v5_observation(
        self, line_id: str, bus_id: int, legacy_observation
    ) -> list[float]:
        local = self._v4_observation(line_id, bus_id, legacy_observation)
        passenger_time = self.passenger_time_at_event((line_id, bus_id))
        legacy_time = float(list(legacy_observation)[10])
        if abs(passenger_time.time_seconds - legacy_time) > 1e-9:
            raise RuntimeError("simulator v5 event and passenger-time clocks disagree")
        return local + [
            float(passenger_time.waiting_passengers),
            float(passenger_time.in_vehicle_passengers),
        ]

    def _selected_schema(self, nested: dict[str, dict[int, Any]]):
        if self.observation_schema == "v3":
            return nested
        converted = {}
        for line_id, buses in nested.items():
            converted[line_id] = {}
            for bus_id, observations in buses.items():
                converted[line_id][bus_id] = [
                    (
                        self._v5_observation(line_id, int(bus_id), observation)
                        if self.observation_schema in {"v5", "v6", "v7"}
                        else self._v4_observation(
                            line_id, int(bus_id), observation
                        )
                    )
                    for observation in observations
                    if observation
                ]
        return converted

    def _sync(self, state: dict[str, dict[int, Any]], reward: dict[str, dict[int, float]], done: bool):
        self.state = self._flatten(self._selected_schema(state))
        self.reward = self._flatten(reward)
        self.done = bool(done)
        return self.state, self.reward, self.done

    def reset(self, snapshot=None):
        if snapshot is not None:
            raise NotImplementedError(
                "snapshot reset is intentionally excluded from the canonical v3 protocol"
            )
        state, reward, done = super().reset()
        return self._sync(state, reward, done)[0]

    def _nested_actions(self, flat_actions: dict[AgentKey, Any]) -> dict[str, dict[int, Any]]:
        nested = {
            line_id: {bus_id: None for bus_id in range(line_env.max_agent_num)}
            for line_id, line_env in self.line_map.items()
        }
        for key, action in flat_actions.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("all-lines actions must use (line_id, bus_id) keys")
            line_id, bus_id = key
            if line_id not in nested or int(bus_id) not in nested[line_id]:
                raise KeyError(f"unknown simulator agent {key!r}")
            if action is None:
                nested[line_id][int(bus_id)] = None
                continue
            command = list(action) if isinstance(action, (list, tuple)) else action
            if hasattr(command, "reshape"):
                command = command.reshape(-1).tolist()
            if isinstance(command, list):
                if len(command) not in (1, 2):
                    raise ValueError(
                        "holding command must be a scalar or [hold_seconds, 1.0]"
                    )
                hold_seconds = float(command[0])
                if len(command) == 2 and abs(float(command[1]) - 1.0) > 1e-9:
                    raise ValueError(
                        "speed ratio is fixed at 1.0 in the holding-only contract"
                    )
            else:
                hold_seconds = float(command)
            if not 0.0 <= hold_seconds <= 60.0:
                raise ValueError("additional hold must lie in [0, 60] seconds")
            nested[line_id][int(bus_id)] = [hold_seconds, 1.0]
        return nested

    def step_to_event(self, action_dict: dict[AgentKey, Any], **kwargs):
        state, reward, done = super().step_to_event(
            self._nested_actions(action_dict), **kwargs
        )
        return self._sync(state, reward, done)

    def passenger_time_at_event(self, agent: AgentKey) -> PassengerTimeSnapshot:
        """Return the system-wide cumulative passenger time at an event."""

        key = (str(agent[0]), int(agent[1]))
        try:
            return self._event_passenger_time[key]
        except KeyError as error:
            raise KeyError(f"no passenger-time snapshot for event {key!r}") from error

    def capture_full_system_snapshot(self) -> dict[str, Any]:
        all_buses = []
        all_stations = []
        for line_id, line_env in self.line_map.items():
            route_length = sum(route.distance for route in line_env.routes)
            for bus in line_env.bus_all:
                if not bus.on_route:
                    continue
                all_buses.append(
                    {
                        "bus_id": int(bus.bus_id),
                        "line_id": line_id,
                        "pos": float(bus.absolute_distance),
                        "route_length": float(route_length),
                        "speed": float(bus.current_speed),
                        "load": int(len(bus.passengers)),
                        "direction": int(0 if bus.direction else 1),
                    }
                )
            cumulative_position = 0.0
            for station_index, station in enumerate(line_env.stations):
                all_stations.append(
                    {
                        "station_id": int(station.station_id),
                        "station_name": station.station_name,
                        "line_id": line_id,
                        "pos": float(cumulative_position),
                        "route_length": float(route_length),
                        "waiting_count": int(len(station.waiting_passengers)),
                    }
                )
                if station_index < len(line_env.routes):
                    cumulative_position += line_env.routes[station_index].distance
        return {
            "sim_time": float(self.current_time),
            "current_time": float(self.current_time),
            "all_buses": all_buses,
            "all_stations": all_stations,
        }
