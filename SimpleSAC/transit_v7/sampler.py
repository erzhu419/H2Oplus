"""Persistent chronological online sampler for transit-smdp-v7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from bus_h2o.mdp_contract_v7 import (
    ACTION_DIM,
    RAW_EVENT_STATE_DIM,
    SCHEMA_VERSION,
    TransitMDPConfigV7,
    continuous_time_discount_v7,
    pack_event_batch_v7,
    policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7,
    terminal_raw_state_v7,
    transition_duration_seconds_v7,
    transition_reward_v7,
)
from bus_h2o.passenger_time import (
    PassengerTimeSnapshot,
    aggregate_passenger_time_snapshots,
)


def _event_observation(value: Any) -> np.ndarray:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"v7 simulator event has shape {array.shape}; "
            f"expected ({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(array).all():
        raise ValueError("v7 simulator event contains non-finite values")
    return array.copy()


def _active_event_batch(state: dict) -> list[tuple[object, np.ndarray]]:
    active = []
    for agent, observations in state.items():
        if not observations:
            continue
        active.append((agent, _event_observation(observations[-1])))
    return sorted(active, key=lambda row: (str(row[0][0]), int(row[0][1])))


def _terminal_passenger_time(env) -> PassengerTimeSnapshot:
    return aggregate_passenger_time_snapshots(
        line_env.passenger_time_ledger.snapshot()
        for line_env in env.line_map.values()
    )


@dataclass
class PendingChronologicalTransition:
    raw_observation: np.ndarray
    canonical_observation: np.ndarray
    action: np.ndarray
    passenger_time: PassengerTimeSnapshot
    episode_id: int
    trajectory_id: int


class AllLinesTransitSamplerV7:
    """Serialize each simultaneous event batch before advancing the simulator."""

    RUNTIME_SCHEMA = "transit-sampler-runtime-v3"

    def __init__(
        self,
        env,
        replay_buffer,
        *,
        station_max_by_line: np.ndarray,
        contract_config: TransitMDPConfigV7,
        demand_scale_schedule: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.station_max_by_line = np.asarray(
            station_max_by_line, dtype=np.float32
        )
        self.config = contract_config
        schedule = (
            (float(getattr(env, "demand_scale", 1.0)),)
            if demand_scale_schedule is None
            else tuple(float(value) for value in demand_scale_schedule)
        )
        if (
            not schedule
            or not np.isfinite(schedule).all()
            or any(value <= 0.0 for value in schedule)
            or len(set(schedule)) != len(schedule)
        ):
            raise ValueError(
                "v7 online demand-scale schedule must contain unique positive values"
            )
        self.demand_scale_schedule = schedule
        self.demand_scale_episode_counts = {value: 0 for value in schedule}
        self.current_demand_scale: float | None = None
        self.pending: PendingChronologicalTransition | None = None
        self.episode_active = False
        self.episodes = 0
        self.current_episode_id = -1
        self.current_trajectory_id = -1
        self.next_trajectory_id = 0

    def runtime_state_dict(self) -> dict:
        return {
            "schema_version": self.RUNTIME_SCHEMA,
            "sampler_contract": SCHEMA_VERSION,
            "contract_config": asdict(self.config),
            "station_max_by_line": self.station_max_by_line.copy(),
            "demand_scale_schedule": self.demand_scale_schedule,
            "demand_scale_episode_counts": dict(
                self.demand_scale_episode_counts
            ),
            "current_demand_scale": self.current_demand_scale,
            "env": self.env,
            "pending": self.pending,
            "episode_active": self.episode_active,
            "episodes": self.episodes,
            "current_episode_id": self.current_episode_id,
            "current_trajectory_id": self.current_trajectory_id,
            "next_trajectory_id": self.next_trajectory_id,
        }

    def load_runtime_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != self.RUNTIME_SCHEMA:
            raise ValueError("v7 sampler runtime state has an unsupported schema")
        if state.get("sampler_contract") != SCHEMA_VERSION:
            raise ValueError("v7 sampler runtime MDP schema changed")
        if state.get("contract_config") != asdict(self.config):
            raise ValueError("v7 sampler runtime MDP contract changed")
        station_max = np.asarray(state.get("station_max_by_line"), dtype=np.float32)
        if not np.array_equal(station_max, self.station_max_by_line):
            raise ValueError("v7 sampler station normalization changed")
        if tuple(state.get("demand_scale_schedule", ())) != (
            self.demand_scale_schedule
        ):
            raise ValueError("v7 sampler demand-scale schedule changed")
        counts = {
            float(key): int(value)
            for key, value in state.get(
                "demand_scale_episode_counts", {}
            ).items()
        }
        if set(counts) != set(self.demand_scale_schedule) or any(
            value < 0 for value in counts.values()
        ):
            raise ValueError("v7 sampler demand-scale counts are invalid")
        self.env = state["env"]
        self.pending = state["pending"]
        self.episode_active = bool(state["episode_active"])
        self.episodes = int(state["episodes"])
        self.current_episode_id = int(state["current_episode_id"])
        self.current_trajectory_id = int(state["current_trajectory_id"])
        self.next_trajectory_id = int(state["next_trajectory_id"])
        self.demand_scale_episode_counts = counts
        current_scale = state.get("current_demand_scale")
        self.current_demand_scale = (
            None if current_scale is None else float(current_scale)
        )
        if self.current_demand_scale not in (
            None,
            *self.demand_scale_schedule,
        ):
            raise ValueError("v7 sampler current demand scale is invalid")

    def _start_episode(self) -> None:
        if self.pending is not None:
            raise RuntimeError("cannot reset v7 with an unclosed chronological row")
        demand_scale = self.demand_scale_schedule[
            self.episodes % len(self.demand_scale_schedule)
        ]
        self.env.demand_scale = demand_scale
        for line_env in self.env.line_map.values():
            line_env.demand_scale = demand_scale
        self.env.reset()
        self.current_demand_scale = demand_scale
        self.demand_scale_episode_counts[demand_scale] += 1
        self.episode_active = True
        self.episodes += 1
        self.current_episode_id = self.episodes - 1
        self.current_trajectory_id = self.next_trajectory_id
        self.next_trajectory_id += 1
        if not _active_event_batch(self.env.state):
            self.env.step_to_event({})
        if self.env.done and not _active_event_batch(self.env.state):
            raise RuntimeError("v7 simulator episode ended before a control event")

    def _canonical(self, raw_state: np.ndarray) -> np.ndarray:
        return raw_state_to_canonical_v7(
            raw_state, self.station_max_by_line, self.config
        )

    def _append_pending(
        self,
        raw_next_state: np.ndarray,
        passenger_time_end: PassengerTimeSnapshot,
        *,
        terminal: bool,
    ) -> tuple[float, float]:
        previous = self.pending
        if previous is None:
            raise RuntimeError("v7 sampler has no chronological row to close")
        duration = float(
            passenger_time_end.time_seconds - previous.passenger_time.time_seconds
        )
        state_duration = float(
            transition_duration_seconds_v7(
                previous.raw_observation, raw_next_state
            )
        )
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("v7 online transition has an invalid duration")
        if abs(duration - state_duration) > 1e-9:
            raise RuntimeError(
                "v7 online passenger-time and state clocks disagree: "
                f"{duration} versus {state_duration}"
            )
        reward = float(
            transition_reward_v7(
                previous.passenger_time, passenger_time_end, self.config
            )
        )
        discount = (
            0.0
            if terminal
            else float(continuous_time_discount_v7(duration, self.config))
        )
        self.replay_buffer.append(
            previous.canonical_observation,
            previous.action,
            reward,
            self._canonical(raw_next_state),
            discount,
            terminal=float(terminal),
            episode_id=previous.episode_id,
            trajectory_id=previous.trajectory_id,
            delta_t_seconds=duration,
        )
        if terminal:
            self.pending = None
        return reward, duration

    @staticmethod
    def _policy_action(
        policy: Callable[[np.ndarray, bool], np.ndarray],
        observation: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        actions = np.asarray(
            policy(observation[None, :], deterministic), dtype=np.float32
        )
        if actions.ndim == 1:
            actions = actions[:, None]
        if actions.shape != (1, ACTION_DIM):
            raise ValueError(
                f"v7 policy returned {actions.shape}; expected {(1, ACTION_DIM)}"
            )
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.00001):
            raise ValueError("v7 policy emitted an invalid action")
        return np.clip(actions[0], -1.0, 1.0)

    def collect(
        self,
        policy: Callable[[np.ndarray, bool], np.ndarray],
        n_events: int,
        *,
        deterministic: bool = False,
        hard_event_cap: bool = False,
    ) -> dict[str, Any]:
        if n_events <= 0:
            raise ValueError("n_events must be positive")
        start_episodes = self.episodes
        start_scale_counts = dict(self.demand_scale_episode_counts)
        if not self.episode_active or self.env.done:
            self._start_episode()

        events = 0
        transitions = 0
        rewards: list[float] = []
        durations: list[float] = []
        holds: list[float] = []
        maximum_batch = 0
        blocked_batch_size = 0
        while events < n_events:
            active = _active_event_batch(self.env.state)
            if not active:
                _, _, done = self.env.step_to_event({})
                if done:
                    if self.pending is not None:
                        end = _terminal_passenger_time(self.env)
                        terminal_state = terminal_raw_state_v7(
                            self.pending.raw_observation, end
                        )
                        reward, duration = self._append_pending(
                            terminal_state, end, terminal=True
                        )
                        transitions += 1
                        rewards.append(reward)
                        durations.append(duration)
                    self.episode_active = False
                    self._start_episode()
                continue

            maximum_batch = max(maximum_batch, len(active))
            if hard_event_cap and events + len(active) > n_events:
                blocked_batch_size = len(active)
                break
            local_batch = np.stack([observation for _, observation in active])
            contexts = [
                self.env.passenger_time_at_event(agent) for agent, _ in active
            ]
            passenger_time = contexts[0]
            if any(context != passenger_time for context in contexts[1:]):
                raise RuntimeError(
                    "simultaneous simulator events disagree on passenger time"
                )

            prior_actions = np.full(len(active), -1.0, dtype=np.float32)
            env_actions = {}
            for index, (agent, _) in enumerate(active):
                raw_state = pack_event_batch_v7(
                    local_batch, index, prior_actions, self.config
                )
                canonical_state = self._canonical(raw_state)
                if self.pending is not None:
                    reward, duration = self._append_pending(
                        raw_state, passenger_time, terminal=False
                    )
                    transitions += 1
                    rewards.append(reward)
                    durations.append(duration)

                action = self._policy_action(
                    policy, canonical_state, deterministic
                )
                hold_seconds = float(
                    policy_action_to_hold_seconds_v7(action[0], self.config)
                )
                holds.append(hold_seconds)
                env_actions[agent] = [hold_seconds, 1.0]
                self.pending = PendingChronologicalTransition(
                    raw_observation=raw_state.copy(),
                    canonical_observation=canonical_state.copy(),
                    action=action.copy(),
                    passenger_time=passenger_time,
                    episode_id=self.current_episode_id,
                    trajectory_id=self.current_trajectory_id,
                )
                prior_actions[index] = action[0]
                events += 1

            _, _, done = self.env.step_to_event(env_actions)
            if done:
                end = _terminal_passenger_time(self.env)
                terminal_state = terminal_raw_state_v7(
                    self.pending.raw_observation, end
                )
                reward, duration = self._append_pending(
                    terminal_state, end, terminal=True
                )
                transitions += 1
                rewards.append(reward)
                durations.append(duration)
                self.episode_active = False
                if events < n_events:
                    self._start_episode()

        return {
            "events": events,
            "transitions": transitions,
            "episodes_started": self.episodes - start_episodes,
            "demand_scale_episode_starts": {
                f"{scale:g}": self.demand_scale_episode_counts[scale]
                - start_scale_counts[scale]
                for scale in self.demand_scale_schedule
            },
            "current_demand_scale": self.current_demand_scale,
            "discarded_pending_total": 0,
            "trip_boundaries_total": 0,
            "maximum_simultaneous_decisions": maximum_batch,
            "blocked_simultaneous_batch_size": blocked_batch_size,
            "hard_event_cap_reached": bool(blocked_batch_size),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_delta_t": float(np.mean(durations)) if durations else 0.0,
            "mean_hold_seconds": float(np.mean(holds)) if holds else 0.0,
            "online_buffer_size": int(self.replay_buffer.online_size),
        }


__all__ = ["AllLinesTransitSamplerV7", "PendingChronologicalTransition"]
