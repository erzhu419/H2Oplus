"""Persistent all-lines online sampler for the canonical transit SMDP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from bus_h2o.mdp_contract import (
    ACTION_DIM,
    RAW_V2_OBS_DIM,
    TransitMDPConfig,
    continuous_time_discount,
    policy_action_to_hold_seconds,
    raw_v2_to_canonical,
    transition_duration_seconds,
    transition_reward_from_v2,
)


def _active_observations(state: dict) -> list[tuple[object, np.ndarray]]:
    active = []
    for agent, values in state.items():
        if not values:
            continue
        observation = values[-1]
        if isinstance(observation, list) and observation and isinstance(observation[0], list):
            observation = observation[-1]
        array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if array.size:
            active.append((agent, array))
    return active


def _to_raw_v2(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.size == RAW_V2_OBS_DIM:
        return observation.copy()
    if observation.size == RAW_V2_OBS_DIM - 2:
        return np.concatenate((observation, np.zeros(2, dtype=np.float32)))
    raise ValueError(f"sim observation has unsupported width {observation.size}")


def _v3_reward(previous_raw, next_raw, action, config):
    del previous_raw
    return transition_reward_from_v2(next_raw, action, config)


@dataclass
class PendingTransition:
    raw_observation: np.ndarray
    canonical_observation: np.ndarray
    action: np.ndarray
    station_index: int
    episode_id: int
    trajectory_id: int
    reward_context: Any = None


class AllLinesTransitSampler:
    """Collect same-bus transitions without resetting at epoch boundaries."""

    def __init__(
        self,
        env,
        replay_buffer,
        *,
        station_max_by_line: np.ndarray,
        contract_config: TransitMDPConfig,
        raw_transform: Callable[[np.ndarray], np.ndarray] = _to_raw_v2,
        canonical_transform: Callable = raw_v2_to_canonical,
        reward_transform: Callable = _v3_reward,
        duration_transform: Callable = transition_duration_seconds,
        discount_transform: Callable = continuous_time_discount,
        action_to_hold_transform: Callable = policy_action_to_hold_seconds,
        station_index_position: int = 2,
        sampler_contract: str = "transit-smdp-v3",
        reward_context_transform: Callable[[object, object], Any] | None = None,
        context_reward_transform: Callable[[Any, Any, object], float] | None = None,
        context_duration_transform: Callable[[Any, Any], float] | None = None,
        completed_contexts_transform: Callable[[object], dict[object, Any]] | None = None,
        terminal_context_transform: Callable[[object], Any] | None = None,
    ) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.station_max_by_line = np.asarray(station_max_by_line, dtype=np.float32)
        self.config = contract_config
        self.raw_transform = raw_transform
        self.canonical_transform = canonical_transform
        self.reward_transform = reward_transform
        self.duration_transform = duration_transform
        self.discount_transform = discount_transform
        self.action_to_hold_transform = action_to_hold_transform
        self.station_index_position = int(station_index_position)
        self.sampler_contract = str(sampler_contract)
        self.reward_context_transform = reward_context_transform
        self.context_reward_transform = context_reward_transform
        self.context_duration_transform = context_duration_transform
        self.completed_contexts_transform = completed_contexts_transform
        self.terminal_context_transform = terminal_context_transform
        context_hooks = (
            reward_context_transform,
            context_reward_transform,
            context_duration_transform,
        )
        if any(hook is not None for hook in context_hooks) and not all(
            hook is not None for hook in context_hooks
        ):
            raise ValueError(
                "context reward, duration, and event transforms must be configured together"
            )
        if (
            completed_contexts_transform is not None
            or terminal_context_transform is not None
        ) and context_reward_transform is None:
            raise ValueError("terminal context hooks require context-based rewards")
        self.pending: dict[object, PendingTransition] = {}
        self.episode_active = False
        self.episodes = 0
        self.discarded_pending = 0
        self.trip_boundaries = 0
        self.current_episode_id = -1
        self._trajectory_by_agent: dict[object, int] = {}
        self._next_trajectory_id = 0

    def runtime_state_dict(self) -> dict:
        """Return the exact simulator/sampler state without duplicating replay."""

        return {
            "schema_version": "transit-sampler-runtime-v1",
            "sampler_contract": self.sampler_contract,
            "contract_config": asdict(self.config),
            "station_max_by_line": self.station_max_by_line.copy(),
            "env": self.env,
            "pending": self.pending,
            "episode_active": self.episode_active,
            "episodes": self.episodes,
            "discarded_pending": self.discarded_pending,
            "trip_boundaries": self.trip_boundaries,
            "current_episode_id": self.current_episode_id,
            "trajectory_by_agent": self._trajectory_by_agent,
            "next_trajectory_id": self._next_trajectory_id,
        }

    def load_runtime_state_dict(self, state: dict) -> None:
        """Restore a state emitted by :meth:`runtime_state_dict`."""

        if state.get("schema_version") != "transit-sampler-runtime-v1":
            raise ValueError("sampler runtime state has an unsupported schema")
        if state.get("sampler_contract", "transit-smdp-v3") != self.sampler_contract:
            raise ValueError("sampler runtime MDP schema changed")
        if state.get("contract_config") != asdict(self.config):
            raise ValueError("sampler runtime MDP contract changed")
        station_max = np.asarray(state.get("station_max_by_line"), dtype=np.float32)
        if not np.array_equal(station_max, self.station_max_by_line):
            raise ValueError("sampler runtime station normalization changed")
        self.env = state["env"]
        self.pending = dict(state["pending"])
        self.episode_active = bool(state["episode_active"])
        self.episodes = int(state["episodes"])
        self.discarded_pending = int(state["discarded_pending"])
        self.trip_boundaries = int(state["trip_boundaries"])
        self.current_episode_id = int(state["current_episode_id"])
        self._trajectory_by_agent = dict(state["trajectory_by_agent"])
        self._next_trajectory_id = int(state["next_trajectory_id"])

    def _trajectory_id(self, agent: object, *, force_new: bool = False) -> int:
        if force_new or agent not in self._trajectory_by_agent:
            self._trajectory_by_agent[agent] = self._next_trajectory_id
            self._next_trajectory_id += 1
        return self._trajectory_by_agent[agent]

    def _start_episode(self) -> None:
        if self.pending:
            self.discarded_pending += len(self.pending)
            self.pending.clear()
        self.env.reset()
        self.episode_active = True
        self.episodes += 1
        self.current_episode_id = self.episodes - 1
        self._trajectory_by_agent.clear()
        if not _active_observations(self.env.state):
            self.env.step_to_event({})

    def _close_terminal_transition(
        self, agent: object, end_context: Any
    ) -> tuple[float, float] | None:
        previous = self.pending.pop(agent, None)
        if previous is None:
            return None
        reward = float(
            self.context_reward_transform(
                previous.reward_context, end_context, self.config
            )
        )
        duration = float(
            self.context_duration_transform(previous.reward_context, end_context)
        )
        if duration < 0.0 or not np.isfinite(duration):
            raise ValueError("terminal transition duration must be finite and non-negative")
        self.replay_buffer.append(
            previous.canonical_observation,
            previous.action,
            reward,
            previous.canonical_observation,
            0.0,
            terminal=1.0,
            episode_id=previous.episode_id,
            trajectory_id=previous.trajectory_id,
        )
        self._trajectory_by_agent.pop(agent, None)
        return reward, duration

    def _close_completed_transitions(self) -> list[tuple[float, float]]:
        if self.completed_contexts_transform is None:
            return []
        closed = []
        for agent, end_context in self.completed_contexts_transform(self.env).items():
            result = self._close_terminal_transition(agent, end_context)
            if result is not None:
                closed.append(result)
        return closed

    def _close_episode_transitions(self) -> list[tuple[float, float]]:
        if self.terminal_context_transform is None or not self.pending:
            return []
        end_context = self.terminal_context_transform(self.env)
        closed = []
        for agent in tuple(self.pending):
            result = self._close_terminal_transition(agent, end_context)
            if result is not None:
                closed.append(result)
        return closed

    def collect(
        self,
        policy: Callable[[np.ndarray, bool], np.ndarray],
        n_events: int,
        *,
        deterministic: bool = False,
    ) -> dict[str, float | int]:
        if n_events <= 0:
            raise ValueError("n_events must be positive")
        start_episodes = self.episodes
        if not self.episode_active or self.env.done:
            self._start_episode()

        events = 0
        transitions = 0
        rewards = []
        durations = []
        holds = []
        while events < n_events:
            active = _active_observations(self.env.state)
            if not active:
                _, _, done = self.env.step_to_event({})
                if done:
                    self.episode_active = False
                    self._start_episode()
                continue

            raw_batch = np.stack(
                [self.raw_transform(observation) for _, observation in active]
            )
            canonical_batch = self.canonical_transform(
                raw_batch, self.station_max_by_line, self.config
            )
            action_batch = np.asarray(
                policy(canonical_batch, deterministic), dtype=np.float32
            )
            if action_batch.ndim == 1:
                action_batch = action_batch[:, None]
            if action_batch.shape != (len(active), ACTION_DIM):
                raise ValueError(
                    f"policy returned {action_batch.shape}; expected {(len(active), ACTION_DIM)}"
                )
            action_batch = np.clip(action_batch, -1.0, 1.0)

            env_actions = {}
            for row, ((agent, _), raw_observation, canonical_observation, action) in enumerate(
                zip(active, raw_batch, canonical_batch, action_batch)
            ):
                del row
                events += 1
                station_index = int(
                    round(float(raw_observation[self.station_index_position]))
                )
                previous = self.pending.pop(agent, None)
                reward_context = (
                    self.reward_context_transform(self.env, agent)
                    if self.reward_context_transform is not None
                    else None
                )
                if previous is not None and station_index > previous.station_index:
                    if self.context_reward_transform is None:
                        reward = self.reward_transform(
                            previous.raw_observation,
                            raw_observation,
                            previous.action,
                            self.config,
                        )
                        duration = self.duration_transform(
                            previous.raw_observation, raw_observation
                        )
                    else:
                        reward = self.context_reward_transform(
                            previous.reward_context,
                            reward_context,
                            self.config,
                        )
                        duration = self.context_duration_transform(
                            previous.reward_context, reward_context
                        )
                    discount = self.discount_transform(duration, self.config)
                    self.replay_buffer.append(
                        previous.canonical_observation,
                        previous.action,
                        reward,
                        canonical_observation,
                        discount,
                        terminal=0.0,
                        episode_id=previous.episode_id,
                        trajectory_id=previous.trajectory_id,
                    )
                    transitions += 1
                    rewards.append(float(reward))
                    durations.append(float(duration))
                    trajectory_id = previous.trajectory_id
                elif previous is not None and station_index <= previous.station_index:
                    self.trip_boundaries += 1
                    trajectory_id = self._trajectory_id(agent, force_new=True)
                else:
                    trajectory_id = self._trajectory_id(agent)

                hold_seconds = float(
                    self.action_to_hold_transform(action[0], self.config)
                )
                holds.append(hold_seconds)
                env_actions[agent] = [hold_seconds, 1.0]
                self.pending[agent] = PendingTransition(
                    raw_observation=raw_observation.copy(),
                    canonical_observation=canonical_observation.copy(),
                    action=action.copy(),
                    station_index=station_index,
                    episode_id=self.current_episode_id,
                    trajectory_id=trajectory_id,
                    reward_context=reward_context,
                )

            _, _, done = self.env.step_to_event(env_actions)
            for reward, duration in self._close_completed_transitions():
                transitions += 1
                rewards.append(float(reward))
                durations.append(float(duration))
            if done:
                for reward, duration in self._close_episode_transitions():
                    transitions += 1
                    rewards.append(float(reward))
                    durations.append(float(duration))
                self.episode_active = False
                if events < n_events:
                    self._start_episode()

        return {
            "events": events,
            "transitions": transitions,
            "episodes_started": self.episodes - start_episodes,
            "discarded_pending_total": self.discarded_pending,
            "trip_boundaries_total": self.trip_boundaries,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_delta_t": float(np.mean(durations)) if durations else 0.0,
            "mean_hold_seconds": float(np.mean(holds)) if holds else 0.0,
            "online_buffer_size": int(self.replay_buffer.online_size),
        }
