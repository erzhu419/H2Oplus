"""Persistent online sampler for the true-welfare transit SMDP."""

from __future__ import annotations

import numpy as np

from bus_h2o.mdp_contract_v5 import (
    RAW_EVENT_STATE_DIM,
    SCHEMA_VERSION,
    TransitMDPConfigV5,
    continuous_time_discount_v5,
    policy_action_to_hold_seconds_v5,
    raw_event_to_canonical_v5,
    transition_duration_seconds_v5,
    transition_reward_v5,
)
from bus_h2o.passenger_time import (
    PassengerTimeSnapshot,
    aggregate_passenger_time_snapshots,
)
from SimpleSAC.transit_v3.sampler import AllLinesTransitSampler


def _raw_v5(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"v5 simulator observation has shape {observation.shape}; "
            f"expected ({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(observation).all():
        raise ValueError("v5 simulator observation contains non-finite values")
    return observation.copy()


def _event_passenger_time(env, agent) -> PassengerTimeSnapshot:
    return env.passenger_time_at_event(agent)


def _completed_passenger_times(env) -> dict[object, PassengerTimeSnapshot]:
    return env.consume_completed_agent_passenger_times()


def _terminal_passenger_time(env) -> PassengerTimeSnapshot:
    return aggregate_passenger_time_snapshots(
        line_env.passenger_time_ledger.snapshot()
        for line_env in env.line_map.values()
    )


def _passenger_time_duration(
    start: PassengerTimeSnapshot, end: PassengerTimeSnapshot
) -> float:
    return float(end.time_seconds - start.time_seconds)


class AllLinesTransitSamplerV5(AllLinesTransitSampler):
    def __init__(
        self,
        env,
        replay_buffer,
        *,
        station_max_by_line: np.ndarray,
        contract_config: TransitMDPConfigV5,
    ) -> None:
        super().__init__(
            env,
            replay_buffer,
            station_max_by_line=station_max_by_line,
            contract_config=contract_config,
            raw_transform=_raw_v5,
            canonical_transform=raw_event_to_canonical_v5,
            duration_transform=transition_duration_seconds_v5,
            discount_transform=continuous_time_discount_v5,
            action_to_hold_transform=policy_action_to_hold_seconds_v5,
            station_index_position=1,
            sampler_contract=SCHEMA_VERSION,
            reward_context_transform=_event_passenger_time,
            context_reward_transform=transition_reward_v5,
            context_duration_transform=_passenger_time_duration,
            completed_contexts_transform=_completed_passenger_times,
            terminal_context_transform=_terminal_passenger_time,
        )


__all__ = ["AllLinesTransitSamplerV5"]
