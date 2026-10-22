"""Persistent all-lines online sampler for transit-smdp-v4."""

from __future__ import annotations

import numpy as np

from bus_h2o.mdp_contract_v4 import (
    RAW_EVENT_STATE_DIM,
    SCHEMA_VERSION,
    TransitMDPConfigV4,
    continuous_time_discount_v4,
    policy_action_to_hold_seconds_v4,
    raw_event_to_canonical_v4,
    transition_duration_seconds_v4,
    transition_reward_v4,
)
from SimpleSAC.transit_v3.sampler import AllLinesTransitSampler


def _raw_v4(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.shape != (RAW_EVENT_STATE_DIM,):
        raise ValueError(
            f"v4 simulator observation has shape {observation.shape}; "
            f"expected ({RAW_EVENT_STATE_DIM},)"
        )
    if not np.isfinite(observation).all():
        raise ValueError("v4 simulator observation contains non-finite values")
    return observation.copy()


def _reward_v4(previous_raw, next_raw, action, config):
    return transition_reward_v4(previous_raw, action, next_raw, config)


class AllLinesTransitSamplerV4(AllLinesTransitSampler):
    def __init__(
        self,
        env,
        replay_buffer,
        *,
        station_max_by_line: np.ndarray,
        contract_config: TransitMDPConfigV4,
    ) -> None:
        super().__init__(
            env,
            replay_buffer,
            station_max_by_line=station_max_by_line,
            contract_config=contract_config,
            raw_transform=_raw_v4,
            canonical_transform=raw_event_to_canonical_v4,
            reward_transform=_reward_v4,
            duration_transform=transition_duration_seconds_v4,
            discount_transform=continuous_time_discount_v4,
            action_to_hold_transform=policy_action_to_hold_seconds_v4,
            station_index_position=1,
            sampler_contract=SCHEMA_VERSION,
        )


__all__ = ["AllLinesTransitSamplerV4"]
