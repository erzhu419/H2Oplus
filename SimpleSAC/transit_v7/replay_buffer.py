"""Replay binding for the transit-smdp-v7 dataset schema."""

from __future__ import annotations

from bus_h2o.mdp_contract_v7 import ACTION_DIM, CANONICAL_STATE_DIM, SCHEMA_VERSION
from SimpleSAC.transit_v3.replay_buffer import TransitReplayBuffer


class TransitReplayBufferV7(TransitReplayBuffer):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(
            dataset_path,
            schema_version=SCHEMA_VERSION,
            state_dim=CANONICAL_STATE_DIM,
            action_dim=ACTION_DIM,
            include_delta_t=True,
            **kwargs,
        )


__all__ = ["TransitReplayBufferV7"]
