"""Replay binding for the transit-smdp-v5 dataset schema."""

from __future__ import annotations

from bus_h2o.mdp_contract_v5 import ACTION_DIM, CANONICAL_STATE_DIM, SCHEMA_VERSION
from SimpleSAC.transit_v3.replay_buffer import TransitReplayBuffer


class TransitReplayBufferV5(TransitReplayBuffer):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(
            dataset_path,
            schema_version=SCHEMA_VERSION,
            state_dim=CANONICAL_STATE_DIM,
            action_dim=ACTION_DIM,
            **kwargs,
        )


__all__ = ["TransitReplayBufferV5"]
