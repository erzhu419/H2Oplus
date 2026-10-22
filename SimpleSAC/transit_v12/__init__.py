"""Staged-release H2O+ development components for the v12 pilot."""

from .trainer import (
    ACTOR_ANCHOR_OFFLINE_FRACTION,
    MINIMUM_RELEASE_MULTIPLIER,
    RELEASE_END_CRITIC_UPDATES,
    RELEASE_START_CRITIC_UPDATES,
    V12_TRAINER_VARIANTS,
    anchored_actor_states,
    release_multiplier,
)

__all__ = [
    "ACTOR_ANCHOR_OFFLINE_FRACTION",
    "MINIMUM_RELEASE_MULTIPLIER",
    "RELEASE_END_CRITIC_UPDATES",
    "RELEASE_START_CRITIC_UPDATES",
    "V12_TRAINER_VARIANTS",
    "anchored_actor_states",
    "release_multiplier",
]
