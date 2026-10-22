"""Proximal actor-release components for the H2O+ v13 development pilot."""

from .trainer import (
    PREDESIGNATED_CANDIDATE,
    PROXIMAL_RATES,
    V13_TRAINER_VARIANTS,
    TransitH2OProximalReleaseTrainerV13,
    proximal_retention,
)

__all__ = [
    "PREDESIGNATED_CANDIDATE",
    "PROXIMAL_RATES",
    "V13_TRAINER_VARIANTS",
    "TransitH2OProximalReleaseTrainerV13",
    "proximal_retention",
]
