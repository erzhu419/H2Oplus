"""Terminal actor-taper components for the H2O+ v14 development pilot."""

from .trainer import (
    BASE_PROXIMAL_RATE,
    PREDESIGNATED_CANDIDATE,
    TERMINAL_ACTOR_LR_MULTIPLIERS,
    V14_TRAINER_VARIANTS,
    TransitH2OTerminalTaperTrainerV14,
    integrated_actor_lr_exposure,
    terminal_actor_lr_multiplier,
)

__all__ = [
    "BASE_PROXIMAL_RATE",
    "PREDESIGNATED_CANDIDATE",
    "TERMINAL_ACTOR_LR_MULTIPLIERS",
    "V14_TRAINER_VARIANTS",
    "TransitH2OTerminalTaperTrainerV14",
    "integrated_actor_lr_exposure",
    "terminal_actor_lr_multiplier",
]
