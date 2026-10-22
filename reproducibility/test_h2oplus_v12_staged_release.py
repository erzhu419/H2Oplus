from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v11.trainer import ACTOR_HOLD_CRITIC_UPDATES
from SimpleSAC.transit_v12.trainer import (
    ACTOR_ANCHOR_OFFLINE_FRACTION,
    MINIMUM_RELEASE_MULTIPLIER,
    REGISTERED_ACTOR_ANCHOR_OFFLINE_ROWS,
    REGISTERED_ACTOR_BATCH_SIZE,
    RELEASE_END_CRITIC_UPDATES,
    RELEASE_START_CRITIC_UPDATES,
    TransitH2OActorRampTrainerV12,
    TransitH2OStagedCombinedTrainerV12,
    TransitH2OTemperatureRampTrainerV12,
    V12_TRAINER_VARIANTS,
    anchored_actor_states,
    release_multiplier,
)
from reproducibility.prepare_h2oplus_v12_staged_release import (
    BRANCH_VARIANTS,
    _source_map,
)


def test_release_schedule_is_registered_and_monotone() -> None:
    assert RELEASE_START_CRITIC_UPDATES == ACTOR_HOLD_CRITIC_UPDATES == 5_000
    assert RELEASE_END_CRITIC_UPDATES == 10_000
    assert MINIMUM_RELEASE_MULTIPLIER == 0.10
    assert release_multiplier(0) == 0.0
    assert release_multiplier(5_000) == 0.0
    assert release_multiplier(5_001) == pytest.approx(0.10018)
    assert release_multiplier(7_500) == pytest.approx(0.55)
    assert release_multiplier(10_000) == 1.0
    assert release_multiplier(15_000) == 1.0
    values = [release_multiplier(update) for update in range(5_001, 10_001)]
    assert values == sorted(values)


def test_actor_anchor_preserves_online_batch_size_and_registered_mix() -> None:
    assert ACTOR_ANCHOR_OFFLINE_FRACTION == 0.20
    assert REGISTERED_ACTOR_BATCH_SIZE == 128
    assert REGISTERED_ACTOR_ANCHOR_OFFLINE_ROWS == 26
    offline = torch.arange(128, dtype=torch.float32).reshape(128, 1)
    online = torch.arange(1_000, 1_128, dtype=torch.float32).reshape(128, 1)
    anchored = anchored_actor_states(offline, online)
    assert anchored.shape == online.shape
    assert torch.equal(anchored[:26], offline[:26])
    assert torch.equal(anchored[26:], online[:102])
    assert REGISTERED_ACTOR_ANCHOR_OFFLINE_ROWS / REGISTERED_ACTOR_BATCH_SIZE == (
        0.203125
    )


@pytest.mark.parametrize(
    ("trainer_class", "expected_actor", "expected_temperature"),
    [
        (TransitH2OActorRampTrainerV12, 0.55, 1.0),
        (TransitH2OTemperatureRampTrainerV12, 1.0, 0.55),
        (TransitH2OStagedCombinedTrainerV12, 0.55, 0.55),
    ],
)
def test_optimizer_release_factors_are_isolated_and_restored(
    trainer_class: type,
    expected_actor: float,
    expected_temperature: float,
) -> None:
    trainer = object.__new__(trainer_class)
    trainer.online_critic_updates = 7_500
    trainer.actor_optimizer = SimpleNamespace(param_groups=[{"lr": 3e-4}])
    trainer.alpha_optimizer = SimpleNamespace(param_groups=[{"lr": 4e-4}])
    observed: dict[str, float] = {}

    def fake_update(*args, **kwargs):
        del args, kwargs
        observed["actor_lr"] = trainer.actor_optimizer.param_groups[0]["lr"]
        observed["temperature_lr"] = trainer.alpha_optimizer.param_groups[0]["lr"]
        zero = torch.tensor(0.0)
        return zero, zero, zero

    with patch(
        "SimpleSAC.transit_v11.trainer.TransitH2OActorHoldTrainerV11._update_sac_actor",
        side_effect=fake_update,
    ):
        trainer._update_sac_actor(
            torch.zeros((2, 3)),
            temperature_states=torch.zeros((2, 3)),
        )

    assert observed["actor_lr"] == pytest.approx(3e-4 * expected_actor)
    assert observed["temperature_lr"] == pytest.approx(
        4e-4 * expected_temperature
    )
    assert trainer.actor_optimizer.param_groups[0]["lr"] == 3e-4
    assert trainer.alpha_optimizer.param_groups[0]["lr"] == 4e-4


def test_variant_matrix_is_complete_and_single_factor() -> None:
    assert BRANCH_VARIANTS == (
        "abrupt-control",
        "actor-ramp",
        "temperature-ramp",
        "actor-anchor",
        "staged-combined",
    )
    assert set(V12_TRAINER_VARIANTS) == {"shared-prefix", *BRANCH_VARIANTS}
    interventions = {
        name: object.__new__(trainer).v12_intervention
        for name, trainer in V12_TRAINER_VARIANTS.items()
    }
    assert not any(
        interventions["abrupt-control"][key]
        for key in (
            "actor_learning_rate_ramp",
            "temperature_learning_rate_ramp",
            "actor_state_anchor",
        )
    )
    assert interventions["actor-ramp"]["actor_learning_rate_ramp"] is True
    assert interventions["temperature-ramp"][
        "temperature_learning_rate_ramp"
    ] is True
    assert interventions["actor-anchor"]["actor_state_anchor"] is True
    assert all(
        interventions["staged-combined"][key]
        for key in (
            "actor_learning_rate_ramp",
            "temperature_learning_rate_ramp",
            "actor_state_anchor",
        )
    )


def test_launcher_and_generator_schemas_match() -> None:
    launcher_path = ROOT / "SimpleSAC/train_transit_v12_staged_release.py"
    generator_path = ROOT / "reproducibility/prepare_h2oplus_v12_staged_release.py"
    launcher = ast.parse(launcher_path.read_text())
    generator = ast.parse(generator_path.read_text())

    def constants(tree: ast.Module) -> dict[str, str]:
        names = {"PROTOCOL_SCHEMA", "ADMISSION_SCHEMA"}
        return {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in names
        }

    expected = {
        "PROTOCOL_SCHEMA": "transit-h2oplus-v12-staged-release-protocol-v2",
        "ADMISSION_SCHEMA": (
            "transit-training-admission-v12-staged-release-development-v2"
        ),
    }
    assert constants(launcher) == expected
    assert constants(generator) == expected


def test_source_contract_contains_transitive_v10_v11_dependencies() -> None:
    expected = {
        "SimpleSAC/train_transit_v12_staged_release.py",
        "SimpleSAC/transit_v10/__init__.py",
        "SimpleSAC/transit_v10/provenance.py",
        "SimpleSAC/transit_v10/trainer.py",
        "SimpleSAC/transit_v11/__init__.py",
        "SimpleSAC/transit_v11/provenance.py",
        "SimpleSAC/transit_v11/trainer.py",
        "SimpleSAC/transit_v12/__init__.py",
        "SimpleSAC/transit_v12/provenance.py",
        "SimpleSAC/transit_v12/trainer.py",
    }
    assert set(_source_map()) == expected
    launcher = (ROOT / "SimpleSAC/train_transit_v12_staged_release.py").read_text()
    verifier = (
        ROOT / "reproducibility/verify_h2oplus_v12_staged_release_deployment_v2.py"
    ).read_text()
    for path in expected:
        assert path in launcher or path == "SimpleSAC/train_transit_v12_staged_release.py"
        assert path in verifier


def test_frozen_scope_prohibits_local_sumo_and_formal_access() -> None:
    source = (
        ROOT / "reproducibility/prepare_h2oplus_v12_staged_release.py"
    ).read_text()
    assert '"local_training_or_sumo_allowed": False' in source
    assert '"formal_held_out_outcomes_read": False' in source
    assert "do not launch a formal confirmation stage" in source
