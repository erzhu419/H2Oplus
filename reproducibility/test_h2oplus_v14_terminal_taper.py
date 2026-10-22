from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v14.trainer import (
    BASE_PROXIMAL_RATE,
    FINAL_REGISTERED_ACTOR_UPDATE,
    PREDESIGNATED_CANDIDATE,
    TERMINAL_ACTOR_LR_MULTIPLIERS,
    V14_TRAINER_VARIANTS,
    TransitH2OTerminalQuarterTrainerV14,
    integrated_actor_lr_exposure,
    terminal_actor_lr_multiplier,
)


def test_registered_terminal_tapers_and_candidate_are_fixed() -> None:
    assert BASE_PROXIMAL_RATE == 5.0e-5
    assert TERMINAL_ACTOR_LR_MULTIPLIERS == {
        "terminal-half": 0.50,
        "terminal-quarter": 0.25,
        "terminal-zero": 0.0,
    }
    assert set(V14_TRAINER_VARIANTS) == set(TERMINAL_ACTOR_LR_MULTIPLIERS)
    assert PREDESIGNATED_CANDIDATE == "terminal-quarter"


def test_terminal_taper_boundaries_and_monotonicity() -> None:
    for terminal in TERMINAL_ACTOR_LR_MULTIPLIERS.values():
        assert terminal_actor_lr_multiplier(10_000, terminal) == 1.0
        assert terminal_actor_lr_multiplier(
            FINAL_REGISTERED_ACTOR_UPDATE, terminal
        ) == terminal
        values = [
            terminal_actor_lr_multiplier(update, terminal)
            for update in range(10_000, FINAL_REGISTERED_ACTOR_UPDATE + 1)
        ]
        assert all(left >= right for left, right in zip(values, values[1:]))
        assert all(terminal <= value <= 1.0 for value in values)


def test_terminal_taper_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        terminal_actor_lr_multiplier(-1, 0.25)
    with pytest.raises(ValueError):
        terminal_actor_lr_multiplier(10_000, -0.01)
    with pytest.raises(ValueError):
        terminal_actor_lr_multiplier(10_000, 1.01)


def test_full_rate_equivalent_exposure_is_ordered() -> None:
    exposures = {
        name: integrated_actor_lr_exposure(value)
        for name, value in TERMINAL_ACTOR_LR_MULTIPLIERS.items()
    }
    assert exposures["terminal-half"] > exposures["terminal-quarter"]
    assert exposures["terminal-quarter"] > exposures["terminal-zero"]
    assert exposures["terminal-half"] < 15_000
    assert exposures["terminal-zero"] > 0


def test_terminal_taper_scales_actor_only_before_v13_proximal_update() -> None:
    trainer = object.__new__(TransitH2OTerminalQuarterTrainerV14)
    parameter = torch.nn.Parameter(torch.ones(1))
    trainer.actor_optimizer = torch.optim.SGD([parameter], lr=0.3)
    trainer.online_critic_updates = FINAL_REGISTERED_ACTOR_UPDATE
    trainer._v14_last_terminal_taper_multiplier = 1.0
    observed: dict[str, float] = {}

    def fake_parent(*args, **kwargs):
        del args, kwargs
        observed["lr"] = trainer.actor_optimizer.param_groups[0]["lr"]
        zero = torch.tensor(0.0)
        return zero, zero, zero

    with patch(
        "SimpleSAC.transit_v13.trainer."
        "TransitH2OProximalReleaseTrainerV13._update_sac_actor",
        side_effect=fake_parent,
    ):
        trainer._update_sac_actor(
            torch.zeros((1, 1)),
            temperature_states=torch.zeros((1, 1)),
        )

    assert observed["lr"] == pytest.approx(0.3 * 0.25)
    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(0.3)
    assert trainer._v14_last_terminal_taper_multiplier == 0.25


def test_interventions_change_only_registered_terminal_multiplier() -> None:
    payloads = {
        name: object.__new__(trainer).v14_intervention
        for name, trainer in V14_TRAINER_VARIANTS.items()
    }
    ignored = {
        "variant",
        "terminal_actor_lr_multiplier",
        "full_rate_equivalent_actor_updates",
        "predesignated_candidate",
    }
    reference = {
        key: value
        for key, value in payloads["terminal-quarter"].items()
        if key not in ignored
    }
    for name, payload in payloads.items():
        assert payload["terminal_actor_lr_multiplier"] == (
            TERMINAL_ACTOR_LR_MULTIPLIERS[name]
        )
        assert payload["predesignated_candidate"] is (
            name == PREDESIGNATED_CANDIDATE
        )
        assert payload["checkpoint_selection_by_behavior"] is False
        assert math.isclose(
            payload["proximal_rate_per_actor_update"], BASE_PROXIMAL_RATE
        )
        assert {
            key: value for key, value in payload.items() if key not in ignored
        } == reference


def test_runtime_source_map_includes_every_inherited_trainer_layer() -> None:
    wrapper = ROOT / "SimpleSAC/train_transit_v14_terminal_taper.py"
    tree = ast.parse(wrapper.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_v14_source_sha256"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "paths"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Tuple)
    source_paths = {
        node.value
        for node in ast.walk(assignment.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for version in ("v10", "v11", "v12", "v13", "v14"):
        for name in ("__init__.py", "provenance.py", "trainer.py"):
            assert f"SimpleSAC/transit_{version}/{name}" in source_paths


def test_preflight_returns_before_materialization_or_training() -> None:
    wrapper = ROOT / "SimpleSAC/train_transit_v14_terminal_taper.py"
    tree = ast.parse(wrapper.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first = function.body[0]
    assert isinstance(first, ast.If)
    assert isinstance(first.test, ast.Name)
    assert first.test.id == "PREFLIGHT_ONLY"
    calls = [
        node.func.id
        for node in ast.walk(first)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls == ["_run_admission_preflight"]
    assert isinstance(first.body[-1], ast.Return)
