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

from SimpleSAC.transit_v13.trainer import (
    PREDESIGNATED_CANDIDATE,
    PROXIMAL_RATES,
    REGISTERED_POST_10K_ACTOR_UPDATES,
    TransitH2OProximalMediumTrainerV13,
    V13_TRAINER_VARIANTS,
    proximal_retention,
)


def test_registered_doses_and_candidate_are_fixed() -> None:
    assert PROXIMAL_RATES == {
        "proximal-weak": 1.0e-5,
        "proximal-medium": 2.5e-5,
        "proximal-strong": 5.0e-5,
    }
    assert set(V13_TRAINER_VARIANTS) == set(PROXIMAL_RATES)
    assert PREDESIGNATED_CANDIDATE == "proximal-medium"
    assert REGISTERED_POST_10K_ACTOR_UPDATES == 15_000
    assert proximal_retention(2.5e-5, 15_000) == pytest.approx(
        0.6872860570757895
    )


def test_proximal_retention_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        proximal_retention(-1.0e-5, 10)
    with pytest.raises(ValueError):
        proximal_retention(1.0, 10)
    with pytest.raises(ValueError):
        proximal_retention(1.0e-5, -1)


def test_medium_pull_is_applied_after_parent_actor_update() -> None:
    trainer = object.__new__(TransitH2OProximalMediumTrainerV13)
    trainer.actor = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        trainer.actor.weight.zero_()
    anchor = torch.ones_like(trainer.actor.weight)
    trainer._v13_actor_anchor = (("weight", anchor),)
    trainer._v13_actor_anchor_sha256 = "registered"
    trainer._v13_last_distance_before = 0.0
    trainer._v13_last_distance_after = 0.0
    trainer._v13_proximal_update_executed = False

    def fake_parent(*args, **kwargs):
        del args, kwargs
        with torch.no_grad():
            trainer.actor.weight.fill_(0.25)
        zero = torch.tensor(0.0)
        return zero, zero, zero

    with patch(
        "SimpleSAC.transit_v12.trainer.TransitH2OActorRampTrainerV12._update_sac_actor",
        side_effect=fake_parent,
    ):
        trainer._update_sac_actor(
            torch.zeros((2, 2)),
            temperature_states=torch.zeros((2, 2)),
        )

    expected = 0.25 * (1.0 - PROXIMAL_RATES["proximal-medium"]) + (
        PROXIMAL_RATES["proximal-medium"]
    )
    assert torch.allclose(trainer.actor.weight, torch.full_like(anchor, expected))
    assert trainer._v13_proximal_update_executed is True
    assert trainer._v13_last_distance_after == pytest.approx(
        trainer._v13_last_distance_before
        * (1.0 - PROXIMAL_RATES["proximal-medium"])
    )


def test_interventions_change_only_registered_proximal_rate() -> None:
    interventions = {
        name: object.__new__(trainer).v13_intervention
        for name, trainer in V13_TRAINER_VARIANTS.items()
    }
    for name, payload in interventions.items():
        assert payload["proximal_rate_per_actor_update"] == PROXIMAL_RATES[name]
        assert payload["predesignated_candidate"] is (
            name == PREDESIGNATED_CANDIDATE
        )
        assert payload["checkpoint_selection_by_behavior"] is False
        assert math.isclose(
            payload["no_gradient_anchor_retention_at_25k"],
            proximal_retention(PROXIMAL_RATES[name], 15_000),
        )


def test_runtime_source_map_includes_every_inherited_trainer_layer() -> None:
    wrapper = ROOT / "SimpleSAC/train_transit_v13_proximal_release.py"
    tree = ast.parse(wrapper.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_v13_source_sha256"
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
    for version in ("v10", "v11", "v12", "v13"):
        for name in ("__init__.py", "provenance.py", "trainer.py"):
            assert f"SimpleSAC/transit_{version}/{name}" in source_paths


def test_preflight_returns_before_materialization_or_training() -> None:
    wrapper = ROOT / "SimpleSAC/train_transit_v13_proximal_release.py"
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
