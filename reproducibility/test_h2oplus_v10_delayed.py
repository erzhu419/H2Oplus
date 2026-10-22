from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v10.provenance import checkpoint_state_sha256
from SimpleSAC.transit_v10.trainer import (
    TRANSITION_END_EVENTS,
    TRANSITION_START_EVENTS,
    UnitWeightDynamicsRatioV10,
    V10_TRAINER_VARIANTS,
    delayed_offline_fraction,
    transition_is_active,
)


def test_launcher_runtime_schemas_match_deployment_builder() -> None:
    launcher = ast.parse((ROOT / "SimpleSAC/train_transit_v10_delayed.py").read_text())
    schema_names = {"PROTOCOL_SCHEMA", "ADMISSION_SCHEMA"}
    launcher_assignments = {
        target.id: ast.literal_eval(node.value)
        for node in launcher.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in schema_names
    }
    deployment_builder = ast.parse(
        (
            ROOT / "reproducibility/prepare_h2oplus_v10_delayed_pilot_v4.py"
        ).read_text()
    )
    builder_assignments = {
        target.id: ast.literal_eval(node.value)
        for node in deployment_builder.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in schema_names
    }
    expected = {
        "PROTOCOL_SCHEMA": "transit-h2oplus-v10-delayed-protocol-v4",
        "ADMISSION_SCHEMA": "transit-training-admission-v10-delayed-development-v4",
    }
    assert launcher_assignments == expected
    assert builder_assignments == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (0, 0.50),
        (9_999, 0.50),
        (10_000, 0.50),
        (17_500, 0.375),
        (25_000, 0.25),
        (100_000, 0.25),
    ],
)
def test_delayed_replay_schedule(events: int, expected: float) -> None:
    assert delayed_offline_fraction(events) == pytest.approx(expected)


def test_transition_starts_strictly_after_common_prefix() -> None:
    assert TRANSITION_START_EVENTS == 10_000
    assert TRANSITION_END_EVENTS == 25_000
    assert not transition_is_active(10_000)
    assert transition_is_active(10_001)


def test_variants_isolate_registered_factors() -> None:
    flags = {
        name: (
            trainer.USE_DELAYED_REPLAY,
            trainer.USE_DELAYED_ACTOR_STATES,
            trainer.USE_DELAYED_UPDATE_ORDER,
        )
        for name, trainer in V10_TRAINER_VARIANTS.items()
    }
    assert flags == {
        "shared-prefix": (False, False, False),
        "unit-control": (False, False, False),
        "delayed-replay": (True, False, False),
        "delayed-actor": (False, True, False),
        "delayed-order": (False, False, True),
        "delayed-combined": (True, True, True),
    }


def test_shared_prefix_actor_states_match_v7_mixed_replay_contract() -> None:
    trainer = object.__new__(V10_TRAINER_VARIANTS["shared-prefix"])
    offline = {"observations": torch.tensor([[1.0, 2.0]])}
    online = {"observations": torch.tensor([[3.0, 4.0]])}

    states = trainer._actor_states(offline, online)

    assert torch.equal(states, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_unit_ratio_weights_do_not_apply_classifier_ratio() -> None:
    instance = object.__new__(UnitWeightDynamicsRatioV10)
    joint = torch.tensor([[5.0, -1.0], [-2.0, 4.0]])
    state_action = torch.tensor([[-3.0, 2.0], [3.0, -4.0]])
    with patch.object(
        UnitWeightDynamicsRatioV10,
        "logits",
        return_value=(joint, state_action),
    ):
        weights, metrics = UnitWeightDynamicsRatioV10.weights(instance, {})
    assert torch.equal(weights, torch.ones_like(weights))
    assert metrics["ratio/v10_unit_weights"].item() == 1.0
    assert metrics["ratio/weight_min"].item() == 1.0
    assert metrics["ratio/weight_max"].item() == 1.0


def _synthetic_checkpoint(path: Path, *, extra: dict[str, object]) -> None:
    torch.save(
        {
            "checkpoint_version": "test-v10",
            "total_updates": 55_000,
            "actor": {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)},
            "optimizer": {
                "state": {0: {"step": torch.tensor(7), "exp_avg": torch.ones(4)}},
                "param_groups": [{"lr": 3e-4, "params": [0]}],
            },
            "numpy_rng_state": (
                "MT19937",
                np.arange(8, dtype=np.uint32),
                3,
                0,
                0.0,
            ),
            "extra": extra,
        },
        path,
    )


def test_checkpoint_state_hash_excludes_only_provenance_envelope(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    changed = tmp_path / "changed.pt"
    _synthetic_checkpoint(first, extra={"run_signature": "prefix"})
    payload = torch.load(first, map_location="cpu", weights_only=False)
    payload["extra"] = {"run_signature": "branch", "variant": "delayed-replay"}
    torch.save(payload, second)
    payload["actor"]["weight"][0, 0] = -1.0
    torch.save(payload, changed)
    assert checkpoint_state_sha256(first) == checkpoint_state_sha256(second)
    assert checkpoint_state_sha256(first) != checkpoint_state_sha256(changed)


def test_checkpoint_state_hash_is_stable_across_processes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _synthetic_checkpoint(checkpoint, extra={"run_signature": "prefix"})
    code = (
        "from pathlib import Path; "
        "from SimpleSAC.transit_v10.provenance import checkpoint_state_sha256; "
        f"print(checkpoint_state_sha256(Path({json.dumps(str(checkpoint))})))"
    )
    digests = {
        subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        for _ in range(3)
    }
    assert len(digests) == 1
