from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from bus_h2o.mdp_contract_v7 import CANONICAL_STATE_DIM, TransitMDPConfigV7
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig
from SimpleSAC.transit_v7.networks import TransitDynamicsRatioV7
from SimpleSAC.transit_v9.trainer import (
    ReliabilityGatedDynamicsRatioV9,
    TransitH2OAdaptationTrainerV9,
    TransitH2OCombinedTrainerV9,
    TransitH2ORatioGatedTrainerV9,
    authorize_registered_initializer_bridge,
    offline_fraction_for_online_events,
)


def _ratio_batch(size: int = 8) -> dict[str, torch.Tensor]:
    observations = torch.zeros(size, CANONICAL_STATE_DIM)
    return {
        "observations": observations,
        "actions": torch.zeros(size, 1),
        "next_observations": observations.clone(),
        "delta_t_seconds": torch.full((size,), 60.0),
    }


def _good_validation_logits(size: int = 8):
    real_joint = torch.full((size,), 2.0)
    real_state_action = torch.zeros(size)
    sim_joint = torch.full((size,), -2.0)
    sim_state_action = torch.zeros(size)
    return iter(
        ((real_joint, real_state_action), (sim_joint, sim_state_action))
    )


def _v7_initializer_base(validator):
    return SimpleNamespace(
        V7_SHARED_IQL_BRANCH_METHODS={
            "h2oplus-no-ratio",
            "h2oplus-lambda0",
            "h2oplus-lambda0p2",
            "h2oplus-lambda0p5",
            "h2oplus-lambda1",
        },
        validate_v7_offline_initializer_runtime=validator,
    )


def test_v9_initializer_bridge_is_process_local_and_fail_closed() -> None:
    validator = object()
    base = _v7_initializer_base(validator)
    authorized = authorize_registered_initializer_bridge(
        base, method="h2oplus", validator=validator
    )
    assert authorized == frozenset(base.V7_SHARED_IQL_BRANCH_METHODS)
    assert "h2oplus" in authorized

    with pytest.raises(RuntimeError, match="validator was not installed"):
        authorize_registered_initializer_bridge(
            _v7_initializer_base(object()),
            method="h2oplus",
            validator=validator,
        )

    drifted = _v7_initializer_base(validator)
    drifted.V7_SHARED_IQL_BRANCH_METHODS.add("unexpected-branch")
    with pytest.raises(RuntimeError, match="registry drifted"):
        authorize_registered_initializer_bridge(
            drifted, method="h2oplus", validator=validator
        )

    with pytest.raises(ValueError, match="only authorizes h2oplus"):
        authorize_registered_initializer_bridge(
            _v7_initializer_base(validator),
            method="sac-online",
            validator=validator,
        )


def test_v9_ratio_gate_defaults_closed_and_uses_hysteresis(monkeypatch) -> None:
    ratio = ReliabilityGatedDynamicsRatioV9(
        (16, 16), min_weight=1e-5, max_weight=1.0
    )
    batch = _ratio_batch()
    heterogeneous = torch.tensor([-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    monkeypatch.setattr(
        ratio,
        "logits",
        lambda unused: (heterogeneous, torch.zeros_like(heterogeneous)),
    )
    weights, diagnostics = ratio.weights(batch)
    torch.testing.assert_close(weights, torch.ones_like(weights))
    assert diagnostics["ratio/v9_gate_open"] == 0

    for expected_open in (0, 1):
        logits = _good_validation_logits()
        monkeypatch.setattr(ratio, "logits", lambda unused: next(logits))
        metrics = ratio.evaluate(batch, batch)
        assert metrics["ratio/validation_v9_gate_pass"] == 1
        assert metrics["ratio/validation_v9_gate_open"] == expected_open

    monkeypatch.setattr(
        ratio,
        "logits",
        lambda unused: (heterogeneous, torch.zeros_like(heterogeneous)),
    )
    weights, diagnostics = ratio.weights(batch)
    assert diagnostics["ratio/v9_gate_open"] == 1
    assert float(weights.min()) >= ratio.APPLIED_MIN_WEIGHT
    assert float(weights.max()) <= ratio.APPLIED_MAX_WEIGHT
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(diagnostics["ratio/ess_fraction"]) >= ratio.TARGET_ESS_FRACTION - 1e-5
    assert not torch.equal(weights, torch.ones_like(weights))

    for expected_open in (1, 0):
        zeros = iter(
            (
                (torch.zeros(8), torch.zeros(8)),
                (torch.zeros(8), torch.zeros(8)),
            )
        )
        monkeypatch.setattr(ratio, "logits", lambda unused: next(zeros))
        metrics = ratio.evaluate(batch, batch)
        assert metrics["ratio/validation_v9_gate_pass"] == 0
        assert metrics["ratio/validation_v9_gate_open"] == expected_open


def test_v9_ratio_loads_v7_initializer_closed_and_v9_state_roundtrips() -> None:
    legacy = TransitDynamicsRatioV7(
        (16, 16), min_weight=1e-5, max_weight=1.0
    )
    ratio = ReliabilityGatedDynamicsRatioV9(
        (16, 16), min_weight=1e-5, max_weight=1.0
    )
    ratio.load_state_dict(legacy.state_dict())
    assert ratio.v9_gate_open.item() == 0
    assert ratio.v9_validation_count.item() == 0
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(ratio.state_dict()[key], value)

    ratio.v9_gate_open.fill_(1)
    ratio.v9_validation_count.fill_(7)
    ratio.v9_tempering_exponent.fill_(0.4)
    restored = ReliabilityGatedDynamicsRatioV9(
        (16, 16), min_weight=1e-5, max_weight=1.0
    )
    restored.load_state_dict(ratio.state_dict())
    assert restored.v9_gate_open.item() == 1
    assert restored.v9_validation_count.item() == 7
    assert restored.v9_tempering_exponent.item() == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("events", "expected"),
    (
        (0, 0.50),
        (1_000, 0.50),
        (5_500, 0.375),
        (10_000, 0.25),
        (17_500, 0.175),
        (25_000, 0.10),
        (100_000, 0.10),
    ),
)
def test_v9_offline_replay_schedule(events: int, expected: float) -> None:
    assert offline_fraction_for_online_events(events) == pytest.approx(expected)


def test_v9_variants_keep_interventions_separate() -> None:
    replay = SimpleNamespace(
        contract_config=asdict(TransitMDPConfigV7()), online_size=0
    )
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=4,
        target_subset_size=2,
        offline_objective="iql",
        online_backup="h2oplus",
        target_entropy=0.0,
        ratio_min_weight=1e-5,
        ratio_max_weight=1.0,
    )
    ratio_only = TransitH2ORatioGatedTrainerV9(replay, config=config, seed=1)
    adaptation = TransitH2OAdaptationTrainerV9(replay, config=config, seed=1)
    combined = TransitH2OCombinedTrainerV9(replay, config=config, seed=1)
    assert isinstance(ratio_only.ratio, ReliabilityGatedDynamicsRatioV9)
    assert isinstance(adaptation.ratio, TransitDynamicsRatioV7)
    assert not isinstance(adaptation.ratio, ReliabilityGatedDynamicsRatioV9)
    assert isinstance(combined.ratio, ReliabilityGatedDynamicsRatioV9)
    assert ratio_only.v9_intervention["adaptive_online_phase"] is False
    assert adaptation.v9_intervention["reliability_gated_ratio"] is False
    assert combined.v9_intervention["adaptive_online_phase"] is True
    assert combined.v9_intervention["reliability_gated_ratio"] is True


def test_v9_adaptation_uses_online_actor_states_and_restores_config(monkeypatch) -> None:
    trainer = object.__new__(TransitH2OAdaptationTrainerV9)
    original = TransitTrainerConfig(
        online_fraction=0.5,
        online_update_order="actor-before-critic",
    )
    trainer.config = original
    trainer.replay_buffer = SimpleNamespace(online_size=10_000)
    trainer.online_critic_updates = 0
    trainer._online_actor_state_batches = []
    trainer._online_temperature_state_batches = []
    observed = {}

    def fake_update(self, *, online, online_only=False, collect_metrics=True):
        observed.update(
            online=online,
            online_only=online_only,
            online_fraction=self.config.online_fraction,
            online_update_order=self.config.online_update_order,
        )
        return {"train/update": 1.0} if collect_metrics else {}

    monkeypatch.setattr(TransitH2OTrainer, "update", fake_update)
    metrics = trainer.update(online=True)
    assert observed == {
        "online": True,
        "online_only": False,
        "online_fraction": 0.75,
        "online_update_order": "critic-before-actor",
    }
    assert trainer.config is original
    assert metrics["v9/offline_replay_fraction"] == pytest.approx(0.25)

    offline = {"observations": torch.zeros(3, 2)}
    online = {"observations": torch.ones(2, 2)}
    torch.testing.assert_close(
        trainer._actor_states(offline, online), online["observations"]
    )
    for _ in range(trainer.ACTOR_BURN_IN_UPDATES):
        assert (
            trainer._scheduled_online_actor_states(
                online["observations"], online["observations"]
            )
            is None
        )
    assert trainer.online_critic_updates == trainer.ACTOR_BURN_IN_UPDATES
    scheduled = trainer._scheduled_online_actor_states(
        online["observations"], online["observations"]
    )
    assert scheduled is not None
