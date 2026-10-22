from __future__ import annotations

import numpy as np
import torch
from dataclasses import asdict, replace
from torch import nn

from bus_h2o.mdp_contract_v7 import (
    CANONICAL_PREFIX_NAMES,
    CANONICAL_SLOT_WIDTH,
    CANONICAL_STATE_DIM,
    RAW_EVENT_STATE_NAMES,
    TransitMDPConfigV7,
    pack_event_batch_v7,
    raw_state_to_canonical_v7,
)
from SimpleSAC.transit_v7.networks import (
    FEATURE_DIM,
    HurdleTransitActorV7,
    TransitDynamicsRatioV7,
    TransitQEnsembleV7,
    TransitStateEncoderV7,
    TransitValueV7,
    structured_slot_features_v7,
)
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


class _V7ReplayStub:
    contract_config = asdict(TransitMDPConfigV7())
    online_size = 0


class _V7OfflineReplay(_V7ReplayStub):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(7001)
        self.rows = {
            "observations": torch.rand(
                12, CANONICAL_STATE_DIM, generator=generator
            ),
            "actions": torch.rand(12, 1, generator=generator) * 2.0 - 1.0,
            "rewards": -torch.rand(12, generator=generator),
            "next_observations": torch.rand(
                12, CANONICAL_STATE_DIM, generator=generator
            ),
            "discounts": torch.full((12,), 0.99),
            "terminals": torch.zeros(12),
            "delta_t_seconds": torch.full((12,), 60.0),
        }

    def sample_mixed(self, batch_size, online_fraction, rng):
        assert online_fraction == 0.0
        indices = torch.as_tensor(rng.integers(0, 12, size=batch_size))
        return (
            {key: value.index_select(0, indices) for key, value in self.rows.items()},
            None,
        )


class _CountingEncoder(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(states)


def _event(line: int, station: int, time: float) -> np.ndarray:
    values = {
        "line_id": line,
        "station_index": station,
        "sim_time_seconds": time,
        "forward_headway_seconds": 300.0,
        "backward_headway_seconds": 360.0,
        "target_forward_headway_seconds": 330.0,
        "target_backward_headway_seconds": 330.0,
        "eligible_stranded_passengers": 5,
        "onboard_passengers": 20,
        "eligible_arrival_rate_per_second": 0.02,
        "base_dwell_seconds": 9,
        "forward_bus_present": 1,
        "backward_bus_present": 1,
        "bus_capacity": 50,
        "system_waiting_passengers": 100,
        "system_in_vehicle_passengers": 200,
    }
    return np.asarray([values[name] for name in RAW_EVENT_STATE_NAMES], dtype=np.float32)


def _states() -> torch.Tensor:
    config = TransitMDPConfigV7()
    station_max = np.full(12, 24.0)
    batch = np.stack((_event(1, 3, 100.0), _event(11, 4, 100.0), _event(7, 2, 100.0)))
    rows = [
        pack_event_batch_v7(batch, 0, np.asarray([-1.0, -1.0, -1.0]), config),
        pack_event_batch_v7(batch, 1, np.asarray([-1.0, -1.0, -1.0]), config),
        pack_event_batch_v7(batch, 2, np.asarray([-1.0, 0.25, -1.0]), config),
    ]
    canonical = raw_state_to_canonical_v7(np.stack(rows), station_max, config)
    return torch.as_tensor(canonical, dtype=torch.float32)


def test_v7_structured_features_disambiguate_selected_zero_action() -> None:
    decoded = structured_slot_features_v7(_states()[1:2])
    assert decoded["selected"][0, 0]
    assert decoded["current"][0, 1]
    assert decoded["future"][0, 2]
    assert decoded["local"][0, 0, -4] == -1.0
    assert decoded["local"][0, 0, -3] == 1.0
    assert decoded["local"][0, 2, -4] == -1.0
    assert decoded["local"][0, 2, -3] == 0.0


def test_v7_encoder_masks_inactive_slots() -> None:
    states = _states()[0:1]
    changed = states.clone()
    inactive_start = len(CANONICAL_PREFIX_NAMES) + 3 * CANONICAL_SLOT_WIDTH
    changed[:, inactive_start + 1 : inactive_start + CANONICAL_SLOT_WIDTH] = 0.73
    encoder = TransitStateEncoderV7().eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(states), encoder(changed))
    assert encoder(states).shape == (1, FEATURE_DIM)


def test_v7_network_and_ratio_shapes_are_finite() -> None:
    states = _states()
    actions = torch.tensor([[-1.0], [0.0], [0.5]])
    actor = HurdleTransitActorV7((32, 32))
    critic = TransitQEnsembleV7(3, (32, 32))
    value = TransitValueV7((32, 32))
    ratio = TransitDynamicsRatioV7((32, 32))

    actor_actions, log_prob = actor(states, deterministic=True)
    assert actor_actions.shape == (3, 1)
    assert log_prob.shape == (3,)
    assert critic(states, actions).shape == (3, 3)
    assert value(states).shape == (3,)
    batch = {
        "observations": states,
        "actions": actions,
        "next_observations": states.roll(1, dims=0),
        "discounts": torch.ones(3),
        "delta_t_seconds": torch.tensor([0.0, 60.0, 24_000.0]),
    }
    joint_features = ratio._joint_features(batch)
    torch.testing.assert_close(
        joint_features[:, -1], torch.tensor([0.0, 60.0 / 24_000.0, 1.0])
    )
    counting_encoder = _CountingEncoder(ratio.feature_fn)
    ratio.feature_fn = counting_encoder
    joint, state_action = ratio.logits(batch)
    assert counting_encoder.calls == 1
    assert joint.shape == state_action.shape == (3,)
    assert torch.isfinite(joint).all()
    assert any(name.startswith("feature_fn.") for name in ratio.state_dict())
    assert states.shape[1] == CANONICAL_STATE_DIM


def test_v7_critic_feature_reuse_matches_repeated_state_evaluation() -> None:
    states = _states()
    critic = TransitQEnsembleV7(3, (32, 32))
    repeats = 4
    actions = torch.linspace(-1.0, 1.0, len(states) * repeats).reshape(-1, 1)
    repeated_states = states[:, None, :].expand(-1, repeats, -1).reshape(
        len(states) * repeats, -1
    )
    features = critic.encode_states(states)
    repeated_features = features[:, None, :].expand(-1, repeats, -1).reshape(
        len(states) * repeats, -1
    )
    torch.testing.assert_close(
        critic(repeated_states, actions),
        critic.forward_features(repeated_features, actions),
    )
    indices = torch.tensor([2, 0], dtype=torch.long)
    torch.testing.assert_close(
        critic.forward_subset(repeated_states, actions, indices),
        critic(repeated_states, actions).index_select(0, indices),
    )
    torch.testing.assert_close(
        critic.forward_features_subset(repeated_features, actions, indices),
        critic.forward_features(repeated_features, actions).index_select(0, indices),
    )


def test_v7_actor_repeated_components_match_direct_encoding() -> None:
    states = _states()
    actor = HurdleTransitActorV7((32, 32))
    repeats = 3
    repeated_states = states[:, None, :].expand(-1, repeats, -1).reshape(
        len(states) * repeats, -1
    )
    direct = actor.soft_action_components(
        repeated_states, deterministic_positive=True
    )
    reused = actor.repeated_soft_action_components(
        states, repeats, deterministic_positive=True
    )
    for key in direct:
        torch.testing.assert_close(direct[key], reused[key])


def test_v7_compilation_includes_all_structured_encoders() -> None:
    trainer = TransitH2OTrainerV7(
        _V7ReplayStub(),
        config=TransitTrainerConfig(
            ensemble_size=2,
            hidden_dim=16,
            batch_size=4,
            target_subset_size=2,
            offline_objective="iql",
            online_backup="h2oplus",
        ),
        seed=81,
    )
    compiled = set(trainer.compile_networks())
    assert {
        "actor_encoder",
        "critic_encoder",
        "target_critic_encoder",
        "value_encoder",
        "ratio_encoder",
    } <= compiled
    assert all("_orig_mod" not in key for key in trainer.actor.state_dict())


def _assert_nested_equal(first, second) -> None:
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_v7_shared_iql_initializer_is_bit_exact_across_online_only_differences(
    tmp_path,
) -> None:
    source_config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=4,
        target_subset_size=2,
        target_entropy=0.0,
        offline_objective="iql",
        online_backup="h2oplus",
        mixed_backup_lambda=0.1,
        use_ratio_correction=True,
    )
    target_config = replace(
        source_config,
        mixed_backup_lambda=0.5,
        use_ratio_correction=False,
    )
    source = TransitH2OTrainerV7(
        _V7OfflineReplay(), config=source_config, seed=91
    )
    for _ in range(3):
        source.update(online=False, collect_metrics=False)
    initializer = tmp_path / "offline_initializer.pt"
    source.save_offline_initializer(
        initializer,
        extra={"phase": "offline_pretraining", "training_seed": 91},
    )

    independent = TransitH2OTrainerV7(
        _V7OfflineReplay(), config=target_config, seed=91
    )
    for _ in range(3):
        independent.update(online=False, collect_metrics=False)
    expected_torch_rng = torch.get_rng_state().clone()
    expected_numpy_rng = independent.rng.bit_generator.state

    branched = TransitH2OTrainerV7(
        _V7OfflineReplay(), config=target_config, seed=91
    )
    extra = branched.load_offline_initializer(initializer)
    assert extra == {"phase": "offline_pretraining", "training_seed": 91}
    assert branched.total_updates == independent.total_updates == 3
    for name in ("actor", "critic", "target_critic", "value", "ratio"):
        _assert_nested_equal(
            getattr(branched, name).state_dict(),
            getattr(independent, name).state_dict(),
        )
    for name in (
        "actor_optimizer",
        "critic_optimizer",
        "value_optimizer",
        "alpha_optimizer",
    ):
        _assert_nested_equal(
            getattr(branched, name).state_dict(),
            getattr(independent, name).state_dict(),
        )
    _assert_nested_equal(
        branched.ratio.optimizer.state_dict(),
        independent.ratio.optimizer.state_dict(),
    )
    assert branched.rng.bit_generator.state == expected_numpy_rng
    torch.testing.assert_close(
        torch.get_rng_state(), expected_torch_rng, rtol=0.0, atol=0.0
    )


def test_v7_actor_reports_hybrid_entropy_components() -> None:
    trainer = TransitH2OTrainerV7(
        _V7ReplayStub(),
        config=TransitTrainerConfig(
            ensemble_size=2,
            hidden_dim=16,
            batch_size=3,
            target_subset_size=2,
            target_entropy=0.5,
            offline_objective="cql",
            online_backup="sac",
        ),
        seed=82,
    )
    actor_loss, alpha_loss, grad_norm = trainer._update_sac_actor(_states())
    assert torch.isfinite(actor_loss)
    assert torch.isfinite(alpha_loss)
    assert torch.isfinite(grad_norm)
    diagnostics = trainer._last_actor_diagnostics
    assert {
        "train/hybrid_entropy",
        "train/hybrid_gate_entropy",
        "train/hybrid_weighted_positive_entropy",
        "train/hold_probability",
        "train/positive_action_mean",
    } == set(diagnostics)
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_v7_hybrid_cql_uses_feature_subset_for_registered_wsrl_penalty(
    monkeypatch,
) -> None:
    trainer = TransitH2OTrainerV7(
        _V7ReplayStub(),
        config=TransitTrainerConfig(
            ensemble_size=10,
            hidden_dim=16,
            batch_size=3,
            target_subset_size=2,
            cql_penalty_subset_size=2,
            cql_penalty_subset_replacement=True,
            cql_n_actions=2,
            offline_objective="cql",
            online_backup="sac",
            use_ratio_correction=False,
        ),
        seed=83,
    )
    batch = {
        "observations": _states(),
        "actions": torch.tensor([[-1.0], [0.0], [0.5]]),
        "rewards": torch.zeros(3),
        "next_observations": _states().roll(1, dims=0),
        "discounts": torch.ones(3),
        "terminals": torch.zeros(3),
    }
    data_q = trainer.critic(batch["observations"], batch["actions"])
    subset_calls = []
    original = trainer.critic.forward_features_subset

    def counted_subset(features, actions, indices):
        subset_calls.append(indices.detach().clone())
        return original(features, actions, indices)

    monkeypatch.setattr(trainer.critic, "forward_features_subset", counted_subset)
    monkeypatch.setattr(
        trainer,
        "_sample_cql_penalty_subset",
        lambda: torch.tensor([7, 7]),
    )

    penalty = trainer._cql_penalty(batch, data_q)

    assert torch.isfinite(penalty)
    assert len(subset_calls) == 2
    assert all(torch.equal(call, torch.tensor([7, 7])) for call in subset_calls)
