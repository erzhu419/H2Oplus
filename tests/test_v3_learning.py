from __future__ import annotations

import copy
import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from SimpleSAC.train_transit_v3 import (
    make_warmup_policy,
    validate_v7_hybrid_target_entropy,
)
from SimpleSAC.transit_v3.domain_ratio import (
    FactoredDynamicsRatio,
    conditional_log_ratio,
)
from SimpleSAC.transit_v3.networks import TransitActor, TransitQEnsemble, TransitValue
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.transit_v3.trainer import TransitH2OTrainer, TransitTrainerConfig
from bus_h2o.mdp_contract import CANONICAL_STATE_DIM
from reproducibility.benchmark_training_kernel_v4 import trainer_config_for_method


def make_batch(batch_size=16):
    observations = torch.rand(batch_size, CANONICAL_STATE_DIM)
    observations[:, 0] = torch.arange(batch_size) % 12
    next_observations = observations.clone()
    next_observations[:, 1:] += 0.01 * torch.randn_like(next_observations[:, 1:])
    return {
        "observations": observations,
        "actions": torch.rand(batch_size, 1) * 2 - 1,
        "rewards": -torch.rand(batch_size),
        "next_observations": next_observations,
        "discounts": torch.full((batch_size,), 0.98),
        "terminals": torch.zeros(batch_size),
    }


class TinyReplay:
    def __init__(self):
        self.online_size = 100

    def sample_mixed(self, batch_size, online_fraction, rng):
        del rng
        online_size = int(round(batch_size * online_fraction))
        offline = make_batch(batch_size - online_size)
        online = make_batch(online_size) if online_size else None
        return offline, online

    def sample(self, batch_size, *, scope, rng, partition="all"):
        del scope, rng, partition
        return make_batch(batch_size)


class RatioReplay(TinyReplay):
    def sample(self, batch_size, *, scope, rng, partition="all"):
        del scope, partition
        rng.integers(0, 1_000_000, size=batch_size)
        return make_batch(batch_size)


class RatioReplayWithoutValidation(RatioReplay):
    def partition_size(self, scope, partition="all"):
        del scope
        return 0 if partition == "ratio_validation" else 100


def test_actor_and_independent_critic_shapes():
    batch = make_batch(7)
    actor = TransitActor((32, 32))
    critic = TransitQEnsemble(3, (32, 32))
    actions, log_prob = actor(batch["observations"])
    assert actions.shape == (7, 1)
    assert log_prob.shape == (7,)
    assert torch.all(actions.abs() <= 1.0)
    assert critic(batch["observations"], actions).shape == (3, 7)
    assert TransitValue((32, 32))(batch["observations"]).shape == (7,)


def test_critic_subset_matches_full_ensemble_selection():
    batch = make_batch(7)
    critic = TransitQEnsemble(4, (32, 32), layer_norm=True)
    actions, _ = TransitActor((32, 32))(batch["observations"])
    indices = torch.tensor([3, 1], dtype=torch.long)
    torch.testing.assert_close(
        critic.forward_subset(batch["observations"], actions, indices),
        critic(batch["observations"], actions).index_select(0, indices),
    )


def test_target_subset_sampling_matches_rlpd_and_wsrl_sources(monkeypatch):
    trainer = object.__new__(TransitH2OTrainer)
    trainer.device = torch.device("cpu")

    trainer.config = TransitTrainerConfig(
        ensemble_size=10,
        target_subset_size=2,
        target_subset_replacement=False,
    )
    for _ in range(100):
        chosen = trainer._sample_target_subset()
        assert chosen.unique().numel() == 2

    calls = []

    def duplicate_randint(high, size, *, device):
        calls.append((high, size, device))
        return torch.tensor([3, 3], device=device)

    monkeypatch.setattr(torch, "randint", duplicate_randint)
    trainer.config = TransitTrainerConfig(
        ensemble_size=10,
        target_subset_size=2,
        target_subset_replacement=True,
        cql_penalty_subset_size=2,
        cql_penalty_subset_replacement=True,
    )
    torch.testing.assert_close(
        trainer._sample_target_subset(), torch.tensor([3, 3])
    )
    torch.testing.assert_close(
        trainer._sample_cql_penalty_subset(), torch.tensor([3, 3])
    )
    assert calls == [
        (10, (2,), torch.device("cpu")),
        (10, (2,), torch.device("cpu")),
    ]


class _ZeroActor(nn.Module):
    def forward(self, states, deterministic=False):
        del deterministic
        return torch.zeros((len(states), 1)), torch.zeros(len(states))


class _SubsetOnlyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, states, actions):
        del states, actions
        raise AssertionError("target backup evaluated the full critic ensemble")

    def forward_subset(self, states, actions, indices):
        del actions
        self.calls.append(indices.detach().clone())
        return indices[:, None].to(states.dtype).expand(-1, len(states))


class _FixedCritic(nn.Module):
    def forward(self, states, actions):
        del actions
        return torch.stack(
            (torch.full((len(states),), 2.0), torch.full((len(states),), 3.0))
        )


class _FixedValue(nn.Module):
    def forward(self, states):
        return torch.full((len(states),), 4.0)


class _CQLSubsetCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, states, actions):
        del states, actions
        raise AssertionError("CQL penalty evaluated the full critic ensemble")

    def forward_subset(self, states, actions, indices):
        del actions
        self.calls.append(indices.detach().clone())
        return indices[:, None].to(states.dtype).expand(-1, len(states))


def test_sac_and_cql_backups_evaluate_only_sampled_target_heads():
    config = TransitTrainerConfig(
        ensemble_size=4,
        hidden_dim=16,
        batch_size=3,
        target_subset_size=2,
        cql_n_actions=3,
        backup_entropy=False,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=101)
    trainer.actor = _ZeroActor()
    subset_critic = _SubsetOnlyCritic()
    trainer.target_critic = subset_critic
    batch = make_batch(3)

    assert trainer._sac_bootstrap(batch).shape == (3,)
    assert trainer._cql_max_bootstrap(batch).shape == (3,)
    assert len(subset_critic.calls) == 2
    assert all(call.shape == (config.target_subset_size,) for call in subset_critic.calls)


def test_wsrl_cql_penalty_evaluates_only_its_sampled_heads(monkeypatch):
    config = TransitTrainerConfig(
        ensemble_size=10,
        hidden_dim=16,
        batch_size=3,
        target_subset_size=2,
        cql_penalty_subset_size=2,
        cql_penalty_subset_replacement=True,
        cql_n_actions=2,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=102)
    trainer.actor = _ZeroActor()
    critic = _CQLSubsetCritic()
    trainer.critic = critic
    monkeypatch.setattr(
        trainer,
        "_sample_cql_penalty_subset",
        lambda: torch.tensor([4, 4]),
    )
    data_q = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    penalty = trainer._cql_penalty(make_batch(3), data_q)

    assert torch.isfinite(penalty)
    assert len(critic.calls) == 1
    torch.testing.assert_close(critic.calls[0], torch.tensor([4, 4]))


def test_h2oplus_target_is_the_published_mixed_value_backup():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=4,
        target_subset_size=2,
        offline_objective="iql",
        online_backup="h2oplus",
        mixed_backup_lambda=0.25,
        backup_entropy=False,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=1)
    trainer.actor = _ZeroActor()
    trainer.target_critic = _FixedCritic()
    trainer.value = _FixedValue()
    batch = make_batch(4)
    batch["rewards"] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    batch["discounts"] = torch.tensor([0.5, 0.7, 0.9, 0.6])
    batch["terminals"] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    expected_bootstrap = 0.25 * 4.0 + 0.75 * 2.0
    expected = batch["rewards"] + batch["discounts"] * (
        1.0 - batch["terminals"]
    ) * expected_bootstrap
    torch.testing.assert_close(
        trainer._target(batch, backup="h2oplus"), expected
    )


def test_reward_scale_changes_only_the_immediate_reward_term():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=4,
        reward_scale=10.0,
        target_subset_size=2,
        offline_objective="iql",
        online_backup="h2oplus",
        mixed_backup_lambda=0.25,
        backup_entropy=False,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=2)
    trainer.actor = _ZeroActor()
    trainer.target_critic = _FixedCritic()
    trainer.value = _FixedValue()
    batch = make_batch(4)
    batch["rewards"] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    batch["discounts"] = torch.tensor([0.5, 0.7, 0.9, 0.6])
    batch["terminals"] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    expected_bootstrap = 0.25 * 4.0 + 0.75 * 2.0
    expected = 10.0 * batch["rewards"] + batch["discounts"] * (
        1.0 - batch["terminals"]
    ) * expected_bootstrap
    torch.testing.assert_close(
        trainer._target(batch, backup="h2oplus"), expected
    )


def test_configurable_critic_loss_and_terminal_diagnostics():
    predictions = torch.tensor([[0.0, 2.0], [4.0, -4.0]])
    targets = torch.zeros(2)
    batch = make_batch(2)
    batch["terminals"] = torch.tensor([0.0, 1.0])

    mse = TransitH2OTrainer(
        TinyReplay(),
        config=TransitTrainerConfig(
            hidden_dim=16,
            batch_size=2,
            critic_loss="mse",
            bootstrap_probability=1.0,
        ),
        seed=31,
    )
    huber = TransitH2OTrainer(
        TinyReplay(),
        config=TransitTrainerConfig(
            hidden_dim=16,
            batch_size=2,
            critic_loss="huber",
            huber_delta=1.0,
            bootstrap_probability=1.0,
        ),
        seed=31,
    )

    torch.testing.assert_close(
        mse._ensemble_td_loss(predictions, targets), torch.tensor(9.0)
    )
    torch.testing.assert_close(
        huber._ensemble_td_loss(predictions, targets), torch.tensor(2.125)
    )
    diagnostics = huber._td_diagnostics(
        predictions, targets, batch, prefix="offline"
    )
    assert diagnostics["train/offline_terminal_rows"] == 1
    assert diagnostics["train/offline_terminal_abs_td"] == 3.0
    assert diagnostics["train/offline_nonterminal_abs_td"] == 2.0
    assert 0.0 < diagnostics["train/offline_terminal_loss_fraction"] < 1.0


def test_reference_method_profiles_encode_official_update_protocols():
    rlpd = METHOD_PROFILES["rlpd"]
    assert (rlpd.ensemble_size, rlpd.target_subset_size) == (10, 2)
    assert not rlpd.target_subset_replacement
    assert (rlpd.utd_ratio, rlpd.actor_update_interval) == (20, 20)
    assert rlpd.actor_q_reduction == "mean-all"
    assert rlpd.activation == "relu"
    assert not rlpd.actor_layer_norm and rlpd.critic_layer_norm
    assert rlpd.online_fraction == 0.5
    assert rlpd.default_offline_updates == 0
    assert rlpd.gradient_clip_norm is None
    assert rlpd.online_update_order == "critic-before-actor"
    assert rlpd.warmup_action_mode == "uniform-box"

    sac_mixed = METHOD_PROFILES["sac-mixed"]
    h2oplus = METHOD_PROFILES["h2oplus"]
    assert sac_mixed.reference == "https://proceedings.mlr.press/v80/haarnoja18b.html"
    assert sac_mixed.offline_objective == "none"
    assert sac_mixed.online_backup == "sac"
    assert sac_mixed.online_fraction == h2oplus.online_fraction == 0.5
    assert sac_mixed.ensemble_size == h2oplus.ensemble_size == 2
    assert sac_mixed.batch_size == h2oplus.batch_size == 256
    assert sac_mixed.actor_q_reduction == h2oplus.actor_q_reduction == "min-all"
    assert sac_mixed.activation == h2oplus.activation == "relu"
    assert not sac_mixed.actor_layer_norm and not sac_mixed.critic_layer_norm
    assert h2oplus.online_update_order == "actor-before-critic"
    assert METHOD_PROFILES["h2oplus-source"].online_update_order == (
        "actor-before-critic"
    )

    wsrl = METHOD_PROFILES["wsrl"]
    assert (wsrl.ensemble_size, wsrl.target_subset_size) == (10, 2)
    assert wsrl.target_subset_replacement
    assert wsrl.cql_penalty_subset_size == 2
    assert wsrl.cql_penalty_subset_replacement
    assert wsrl.batch_size == 256
    assert (wsrl.utd_ratio, wsrl.actor_update_interval) == (4, 4)
    assert wsrl.actor_batch_mode == "aggregate"
    assert wsrl.actor_layer_norm and wsrl.critic_layer_norm
    assert wsrl.online_update_mode == "online-only"
    assert wsrl.default_warmup_events == 5_000
    assert wsrl.target_entropy == 0.0
    assert wsrl.cql_max_target_backup
    assert wsrl.offline_initializer == "cql"
    assert wsrl.gradient_clip_norm is None
    assert wsrl.online_update_order == "critic-before-actor"
    assert wsrl.offline_update_order == "actor-before-critic"
    assert wsrl.temperature_state_source == "next-observations"
    assert wsrl.offline_temperature_action_samples == 10
    assert wsrl.warmup_action_mode == "policy"
    wsrl_config = trainer_config_for_method("wsrl")
    assert wsrl_config.target_subset_replacement
    assert wsrl_config.cql_penalty_subset_size == 2
    assert wsrl_config.cql_penalty_subset_replacement
    assert wsrl_config.offline_update_order == "actor-before-critic"
    assert wsrl_config.temperature_state_source == "next-observations"
    assert wsrl_config.offline_temperature_action_samples == 10
    assert not trainer_config_for_method("rlpd").target_subset_replacement

    cql = METHOD_PROFILES["cql"]
    assert cql.cql_max_target_backup

    bc = METHOD_PROFILES["bc"]
    assert bc.reference == "https://proceedings.mlr.press/v15/ross11a.html"

    assert h2oplus.offline_objective == "iql"
    assert h2oplus.online_backup == "h2oplus"
    assert h2oplus.mixed_backup_lambda == 0.1
    assert h2oplus.use_ratio_correction
    assert h2oplus.reference_implementation == "https://github.com/t6-thu/H2Oplus"
    assert h2oplus.reference_commit == "02171768283d307244700c69003423704db83b05"

    source_h2oplus = METHOD_PROFILES["h2oplus-source"]
    assert source_h2oplus.offline_objective == "none"
    assert source_h2oplus.default_offline_updates == 0
    assert source_h2oplus.default_warmup_events == 1_000
    assert source_h2oplus.default_ratio_pretrain_updates == 0
    assert source_h2oplus.ratio_update_interval == 1
    assert source_h2oplus.iql_expectile == 0.5
    assert source_h2oplus.mixed_backup_lambda == 0.1

    lambda_zero = METHOD_PROFILES["h2oplus-lambda0"]
    assert lambda_zero.offline_objective == "iql"
    assert lambda_zero.mixed_backup_lambda == 0.0
    assert lambda_zero.use_ratio_correction

    lambda_half = METHOD_PROFILES["h2oplus-lambda0p5"]
    assert lambda_half.mixed_backup_lambda == 0.5
    assert lambda_half.offline_objective == "iql"

    lambda_point_two = METHOD_PROFILES["h2oplus-lambda0p2"]
    assert lambda_point_two.mixed_backup_lambda == 0.2
    assert lambda_point_two.offline_objective == "iql"

    lambda_one = METHOD_PROFILES["h2oplus-lambda1"]
    assert lambda_one.mixed_backup_lambda == 1.0
    assert lambda_one.offline_objective == "iql"

    no_warmstart = METHOD_PROFILES["h2oplus-no-warmstart"]
    assert no_warmstart.offline_objective == "none"
    assert no_warmstart.default_offline_updates == 0
    assert no_warmstart.online_backup == "h2oplus"
    assert no_warmstart.use_ratio_correction


def test_rlpd_warmup_uses_seeded_uniform_box_actions():
    observations = np.zeros((512, 7), dtype=np.float32)

    def forbidden_policy(*_args, **_kwargs):
        raise AssertionError("RLPD warmup must not sample its initial actor")

    first = make_warmup_policy(
        METHOD_PROFILES["rlpd"], forbidden_policy, seed=5101
    )(observations, False)
    second = make_warmup_policy(
        METHOD_PROFILES["rlpd"], forbidden_policy, seed=5101
    )(observations, False)
    assert first.shape == (512, 1)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all(first >= -1.0) and np.all(first <= 1.0)
    assert abs(float(first.mean())) < 0.1

    policy = object()
    assert make_warmup_policy(METHOD_PROFILES["wsrl"], policy, seed=1) is policy


def test_v7_hybrid_entropy_has_no_artificial_zero_lower_bound():
    assert validate_v7_hybrid_target_entropy(-1.0) == -1.0
    assert validate_v7_hybrid_target_entropy(-20.0) == -20.0
    with pytest.raises(ValueError, match="less than log\\(3\\)"):
        validate_v7_hybrid_target_entropy(math.log(3.0))
    with pytest.raises(ValueError, match="finite"):
        validate_v7_hybrid_target_entropy(float("nan"))


def test_conditional_ratio_subtracts_occupancy_ratio_without_self_normalization():
    joint = torch.tensor([2.0, -1.0])
    marginal = torch.tensor([0.5, -0.5])
    torch.testing.assert_close(
        conditional_log_ratio(joint, marginal), torch.tensor([1.5, -0.5])
    )
    estimator = FactoredDynamicsRatio((16, 16), min_weight=1e-5, max_weight=1.0)
    log_ratios = torch.tensor([math.log(2.0), math.log(0.5), 0.0, math.log(100.0)])
    estimator.logits = lambda batch: (  # type: ignore[method-assign]
        log_ratios,
        torch.zeros_like(log_ratios),
    )
    sim = make_batch(4)
    weights, diagnostics = estimator.weights(sim)
    torch.testing.assert_close(weights, torch.tensor([1.0, 0.5, 1.0, 1.0]))
    torch.testing.assert_close(diagnostics["ratio/weight_mean"], torch.tensor(0.875))
    assert diagnostics["ratio/high_clip_fraction"] == pytest.approx(0.5)
    assert 0.0 < diagnostics["ratio/ess_fraction"] <= 1.0


def test_trainer_runs_offline_and_ratio_corrected_updates():
    config = TransitTrainerConfig(
        ensemble_size=3,
        hidden_dim=32,
        batch_size=12,
        cql_n_actions=2,
        target_subset_size=2,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=3)
    offline_metrics = trainer.update(online=False)
    online_metrics = trainer.update(online=True)
    assert np.isfinite(list(offline_metrics.values())).all()
    assert np.isfinite(list(online_metrics.values())).all()
    assert offline_metrics["train/update"] == 1
    assert online_metrics["train/update"] == 2
    assert "ratio/ess_fraction" in online_metrics


def test_mixed_update_computes_one_combined_bootstrap_batch():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        target_subset_size=2,
        offline_objective="none",
        online_backup="sac",
        online_fraction=0.5,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=4)
    target_batch_sizes = []
    original_target = trainer._target

    def counted_target(batch, *, backup):
        target_batch_sizes.append(batch["observations"].shape[0])
        return original_target(batch, backup=backup)

    trainer._target = counted_target
    metrics = trainer.update(online=True)

    assert target_batch_sizes == [config.batch_size]
    assert metrics["train/offline_td_loss"] >= 0.0
    assert metrics["train/sim_td_loss"] >= 0.0


class _OrderedActor(nn.Module):
    def forward(self, states, deterministic=False):
        del deterministic
        actions = torch.arange(len(states), dtype=states.dtype)[:, None]
        return actions, torch.zeros(len(states), dtype=states.dtype)


class _ActionCritic(nn.Module):
    def forward(self, states, actions):
        del states
        values = actions[:, 0]
        return torch.stack((values, values + 1.0))


def test_cql_max_target_backup_selects_best_policy_proposal():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=2,
        target_subset_size=2,
        cql_n_actions=3,
        cql_max_target_backup=True,
        backup_entropy=False,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=41)
    trainer.actor = _OrderedActor()
    trainer.target_critic = _ActionCritic()
    batch = make_batch(2)

    torch.testing.assert_close(
        trainer._target(batch, backup="cql"),
        batch["rewards"] + batch["discounts"] * torch.tensor([2.0, 5.0]),
    )


def test_online_only_and_behavior_cloning_updates():
    online_config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        cql_n_actions=2,
        target_subset_size=2,
        use_ratio_correction=False,
    )
    online_trainer = TransitH2OTrainer(TinyReplay(), config=online_config, seed=5)
    online_metrics = online_trainer.update(online=True, online_only=True)
    assert online_metrics["train/offline_td_loss"] == 0.0
    assert np.isfinite(list(online_metrics.values())).all()

    cloning_trainer = TransitH2OTrainer(TinyReplay(), config=online_config, seed=6)
    cloning_metrics = cloning_trainer.behavior_clone_step()
    assert cloning_metrics["train/update"] == 1
    assert np.isfinite(list(cloning_metrics.values())).all()


def test_iql_value_update_and_delayed_actor_resume_are_exact(tmp_path):
    iql_config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        target_subset_size=2,
        offline_objective="iql",
        online_backup="h2oplus",
        mixed_backup_lambda=0.1,
        offline_cql_alpha=0.0,
        cql_n_actions=2,
        use_ratio_correction=False,
    )
    iql_trainer = TransitH2OTrainer(TinyReplay(), config=iql_config, seed=12)
    offline_metrics = iql_trainer.update(online=False)
    assert offline_metrics["train/value_loss"] > 0.0
    assert offline_metrics["train/value_grad_norm"] > 0.0
    assert np.isfinite(list(offline_metrics.values())).all()

    delayed_config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        target_subset_size=2,
        offline_objective="none",
        online_backup="sac",
        actor_q_reduction="min-all",
        online_actor_update_interval=4,
        online_actor_batch_mode="aggregate",
        offline_cql_alpha=0.0,
        cql_n_actions=2,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=delayed_config, seed=13)
    for _ in range(3):
        metrics = trainer.update(online=True, online_only=True)
        assert metrics["train/actor_updated"] == 0.0
    assert len(trainer._online_actor_state_batches) == 3
    assert len(trainer._online_temperature_state_batches) == 3

    checkpoint = tmp_path / "delayed.pt"
    trainer.save(checkpoint, include_online_replay=False)
    expected_metrics = trainer.update(online=True, online_only=True)
    assert expected_metrics["train/actor_updated"] == 1.0

    restored = TransitH2OTrainer(TinyReplay(), config=delayed_config, seed=99)
    restored.load(checkpoint)
    assert restored.online_critic_updates == 3
    assert len(restored._online_actor_state_batches) == 3
    assert len(restored._online_temperature_state_batches) == 3
    actual_metrics = restored.update(online=True, online_only=True)
    assert actual_metrics == expected_metrics
    for expected, actual in zip(trainer.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(trainer.critic.parameters(), restored.critic.parameters()):
        torch.testing.assert_close(expected, actual)


def test_wsrl_temperature_batches_use_next_states_and_source_sample_counts():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=3,
        target_subset_size=2,
        offline_objective="cql",
        online_actor_update_interval=4,
        online_actor_batch_mode="aggregate",
        temperature_state_source="next-observations",
        offline_temperature_action_samples=10,
        cql_n_actions=2,
        use_ratio_correction=False,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=131)
    captured = []

    def capture_actor_update(
        actor_states,
        *,
        temperature_states,
        temperature_action_samples,
        collect_metrics,
    ):
        del collect_metrics
        captured.append(
            (
                actor_states.detach().clone(),
                temperature_states.detach().clone(),
                temperature_action_samples,
            )
        )
        zero = torch.zeros(())
        return zero, zero, zero

    trainer._update_sac_actor = capture_actor_update
    online_batches = [make_batch(3) for _ in range(4)]
    for batch in online_batches:
        trainer._update_actor_for_batch(
            online=True,
            offline_batch=None,
            online_batch=batch,
            iql_actor_weights=None,
            collect_metrics=False,
        )
    assert len(captured) == 1
    torch.testing.assert_close(
        captured[0][0],
        torch.cat([batch["observations"] for batch in online_batches], dim=0),
    )
    torch.testing.assert_close(
        captured[0][1],
        torch.cat([batch["next_observations"] for batch in online_batches], dim=0),
    )
    assert captured[0][2] == 1

    offline_batch = make_batch(3)
    trainer._update_actor_for_batch(
        online=False,
        offline_batch=offline_batch,
        online_batch=None,
        iql_actor_weights=None,
        collect_metrics=False,
    )
    assert len(captured) == 2
    torch.testing.assert_close(captured[1][0], offline_batch["observations"])
    torch.testing.assert_close(captured[1][1], offline_batch["next_observations"])
    assert captured[1][2] == 10


def test_online_and_offline_update_orders_are_explicit_and_applied():
    def observed_order(order: str, *, online: bool) -> list[str]:
        config = TransitTrainerConfig(
            ensemble_size=2,
            hidden_dim=16,
            batch_size=8,
            target_subset_size=2,
            offline_objective="none" if online else "cql",
            online_backup="sac",
            offline_cql_alpha=0.0,
            cql_n_actions=2,
            use_ratio_correction=False,
            online_update_order=order if online else "critic-before-actor",
            offline_update_order=order if not online else "critic-before-actor",
        )
        trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=14)
        events: list[str] = []
        actor_update = trainer._update_sac_actor
        critic_step = trainer.critic_optimizer.step

        def recorded_actor_update(*args, **kwargs):
            events.append("actor")
            return actor_update(*args, **kwargs)

        def recorded_critic_step(*args, **kwargs):
            events.append("critic")
            return critic_step(*args, **kwargs)

        trainer._update_sac_actor = recorded_actor_update
        trainer.critic_optimizer.step = recorded_critic_step
        trainer.update(online=online, online_only=online)
        return events

    assert observed_order("actor-before-critic", online=True) == [
        "actor",
        "critic",
    ]
    assert observed_order("critic-before-actor", online=True) == [
        "critic",
        "actor",
    ]
    assert observed_order("actor-before-critic", online=False) == [
        "actor",
        "critic",
    ]
    assert observed_order("critic-before-actor", online=False) == [
        "critic",
        "actor",
    ]


def test_ratio_pretraining_and_checkpoint_round_trip(tmp_path):
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        cql_n_actions=2,
        target_subset_size=2,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=7)
    metrics = trainer.pretrain_ratio(2)
    assert np.isfinite(list(metrics.values())).all()
    assert "ratio/validation_ess_fraction" in metrics
    trainer.update(online=False)
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint, extra={"seed": 7}, include_online_replay=False)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["python_rng_state"] is not None
    assert payload["numpy_global_rng_state"] is not None
    assert payload["torch_cuda_rng_state"] is None
    assert "torch_cuda_rng_state_all" not in payload
    restored = TransitH2OTrainer(TinyReplay(), config=config, seed=99)
    extra = restored.load(checkpoint)
    assert extra == {"seed": 7}
    assert restored.total_updates == 1
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    for expected, actual in zip(trainer.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(expected, actual)


def test_compiled_networks_preserve_uncompiled_checkpoint_keys(tmp_path):
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        cql_n_actions=2,
        target_subset_size=2,
    )
    trainer = TransitH2OTrainer(TinyReplay(), config=config, seed=71)
    compiled = trainer.compile_networks()
    assert compiled == [
        "actor_mlp",
        "critic_ensemble",
        "ratio_joint_mlp",
        "ratio_state_action_mlp",
        "target_critic_ensemble",
    ]
    assert trainer.compiler_config["mode"] == "default"
    assert trainer.compiler_config["cuda_graphs_enabled"] is False
    assert all("_orig_mod" not in key for key in trainer.actor.state_dict())

    checkpoint = tmp_path / "compiled.pt"
    trainer.save(checkpoint, include_online_replay=False)
    restored = TransitH2OTrainer(TinyReplay(), config=config, seed=72)
    restored.load(checkpoint)
    for expected, actual in zip(trainer.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(expected, actual)


def test_ratio_holdout_monitoring_does_not_perturb_training_rng():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        cql_n_actions=2,
        target_subset_size=2,
    )
    trainer = TransitH2OTrainer(RatioReplay(), config=config, seed=17)
    training_state = copy.deepcopy(trainer.rng.bit_generator.state)
    diagnostic_state = copy.deepcopy(trainer.diagnostic_rng.bit_generator.state)

    metrics = trainer.evaluate_ratio()

    assert trainer.rng.bit_generator.state == training_state
    assert trainer.diagnostic_rng.bit_generator.state != diagnostic_state
    assert "ratio/validation_ess_fraction" in metrics


def test_ratio_pretraining_defers_validation_until_an_online_holdout_exists():
    config = TransitTrainerConfig(
        ensemble_size=2,
        hidden_dim=16,
        batch_size=8,
        cql_n_actions=2,
        target_subset_size=2,
    )
    trainer = TransitH2OTrainer(
        RatioReplayWithoutValidation(), config=config, seed=23
    )
    metrics = trainer.pretrain_ratio(2)
    assert metrics["ratio/validation_available"] == 0.0
    assert "ratio/loss" in metrics
    assert trainer.evaluate_ratio() == {"ratio/validation_available": 0.0}
