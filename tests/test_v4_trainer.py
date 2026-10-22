import math

import torch

from bus_h2o.mdp_contract_v4 import CANONICAL_STATE_DIM
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
from SimpleSAC.transit_v4.trainer import TransitH2OTrainerV4


def _batch(size=16):
    observations = torch.rand(size, CANONICAL_STATE_DIM)
    next_observations = torch.rand(size, CANONICAL_STATE_DIM)
    observations[:, 0] = torch.arange(size) % 12
    next_observations[:, 0] = torch.arange(size) % 12
    actions = torch.linspace(-1.0, 0.9, size)[:, None]
    actions[::3] = -1.0
    return {
        "observations": observations,
        "actions": actions,
        "rewards": -torch.rand(size),
        "next_observations": next_observations,
        "discounts": torch.full((size,), 0.98),
        "terminals": torch.zeros(size),
    }


def _trainer(**overrides):
    values = dict(
        ensemble_size=2,
        target_subset_size=2,
        hidden_dim=32,
        batch_size=16,
        offline_objective="iql",
        online_backup="h2oplus",
        offline_cql_alpha=0.0,
        initial_temperature=0.1,
    )
    values.update(overrides)
    return TransitH2OTrainerV4(object(), config=TransitTrainerConfig(**values), seed=7)


def test_v4_soft_backup_enumerates_zero_and_positive_branches():
    trainer = _trainer()
    batch = _batch()
    target = trainer._sac_bootstrap(batch)
    assert target.shape == (16,)
    assert torch.isfinite(target).all()


def test_v4_sac_actor_update_backpropagates_through_hurdle_probability():
    trainer = _trainer()
    states = _batch()["observations"]
    before = trainer.actor.network.network[-1].bias.detach().clone()
    actor_loss, alpha_loss, grad_norm = trainer._update_sac_actor(states)
    after = trainer.actor.network.network[-1].bias.detach()
    assert torch.isfinite(actor_loss)
    assert torch.isfinite(alpha_loss)
    assert torch.isfinite(grad_norm)
    assert not torch.equal(before, after)


def test_v4_iql_likelihood_and_hybrid_cql_penalty_are_finite():
    trainer = _trainer(offline_objective="cql", online_backup="sac", offline_cql_alpha=0.2)
    batch = _batch()
    log_prob = trainer.actor.action_log_prob(batch["observations"], batch["actions"])
    data_q = trainer.critic(batch["observations"], batch["actions"])
    penalty = trainer._cql_penalty(batch, data_q)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(penalty)


def test_hybrid_cql_partition_preserves_zero_atom_measure():
    zero_q = torch.zeros(2, 3)
    proposal = torch.full((2, 3, 7), math.log(2.0))
    partition = TransitH2OTrainerV4._hybrid_cql_log_partition(
        zero_q, (proposal, proposal, proposal)
    )
    torch.testing.assert_close(
        partition, torch.full_like(partition, math.log(3.0))
    )
