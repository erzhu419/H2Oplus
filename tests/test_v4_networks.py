import numpy as np
import torch

from bus_h2o.mdp_contract_v4 import CANONICAL_STATE_DIM
from SimpleSAC.transit_v3.networks import EnsembleLayerNorm
from SimpleSAC.transit_v4.networks import (
    HurdleTransitActor,
    TransitQEnsembleV4,
    TransitValueV4,
    canonical_features_v4,
)


def _states(n=64):
    states = torch.rand(n, CANONICAL_STATE_DIM)
    states[:, 0] = torch.arange(n) % 12
    return states


def test_v4_features_have_explicit_line_one_hot():
    states = _states(3)
    states[:, 0] = torch.tensor([0.0, 4.0, 11.0])
    features = canonical_features_v4(states)
    assert features.shape == (3, 12 + CANONICAL_STATE_DIM - 1)
    assert torch.equal(features[:, :12].sum(dim=1), torch.ones(3))


def test_hurdle_actor_assigns_finite_mass_to_exact_zero():
    torch.manual_seed(3)
    actor = HurdleTransitActor((32, 32))
    states = _states(2048)
    actions, log_prob = actor(states)
    zero = actions[:, 0] == -1.0
    assert zero.any()
    assert (~zero).any()
    assert torch.isfinite(log_prob).all()
    assert torch.all(actions[~zero] > -1.0)
    assert torch.all(actions <= 1.0)


def test_hurdle_likelihood_uses_mass_for_zero_and_density_for_positive():
    actor = HurdleTransitActor((32, 32))
    states = _states(5)
    components = actor.soft_action_components(states, deterministic_positive=True)
    zero_actions = torch.full((5, 1), -1.0)
    positive_actions = components["positive_action"].detach()
    torch.testing.assert_close(
        actor.action_log_prob(states, zero_actions), components["zero_log_prob"]
    )
    torch.testing.assert_close(
        actor.action_log_prob(states, positive_actions),
        components["positive_log_prob"],
        atol=2e-5,
        rtol=2e-5,
    )


def test_only_exact_minus_one_is_the_zero_action_atom():
    actor = HurdleTransitActor((32, 32))
    states = _states(2)
    exact_zero = torch.tensor([[-1.0]])
    smallest_positive = torch.nextafter(
        torch.tensor([[-1.0]]), torch.tensor([[0.0]])
    )
    actions = torch.cat((exact_zero, smallest_positive), dim=0)
    gate_logit, mean, log_std = actor.distribution_parameters(states)
    observed = actor.action_log_prob(states, actions)

    expected_zero = torch.nn.functional.logsigmoid(-gate_logit[0])
    fraction_epsilon = torch.finfo(smallest_positive.dtype).eps * 0.5
    positive_fraction = ((smallest_positive[0, 0] + 1.0) * 0.5).clamp(
        fraction_epsilon, 1.0 - fraction_epsilon
    )
    pre_sigmoid = torch.logit(positive_fraction)
    distribution = torch.distributions.Normal(mean[1], log_std[1].exp())
    expected_positive = (
        torch.nn.functional.logsigmoid(gate_logit[1])
        + distribution.log_prob(pre_sigmoid)
        - actor._positive_log_det(pre_sigmoid)
    )

    torch.testing.assert_close(observed[0], expected_zero)
    torch.testing.assert_close(observed[1], expected_positive)
    assert observed[1] != torch.nn.functional.logsigmoid(-gate_logit[1])


def test_positive_branch_stays_inside_the_open_action_interval_at_extremes():
    pre_sigmoid = torch.tensor([-1000.0, 1000.0], dtype=torch.float32)
    actions = HurdleTransitActor._positive_transform(pre_sigmoid)
    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)


def test_hurdle_soft_components_are_normalized_and_gate_is_differentiable():
    actor = HurdleTransitActor((32, 32))
    components = actor.soft_action_components(_states(11))
    torch.testing.assert_close(
        components["zero_probability"] + components["hold_probability"],
        torch.ones(11),
    )
    objective = (
        components["zero_probability"] * components["zero_log_prob"]
        + components["hold_probability"] * components["positive_log_prob"]
    ).mean()
    objective.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


def test_v4_critic_and_value_shapes():
    states = _states(7)
    actions = torch.from_numpy(np.linspace(-1.0, 1.0, 7, dtype=np.float32))[:, None]
    critic = TransitQEnsembleV4(3, (32, 32))
    value = TransitValueV4((32, 32))
    assert critic(states, actions).shape == (3, 7)
    assert value(states).shape == (7,)


def test_ensemble_layer_norm_matches_direct_per_member_normalization():
    torch.manual_seed(7)
    layer = EnsembleLayerNorm(3, 5)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.bias.copy_(torch.randn_like(layer.bias))
    values = torch.randn(3, 11, 5, requires_grad=True)
    observed = layer(values)
    mean = values.mean(dim=-1, keepdim=True)
    variance = values.var(dim=-1, keepdim=True, unbiased=False)
    expected = (values - mean) * torch.rsqrt(variance + layer.epsilon)
    expected = expected * layer.weight[:, None, :] + layer.bias[:, None, :]
    torch.testing.assert_close(observed, expected)
    observed.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
