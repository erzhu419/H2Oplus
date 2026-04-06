import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
# import ipdb

# import math
# from numbers import Real
# from numbers import Number

# from torch.distributions import constraints
# from torch.distributions.exp_family import ExponentialFamily
# from torch.distributions.utils import _standard_normal, broadcast_all

def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

# Inverse tanh torch function
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0, eps=1e-6, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh
        self.eps = eps

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        # ipdb.set_trace()
        return torch.sum(action_distribution.log_prob(torch.clamp(sample, -1 + self.eps, 1 - self.eps)), dim=-1)
    
    # def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
    #     if is_pretanh_action:
    #         pretanh_action = action
    #         action = torch.tanh(pretanh_action)
    #     else:
    #         pretanh_action = atanh(torch.clamp(action, -1 + self.eps, 1 - self.eps))

    #     pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
    #     log_prob = pretanh_log_prob - torch.log(1 - action ** 2 + self.eps)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     return log_prob, pretanh_log_prob
    
    # def log_prob(self, mean, log_std, sample):
    #     log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    #     std = torch.exp(log_std)
    #     if self.no_tanh:
    #         action_distribution = Normal(mean, std)
    #         pretanh_action = atanh(torch.clamp(sample, -1 + self.eps, 1 - self.eps))
    #     else:
    #         action_distribution = TransformedDistribution(
    #             Normal(mean, std), TanhTransform(cache_size=1)
    #         )
    #         pretanh_action = sample
    #         sample = torch.tanh(pretanh_action)

    #     pretanh_log_prob = action_distribution.log_prob(pretanh_action)
    #     log_prob = pretanh_log_prob - torch.log(1 - sample ** 2 + self.eps)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     # ipdb.set_trace()
    #     return log_prob

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        assert torch.isnan(observations).sum() == 0, print(observations)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        assert torch.isnan(mean).sum() == 0, print(mean)
        assert torch.isnan(log_std).sum() == 0, print(log_std)
        return self.tanh_gaussian(mean, log_std, deterministic)


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, orthogonal_init
        )

    @multiple_action_q_function
    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant


# =============================================================================
# Legacy-aligned architecture (port from sac_v2_bus_SUMO_linear_penalty.py)
# =============================================================================

class EmbeddingLayer(nn.Module):
    """Categorical feature embedding layer, matching the legacy SAC architecture.

    For the SUMO bus env, cat_cols = ['line_id', 'bus_id', 'station_id',
    'time_period', 'direction'] — the first 5 columns of the 15-dim obs vector.
    """

    def __init__(self, cat_code_dict, cat_cols, embedding_dims=None,
                 layer_norm=False, dropout=0.0):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)

        self.embedding_dims = {}
        self.cardinalities = {}
        modules = {}
        for col in self.cat_cols:
            codes = list(cat_code_dict[col].values())
            if len(codes) == 0:
                raise ValueError(f"Categorical column '{col}' has no encoding values.")
            cardinality = max(codes) + 1
            self.cardinalities[col] = cardinality
            dim = (embedding_dims[col]
                   if embedding_dims and col in embedding_dims
                   else self._suggest_dim(cardinality))
            self.embedding_dims[col] = dim
            modules[col] = nn.Embedding(cardinality, dim)

        self.embeddings = nn.ModuleDict(modules)
        self.output_dim = sum(self.embedding_dims.values())
        self.layer_norm = (
            nn.LayerNorm(self.output_dim)
            if layer_norm and self.output_dim > 0 else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def _suggest_dim(cardinality: int) -> int:
        if cardinality <= 1:
            return 1
        return min(32, max(2, int(round(cardinality ** 0.5)) + 1))

    @classmethod
    def compute_output_dim(cls, cat_code_dict, cat_cols, embedding_dims=None):
        total = 0
        for col in cat_cols:
            codes = list(cat_code_dict[col].values())
            if len(codes) == 0:
                continue
            cardinality = max(codes) + 1
            if embedding_dims and col in embedding_dims:
                total += embedding_dims[col]
            else:
                total += cls._suggest_dim(cardinality)
        return total

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1:
            cat_tensor = cat_tensor.unsqueeze(0)
        parts = []
        for idx, col in enumerate(self.cat_cols):
            indices = cat_tensor[:, idx].long()
            max_index = self.cardinalities[col] - 1
            indices = torch.clamp(indices, 0, max_index)
            parts.append(self.embeddings[col](indices))
        if parts:
            embed = torch.cat(parts, dim=1)
            if self.layer_norm is not None:
                embed = self.layer_norm(embed)
            if self.dropout is not None:
                embed = self.dropout(embed)
        else:
            embed = torch.empty(cat_tensor.size(0), 0, device=cat_tensor.device)
        return embed

    def clone(self):
        return copy.deepcopy(self)


class BusEmbeddingPolicy(nn.Module):
    """Policy network matching the legacy SAC PolicyNetwork architecture.

    Architecture:
        obs(15) + last_action(2) = 17 raw dims
        → EmbeddingLayer (5 cat cols → ~29 embed dims)
        → [embed(29) | cont(10) | last_action(2)] = 41 internal dims
        → 4 × Linear(48) + ReLU
        → mean(2), log_std(2)  (TanhGaussian output)

    Output: raw tanh ∈ [-1, 1] × action_dim.
    External mapping converts to env actions: hold = 30*tanh + 30 → [0, 60]s.
    """

    def __init__(self, num_inputs, num_actions, hidden_size,
                 embedding_layer, action_range=1.0,
                 log_std_min=-20, log_std_max=2, init_w=3e-3):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_cat_cols = len(embedding_layer.cat_cols)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def _embed(self, state):
        """Split state into categorical + continuous, return embedded."""
        cat_tensor = state[:, :self.num_cat_cols]
        num_tensor = state[:, self.num_cat_cols:]
        embedding = self.embedding_layer(cat_tensor.long())
        return torch.cat([embedding, num_tensor], dim=1)

    def forward(self, state, deterministic=False, repeat=None):
        if repeat is not None:
            state = extend_and_repeat(state, 1, repeat)
        x = self._embed(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(state.device)

        if deterministic:
            action_0 = torch.tanh(mean)
        else:
            action_0 = torch.tanh(mean + std * z)  # reparameterization

        # Raw tanh output ∈ [-1, 1] (matches ensemble checkpoint)
        # External mapping converts to env: hold = 30*tanh + 30 → [0, 60]s
        action = action_0

        # Log prob (TanhNormal, no action_range scaling)
        log_prob = (
            Normal(mean, std).log_prob(mean + std * z)
            - torch.log(1.0 - action_0.pow(2) + 1e-6)
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def log_prob(self, observations, actions):
        """Compute log probability of given actions under current policy."""
        x = self._embed(observations)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Actions are already in [-1, 1] (raw tanh space)
        action_0 = torch.clamp(actions, -1 + 1e-6, 1 - 1e-6)
        pre_tanh = atanh(action_0)

        log_prob = (
            Normal(mean, std).log_prob(pre_tanh)
            - torch.log(1.0 - action_0.pow(2) + 1e-6)
        )
        return log_prob.sum(dim=-1)


class BusEmbeddingQFunction(nn.Module):
    """Q-function matching the legacy SAC SoftQNetwork architecture.

    Architecture:
        [embed(obs) | cont(obs) | action(2)] → 4 × Linear(48) + ReLU → scalar
    """

    def __init__(self, num_inputs, num_actions, hidden_size,
                 embedding_layer, init_w=3e-3):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_cat_cols = len(embedding_layer.cat_cols)

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def _embed(self, state):
        cat_tensor = state[:, :self.num_cat_cols]
        num_tensor = state[:, self.num_cat_cols:]
        embedding = self.embedding_layer(cat_tensor.long())
        return torch.cat([embedding, num_tensor], dim=1)

    @multiple_action_q_function
    def forward(self, observations, actions):
        x = self._embed(observations)
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return torch.squeeze(x, dim=-1)


class BusEmbeddingVFunction(nn.Module):
    """V-function with categorical embedding, matching the rest of the architecture.

    Architecture:
        [embed(obs) | cont(obs)] → 4 × Linear(48) + ReLU → scalar
    """

    def __init__(self, num_inputs, hidden_size,
                 embedding_layer, init_w=3e-3):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_cat_cols = len(embedding_layer.cat_cols)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def _embed(self, state):
        cat_tensor = state[:, :self.num_cat_cols]
        num_tensor = state[:, self.num_cat_cols:]
        embedding = self.embedding_layer(cat_tensor.long())
        return torch.cat([embedding, num_tensor], dim=1)

    def forward(self, observations):
        x = self._embed(observations)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class BusSamplerPolicy(object):
    """Sampler wrapper for BusEmbeddingPolicy.

    Returns raw tanh actions ∈ [-1, 1] × action_dim.
    External mapping converts to env actions: hold = 30*tanh + 30 → [0, 60]s.
    """

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions

# class Normal(ExponentialFamily):
#     r"""
#     Creates a normal (also called Gaussian) distribution parameterized by
#     :attr:`loc` and :attr:`scale`.

#     Example::

#         >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
#         >>> m.sample()  # normally distributed with loc=0 and scale=1
#         tensor([ 0.1046])

#     Args:
#         loc (float or Tensor): mean of the distribution (often referred to as mu)
#         scale (float or Tensor): standard deviation of the distribution
#             (often referred to as sigma)
#     """
#     arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
#     support = constraints.real
#     has_rsample = True
#     _mean_carrier_measure = 0

#     @property
#     def mean(self):
#         return self.loc

#     @property
#     def stddev(self):
#         return self.scale

#     @property
#     def variance(self):
#         return self.stddev.pow(2)

#     def __init__(self, loc, scale, eps=1e-6, validate_args=None):
#         self.loc, self.scale = broadcast_all(loc, scale)
#         self.eps = eps
#         if isinstance(loc, Number) and isinstance(scale, Number):
#             batch_shape = torch.Size()
#         else:
#             batch_shape = self.loc.size()
#         super(Normal, self).__init__(batch_shape, validate_args=validate_args)

#     def expand(self, batch_shape, _instance=None):
#         new = self._get_checked_instance(Normal, _instance)
#         batch_shape = torch.Size(batch_shape)
#         new.loc = self.loc.expand(batch_shape)
#         new.scale = self.scale.expand(batch_shape)
#         super(Normal, new).__init__(batch_shape, validate_args=False)
#         new._validate_args = self._validate_args
#         return new

#     def sample(self, sample_shape=torch.Size()):
#         shape = self._extended_shape(sample_shape)
#         with torch.no_grad():
#             return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

#     def rsample(self, sample_shape=torch.Size()):
#         shape = self._extended_shape(sample_shape)
#         eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
#         return self.loc + eps * self.scale

#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         # compute the variance
#         var = (self.scale ** 2)
#         log_scale = math.log(self.scale + self.eps) if isinstance(self.scale, Real) else self.scale.log()
#         # ipdb.set_trace()
#         return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

#     def cdf(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

#     def icdf(self, value):
#         return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

#     def entropy(self):
#         return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

#     @property
#     def _natural_params(self):
#         return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

#     def _log_normalizer(self, x, y):
#         return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


# ══════════════════════════════════════════════════════════════════
# Ensemble Q-network (RE-SAC / RORL style)
# ══════════════════════════════════════════════════════════════════

import math as _math

class VectorizedLinear(nn.Module):
    """Batched linear layer for ensemble Q-networks."""
    def __init__(self, in_f, out_f, E):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, in_f, out_f))
        self.b = nn.Parameter(torch.empty(E, 1, out_f))
        for i in range(E):
            nn.init.kaiming_uniform_(self.w[i], a=_math.sqrt(5))
        fan, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
        bd = 1 / _math.sqrt(fan) if fan > 0 else 0
        nn.init.uniform_(self.b, -bd, bd)

    def forward(self, x):
        return x @ self.w + self.b


class BusEnsembleCritic(nn.Module):
    """Vectorized ensemble of E Q-networks with embedding layer.

    Input: (obs[17], action[2]) → EmbeddingLayer → concat → E parallel MLPs → (E, B) Q-values.
    """
    def __init__(self, state_dim, action_dim, hidden, E, embedding_layer):
        super().__init__()
        self.emb = embedding_layer
        self.E = E
        self.num_cat = len(embedding_layer.cat_cols)
        self.net = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden, E),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            VectorizedLinear(hidden, hidden, E),
            nn.ReLU(),
            VectorizedLinear(hidden, hidden, E),
            nn.ReLU(),
            VectorizedLinear(hidden, 1, E),
        )

    def forward(self, state, action):
        cat = state[:, :self.num_cat]
        cont = state[:, self.num_cat:]
        emb = self.emb(cat)
        x = torch.cat([emb, cont, action], dim=1)
        x = x.unsqueeze(0).repeat_interleave(self.E, dim=0)  # (E, B, dim)
        return self.net(x).squeeze(-1)  # (E, B)
