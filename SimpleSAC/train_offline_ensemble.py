"""
train_offline_ensemble.py
=========================
Offline RL with RE-SAC-style ensemble pessimism on SUMO bus holding data.

Key differences from train_offline_only.py (AWR):
  1. Ensemble Q-network (E=5) instead of twin-Q + V-function
  2. Policy gradient through pessimistic Q (mean - β*std) instead of AWR
  3. Independent targets per ensemble member
  4. OOD penalty (Q-std regularization)
  5. Behavior cloning regularization for safety

Based on:
  - RORL: ensemble min-Q for pessimism + smoothing
  - RE-SAC: mean-std pessimism + independent targets + OOD penalty

Run:
    cd H2Oplus/SimpleSAC
    conda run -n LSTM-RL python train_offline_ensemble.py --n_steps 60000
"""

import os, sys, time, math, copy, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument('--n_steps', type=int, default=60000)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--eval_every', type=int, default=5000)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ensemble_size', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=48)
parser.add_argument('--beta', type=float, default=-2.0, help='Pessimism coefficient (negative=pessimistic)')
parser.add_argument('--beta_ood', type=float, default=0.01, help='OOD Q-std penalty')
parser.add_argument('--beta_bc', type=float, default=0.005, help='Behavior cloning weight')
parser.add_argument('--gamma', type=float, default=0.80)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--soft_tau', type=float, default=1e-2)
parser.add_argument('--critic_actor_ratio', type=int, default=3, help='Critic updates per actor update')
parser.add_argument('--max_alpha', type=float, default=0.6)
parser.add_argument('--reward_scale', type=float, default=10.0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device)

# ── Output ──
out_dir = os.path.join(_H2O_ROOT, "experiment_output", "offline_ensemble")
os.makedirs(out_dir, exist_ok=True)

# ── Load data ──
from bus_replay_buffer import BusMixedReplayBuffer
from common.data_utils import set_route_length, build_edge_linear_map

edge_xml = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    em = build_edge_linear_map(edge_xml, "7X")
    set_route_length(max(em.values()) if em else 13119.0)

print("Loading offline data...")
ds_file = os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5")
buf = BusMixedReplayBuffer(
    state_dim=17, action_dim=2, context_dim=30,
    dataset_file=ds_file, device=args.device,
    buffer_ratio=1.0, reward_scale=1.0, reward_bias=0.0,
    action_scale=1.0, action_bias=0.0,
)
print(f"Loaded {buf.fixed_dataset_size:,} transitions")
r_mean, r_std = buf.get_reward_stats()
print(f"Reward stats: mean={r_mean:.2f}, std={r_std:.2f}")

# ══════════════════════════════════════════════════════════════════
# Networks (reuse architecture from model.py)
# ══════════════════════════════════════════════════════════════════

from model import EmbeddingLayer

cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
cat_code_dict = {
    'line_id':     {i: i for i in range(12)},
    'bus_id':      {i: i for i in range(389)},
    'station_id':  {i: i for i in range(1)},
    'time_period': {i: i for i in range(1)},
    'direction':   {0: 0, 1: 1},
}
N_CAT = len(cat_cols)
obs_dim = 17
action_dim = 2

emb_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
emb_dim = emb_template.output_dim
state_dim = emb_dim + (obs_dim - N_CAT)  # embedded cat + continuous


class VectorizedLinear(nn.Module):
    def __init__(self, in_f, out_f, E):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, in_f, out_f))
        self.b = nn.Parameter(torch.empty(E, 1, out_f))
        for i in range(E):
            nn.init.kaiming_uniform_(self.w[i], a=math.sqrt(5))
        fan, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
        bd = 1 / math.sqrt(fan) if fan > 0 else 0
        nn.init.uniform_(self.b, -bd, bd)

    def forward(self, x):
        return x @ self.w + self.b


class EnsembleCritic(nn.Module):
    """Vectorized ensemble of E Q-networks (RE-SAC style)."""
    def __init__(self, state_dim, action_dim, hidden, E, emb):
        super().__init__()
        self.emb = emb
        self.E = E
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
        cat = state[:, :N_CAT]
        cont = state[:, N_CAT:]
        emb = self.emb(cat)
        x = torch.cat([emb, cont, action], dim=1)
        x = x.unsqueeze(0).repeat_interleave(self.E, dim=0)  # (E, B, dim)
        return self.net(x).squeeze(-1)  # (E, B)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden, emb):
        super().__init__()
        self.emb = emb
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.logstd_head.weight, -3e-3, 3e-3)

    def forward(self, state):
        cat = state[:, :N_CAT]
        cont = state[:, N_CAT:]
        emb = self.emb(cat)
        h = self.net(torch.cat([emb, cont], dim=1))
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(-20, 2)
        return mean, logstd

    def evaluate(self, state, eps=1e-6):
        mean, logstd = self.forward(state)
        std = logstd.exp()
        z = torch.randn_like(mean)
        a0 = torch.tanh(mean + std * z)
        logp = (Normal(mean, std).log_prob(mean + std * z)
                - torch.log(1 - a0.pow(2) + eps))
        return a0, logp.sum(1)

    @torch.no_grad()
    def get_action(self, state_vec, deterministic=False):
        s = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        mean, logstd = self.forward(s)
        if deterministic:
            a0 = torch.tanh(mean)
        else:
            a0 = torch.tanh(mean + logstd.exp() * torch.randn_like(mean))
        return a0.cpu().numpy()[0]


# ══════════════════════════════════════════════════════════════════
# Build networks
# ══════════════════════════════════════════════════════════════════

E = args.ensemble_size
qf = EnsembleCritic(state_dim, action_dim, args.hidden_dim, E,
                     copy.deepcopy(emb_template)).to(device)
target_qf = copy.deepcopy(qf)
for p in target_qf.parameters():
    p.requires_grad_(False)

pi = PolicyNet(state_dim, args.hidden_dim,
               copy.deepcopy(emb_template)).to(device)

# Entropy tuning
target_entropy = -float(action_dim) + math.log(30.0) + math.log(0.2)  # match SUMO baseline
log_alpha = torch.tensor([math.log(0.1)], dtype=torch.float32,
                          requires_grad=True, device=device)

opt_q = optim.Adam(qf.parameters(), lr=args.lr)
opt_pi = optim.Adam(pi.parameters(), lr=args.lr)
opt_al = optim.Adam([log_alpha], lr=args.lr)

# ══════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════

csv_path = os.path.join(out_dir, "train_log.csv")
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['step', 'policy_loss', 'qf_loss', 'ood_loss', 'bc_loss',
                      'alpha', 'q_mean', 'q_std_mean', 'wall_sec'])

print(f"\nTraining: {args.n_steps} steps, E={E}, β={args.beta}, batch={args.batch_size}")
t0 = time.time()

for step in range(1, args.n_steps + 1):
    batch = buf.sample(args.batch_size, scope="real")
    S = batch["observations"]
    A = batch["actions"]
    R = batch["rewards"].squeeze()
    S2 = batch["next_observations"]
    D = batch["terminals"].squeeze() if "terminals" in batch else torch.zeros_like(R)

    # Reward normalization (global z-score, matches h2oplus_bus.py)
    R = args.reward_scale * (R - r_mean) / (r_std + 1e-6)

    # ── Critic update (every step) ──────────────────────────────
    with torch.no_grad():
        a2, lp2 = pi.evaluate(S2)
        q_next = target_qf(S2, a2)  # (E, B)
        # Independent targets: each member uses its own target
        lp2_exp = lp2.unsqueeze(0).expand(E, -1)
        alpha = min(args.max_alpha, log_alpha.exp().item())
        td_target = R.unsqueeze(0) + (1 - D.unsqueeze(0)) * args.gamma * (q_next - alpha * lp2_exp)
        # td_target: (E, B)

    q_pred = qf(S, A)  # (E, B)
    qf_loss = F.mse_loss(q_pred, td_target)

    # OOD penalty: penalize Q-disagreement
    ood_loss = q_pred.std(0).mean()
    total_q_loss = qf_loss + args.beta_ood * ood_loss

    opt_q.zero_grad()
    total_q_loss.backward()
    nn.utils.clip_grad_norm_(qf.parameters(), 1.0)
    opt_q.step()

    # ── Actor update (every critic_actor_ratio steps) ───────────
    pi_loss_val = 0.0
    bc_loss_val = 0.0
    if step % args.critic_actor_ratio == 0:
        a_new, lp_new = pi.evaluate(S)
        q_ens = qf(S, a_new)  # (E, B)

        # RE-SAC pessimism: mean + β * std (β < 0 → pessimistic)
        q_pessimistic = q_ens.mean(0) + args.beta * q_ens.std(0)
        pi_loss = (alpha * lp_new - q_pessimistic).mean()

        # Behavior cloning regularization
        bc_loss = F.mse_loss(a_new, A.detach())
        total_pi_loss = pi_loss + args.beta_bc * bc_loss

        opt_pi.zero_grad()
        total_pi_loss.backward()
        opt_pi.step()
        pi_loss_val = pi_loss.item()
        bc_loss_val = bc_loss.item()

    # ── Alpha update ────────────────────────────────────────────
    _, lp = pi.evaluate(S)
    al_loss = -(log_alpha * (lp + target_entropy).detach()).mean()
    opt_al.zero_grad()
    al_loss.backward()
    opt_al.step()

    # ── Target network Polyak update ────────────────────────────
    with torch.no_grad():
        for tp, p in zip(target_qf.parameters(), qf.parameters()):
            tp.data.mul_(1 - args.soft_tau).add_(p.data * args.soft_tau)

    # ── Logging ─────────────────────────────────────────────────
    if step % 100 == 0:
        csv_writer.writerow([
            step, pi_loss_val, qf_loss.item(), ood_loss.item(), bc_loss_val,
            alpha, q_pred.mean().item(), q_pred.std(0).mean().item(),
            time.time() - t0,
        ])
        csv_file.flush()

    if step % 1000 == 0:
        print(f"  Step {step:6d}/{args.n_steps}: "
              f"pi={pi_loss_val:.2f}, qf={qf_loss.item():.1f}, "
              f"ood={ood_loss.item():.3f}, bc={bc_loss_val:.4f}, "
              f"α={alpha:.3f}, Q={q_pred.mean().item():.1f}, "
              f"wall={time.time()-t0:.0f}s")

    if step % args.eval_every == 0:
        # Action distribution check
        with torch.no_grad():
            batch_eval = buf.sample(min(10000, buf.fixed_dataset_size), scope="real")
            obs_e = batch_eval["observations"]
            act_pred, _ = pi.evaluate(obs_e)
            obs_np = obs_e.cpu().numpy()
            act_np = act_pred.cpu().numpy()
            hold_sec = 30.0 * act_np[:, 0] + 30.0
            gap = obs_np[:, 5] - obs_np[:, 8]  # fwd_hw - static_target (approximate)

        bins = [(-999, -200), (-200, -100), (-100, 0), (0, 100), (100, 200), (200, 999)]
        print(f"\n  [Step {step}] Hold by headway gap:")
        print(f"    {'gap':>12s}  {'hold':>8s}  {'std':>8s}  {'N':>6s}")
        for lo, hi in bins:
            mask = (gap >= lo) & (gap < hi)
            if mask.sum() > 0:
                h = hold_sec[mask]
                print(f"    [{lo:>5d},{hi:>5d})  {h.mean():>8.1f}  {h.std():>8.1f}  {mask.sum():>6d}")

csv_file.close()

# ── Save ──
ckpt_path = os.path.join(out_dir, "offline_ensemble_final.pt")
torch.save({
    'policy': pi.state_dict(),
    'qf': qf.state_dict(),
    'step': args.n_steps,
    'args': vars(args),
}, ckpt_path)
print(f"\nModel saved: {ckpt_path}")

# ── Convergence plot ──
import csv as csv_mod
rows = []
with open(csv_path) as f:
    reader = csv_mod.reader(f)
    next(reader)
    for r in reader:
        rows.append([float(x) for x in r])
if rows:
    data = np.array(rows)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(data[:, 0], data[:, 1], 'b-', alpha=0.5)
    axes[0].set_title('Policy Loss')
    axes[1].plot(data[:, 0], data[:, 2], 'r-', alpha=0.5)
    axes[1].set_title('Q Loss')
    axes[2].plot(data[:, 0], data[:, 3], 'g-', alpha=0.5)
    axes[2].set_title('OOD Loss (Q-std)')
    axes[3].plot(data[:, 0], data[:, 5], 'm-', alpha=0.5)
    axes[3].set_title('Alpha')
    for ax in axes:
        ax.set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close()

print(f"Done in {time.time()-t0:.0f}s")
