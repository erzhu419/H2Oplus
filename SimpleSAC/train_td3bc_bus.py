"""
train_td3bc_bus.py
==================
TD3+BC (Fujimoto & Gu, 2021) baseline on bus-holding offline data.

  - Deterministic actor (mean of BusEmbeddingPolicy).
  - Twin Q networks with target smoothing.
  - Policy update: -Q(s, mu(s)) + alpha_bc * MSE(mu(s), a_data)
    (the alpha is normalised by 1/|Q|.mean() per the paper).
  - Target Q uses target-policy + clipped noise.

Run:
    cd H2Oplus/SimpleSAC
    conda run -n LSTM-RL python train_td3bc_bus.py --n_steps 100000 --seed 42
"""

import os, sys, time, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from bus_replay_buffer import BusMixedReplayBuffer
from model import EmbeddingLayer, BusEmbeddingPolicy, BusEmbeddingQFunction
from common.data_utils import set_route_length, build_edge_linear_map

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--policy_noise', type=float, default=0.2)
parser.add_argument('--noise_clip', type=float, default=0.5)
parser.add_argument('--policy_freq', type=int, default=2)
parser.add_argument('--alpha_bc', type=float, default=2.5,
                    help="BC regularization weight (paper default 2.5)")
parser.add_argument('--print_every', type=int, default=2000)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

out_dir = os.path.join(_H2O_ROOT, "experiment_output", f"td3bc_seed{args.seed}")
os.makedirs(out_dir, exist_ok=True)
print(f"[TD3+BC] Output dir: {out_dir}")

edge_xml = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
route_length = max(build_edge_linear_map(edge_xml, "7X").values()) if os.path.exists(edge_xml) else 13119.0
set_route_length(route_length)

print("[TD3+BC] Loading offline data...")
ds_file = os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5")
replay_buffer = BusMixedReplayBuffer(
    state_dim=17, action_dim=2, context_dim=30,
    dataset_file=ds_file, device=args.device, buffer_ratio=1.0,
    reward_scale=1.0, reward_bias=0.0, action_scale=1.0, action_bias=0.0,
)
print(f"[TD3+BC] Loaded {replay_buffer.fixed_dataset_size:,} transitions")
r_mean, r_std = replay_buffer.get_reward_stats()

cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
cat_code_dict = {
    'line_id':     {i: i for i in range(12)},
    'bus_id':      {i: i for i in range(389)},
    'station_id':  {i: i for i in range(1)},
    'time_period': {i: i for i in range(1)},
    'direction':   {0: 0, 1: 1},
}
embedding_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = embedding_template.output_dim + (17 - len(cat_cols))
hidden_size = 48

policy = BusEmbeddingPolicy(num_inputs=state_dim, num_actions=2, hidden_size=hidden_size,
                            embedding_layer=embedding_template.clone(), action_range=1.0).to(args.device)
target_policy = deepcopy(policy)
qf1 = BusEmbeddingQFunction(num_inputs=state_dim, num_actions=2, hidden_size=hidden_size,
                            embedding_layer=embedding_template.clone()).to(args.device)
qf2 = BusEmbeddingQFunction(num_inputs=state_dim, num_actions=2, hidden_size=hidden_size,
                            embedding_layer=embedding_template.clone()).to(args.device)
target_qf1 = deepcopy(qf1)
target_qf2 = deepcopy(qf2)

opt_policy = torch.optim.Adam(policy.parameters(), lr=args.lr)
opt_q1 = torch.optim.Adam(qf1.parameters(), lr=args.lr)
opt_q2 = torch.optim.Adam(qf2.parameters(), lr=args.lr)

def soft_update(net, target, tau):
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

csv_path = os.path.join(out_dir, "train_log.csv")
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['step', 'policy_loss', 'q1_loss', 'q2_loss', 'q_mean', 'bc_mse', 'wall_sec'])

print(f"[TD3+BC] Training for {args.n_steps} steps, batch_size={args.batch_size}, alpha_bc={args.alpha_bc}")
t0 = time.time()
metrics_history = []

for step in range(1, args.n_steps + 1):
    batch = replay_buffer.sample(args.batch_size, scope="real")
    obs = batch["observations"]
    actions = batch["actions"]
    rewards = (batch["rewards"].squeeze() - r_mean) / r_std
    next_obs = batch["next_observations"]
    dones = batch["dones"].squeeze()

    # ── Q regression: target with policy smoothing noise ──
    with torch.no_grad():
        next_actions, _ = target_policy(next_obs, deterministic=True)
        noise = (torch.randn_like(next_actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_actions = (next_actions + noise).clamp(-1.0, 1.0)
        target_q = torch.min(target_qf1(next_obs, next_actions), target_qf2(next_obs, next_actions))
        td_target = rewards + (1.0 - dones) * args.gamma * target_q

    q1_pred = qf1(obs, actions); q2_pred = qf2(obs, actions)
    q1_loss = F.mse_loss(q1_pred, td_target); q2_loss = F.mse_loss(q2_pred, td_target)
    opt_q1.zero_grad(); q1_loss.backward(); opt_q1.step()
    opt_q2.zero_grad(); q2_loss.backward(); opt_q2.step()

    bc_mse = torch.tensor(0.0)
    policy_loss_val = 0.0
    if step % args.policy_freq == 0:
        # Deterministic policy action (mean of Gaussian)
        pi_actions, _ = policy(obs, deterministic=True)
        # Q signal — use qf1 only (TD3 paper convention)
        q_pi = qf1(obs, pi_actions)
        # TD3+BC: normalised lambda = alpha / |Q.detach()|.mean()
        lam = args.alpha_bc / q_pi.abs().mean().detach().clamp_min(1e-6)
        bc_mse = F.mse_loss(pi_actions, actions)
        policy_loss = -lam * q_pi.mean() + bc_mse
        opt_policy.zero_grad(); policy_loss.backward(); opt_policy.step()
        policy_loss_val = policy_loss.item()
        soft_update(qf1, target_qf1, args.tau)
        soft_update(qf2, target_qf2, args.tau)
        soft_update(policy, target_policy, args.tau)

    if step % 100 == 0:
        csv_writer.writerow([step, policy_loss_val, q1_loss.item(), q2_loss.item(),
                             q1_pred.mean().item(), bc_mse.item(), time.time() - t0])
        csv_file.flush()

    if step % args.print_every == 0:
        print(f"  Step {step:6d}/{args.n_steps}  pi={policy_loss_val:.3f}  "
              f"q1={q1_loss.item():.3f}  bc_mse={bc_mse.item():.4f}  "
              f"wall={time.time()-t0:.0f}s")
        metrics_history.append({'step': step, 'policy_loss': policy_loss_val,
                                'q1_loss': q1_loss.item(), 'bc_mse': bc_mse.item()})

csv_file.close()

ckpt_path = os.path.join(out_dir, "td3bc_final.pt")
torch.save({
    'policy': policy.state_dict(),
    'qf1': qf1.state_dict(), 'qf2': qf2.state_dict(),
    'step': args.n_steps,
}, ckpt_path)
print(f"[TD3+BC] Final model saved: {ckpt_path}")

if metrics_history:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    steps = [m['step'] for m in metrics_history]
    axes[0].plot(steps, [m['policy_loss'] for m in metrics_history], 'b-')
    axes[0].set_title('Policy Loss')
    axes[1].plot(steps, [m['q1_loss'] for m in metrics_history], 'r-')
    axes[1].set_title('Q1 Loss')
    axes[2].plot(steps, [m['bc_mse'] for m in metrics_history], 'g-')
    axes[2].set_title('BC MSE')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close()

print(f"[TD3+BC] Done in {time.time()-t0:.0f}s. Log: {csv_path}")
