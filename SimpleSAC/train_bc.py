"""
train_bc.py
===========
Pure Behavior Cloning (BC) baseline for the H2O+ paper (Issue 6 / R2 reviewer).

Trains a BusEmbeddingPolicy on offline SUMO transitions using ONLY the
policy log-likelihood loss (no Q-loss, no V-loss). Two data variants:

    --data full   : all merged offline data (merged_all_v2.h5)
    --data ep39   : ep39-only subset (sumo_sac_seed*.h5 — the
                    checkpoint_episode_39 SAC reference policy rollouts)

Loss
----
We minimise -E[log pi(a | s)] under the TanhGaussian policy. Actions in
the buffer are stored in raw tanh space ([-1, 1]); BusEmbeddingPolicy.log_prob
correctly inverts the tanh and adds the change-of-variable correction.

Run
---
    cd H2Oplus/SimpleSAC
    conda run -n LSTM-RL python train_bc.py --data full --seed 42
    conda run -n LSTM-RL python train_bc.py --data ep39 --seed 42
"""

import os
import sys
import time
import csv
import argparse
from copy import deepcopy

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from bus_replay_buffer import BusMixedReplayBuffer
from model import EmbeddingLayer, BusEmbeddingPolicy
from common.data_utils import set_route_length, build_edge_linear_map

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['full', 'ep39'], required=True,
                    help="Which offline data to BC on")
parser.add_argument('--n_steps', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1000)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Output dir ────────────────────────────────────────────────────────────────
out_dir = os.path.join(_H2O_ROOT, "experiment_output",
                       f"bc_{args.data}_seed{args.seed}")
os.makedirs(out_dir, exist_ok=True)
print(f"[BC] Output dir: {out_dir}")

# ── Route length (mirrors train_offline_only.py) ──────────────────────────────
edge_xml = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    route_length = max(edge_map.values()) if edge_map else 13119.0
else:
    route_length = 13119.0
set_route_length(route_length)

# ── Load offline data ─────────────────────────────────────────────────────────
print(f"[BC] Loading offline data ({args.data})...")
if args.data == 'full':
    ds_file = os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5")
    replay_buffer = BusMixedReplayBuffer(
        state_dim=17, action_dim=2, context_dim=30,
        dataset_file=ds_file, device=args.device,
        buffer_ratio=0.0,  # no online needed
        reward_scale=1.0, reward_bias=0.0,
        action_scale=1.0, action_bias=0.0,
    )
elif args.data == 'ep39':
    # ep39 = checkpoint_episode_39 SAC rollouts (sumo_sac_seed*.h5).
    # Load via dataset_dir + glob to grab only those files.
    ds_dir = os.path.join(_BUS_H2O, "datasets_v2")
    replay_buffer = BusMixedReplayBuffer(
        state_dim=17, action_dim=2, context_dim=30,
        dataset_dir=ds_dir, dataset_glob="sumo_sac_seed*.h5",
        device=args.device,
        buffer_ratio=0.0,
        reward_scale=1.0, reward_bias=0.0,
        action_scale=1.0, action_bias=0.0,
        skip_snapshots=True,  # not needed for BC
    )
print(f"[BC] Loaded {replay_buffer.fixed_dataset_size:,} offline transitions")

# ── Build policy (same architecture as train_offline_only.py) ─────────────────
cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
cat_code_dict = {
    'line_id':     {i: i for i in range(12)},
    'bus_id':      {i: i for i in range(389)},
    'station_id':  {i: i for i in range(1)},
    'time_period': {i: i for i in range(1)},
    'direction':   {0: 0, 1: 1},
}
obs_dim = 17
action_dim = 2
hidden_size = 48

embedding_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = embedding_template.output_dim + (obs_dim - len(cat_cols))

policy = BusEmbeddingPolicy(
    num_inputs=state_dim, num_actions=action_dim,
    hidden_size=hidden_size, embedding_layer=embedding_template.clone(),
    action_range=1.0,
)
policy = policy.to(args.device)
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

n_params = sum(p.numel() for p in policy.parameters())
print(f"[BC] Policy parameters: {n_params:,}")

# ── CSV logger ────────────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, "train_log.csv")
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['step', 'bc_loss', 'mean_log_prob', 'wall_sec'])

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"[BC] Starting training for {args.n_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
t0 = time.time()
loss_history = []

for step in range(1, args.n_steps + 1):
    batch = replay_buffer.sample(args.batch_size, scope="real")
    obs = batch["observations"]
    actions = batch["actions"]

    # BC loss: negative mean log-likelihood of dataset actions under policy
    log_pi = policy.log_prob(obs, actions)  # (B,)
    bc_loss = -log_pi.mean()

    optimizer.zero_grad()
    bc_loss.backward()
    # Light gradient clipping to prevent occasional explosions
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
    optimizer.step()

    if step % args.log_every == 0:
        csv_writer.writerow([
            step, float(bc_loss.item()), float(log_pi.mean().item()),
            time.time() - t0,
        ])
        csv_file.flush()
        loss_history.append((step, float(bc_loss.item())))

    if step % args.print_every == 0:
        print(f"  Step {step:6d}/{args.n_steps}: "
              f"bc_loss={bc_loss.item():.4f}  "
              f"mean_log_pi={log_pi.mean().item():.4f}  "
              f"wall={time.time()-t0:.0f}s")

csv_file.close()

# ── Save final checkpoint ─────────────────────────────────────────────────────
ckpt_path = os.path.join(out_dir, "bc_final.pt")
torch.save({
    'policy': policy.state_dict(),
    'step': args.n_steps,
    'data': args.data,
    'seed': args.seed,
    'final_bc_loss': loss_history[-1][1] if loss_history else None,
}, ckpt_path)
print(f"[BC] Final model saved: {ckpt_path}")

# ── Quick sanity check: forward pass produces valid actions ───────────────────
print("[BC] Sanity check: forward pass on 256 random offline obs...")
policy.eval()
with torch.no_grad():
    batch = replay_buffer.sample(256, scope="real")
    obs = batch["observations"]
    a_det, _ = policy(obs, deterministic=True)
    a_sto, _ = policy(obs, deterministic=False)
    nan_det = torch.isnan(a_det).any().item()
    nan_sto = torch.isnan(a_sto).any().item()
    in_range_det = (a_det.min().item() >= -1.0 - 1e-4 and a_det.max().item() <= 1.0 + 1e-4)
    in_range_sto = (a_sto.min().item() >= -1.0 - 1e-4 and a_sto.max().item() <= 1.0 + 1e-4)

sanity_pass = (not nan_det) and (not nan_sto) and in_range_det and in_range_sto
print(f"  deterministic action: shape={tuple(a_det.shape)}, "
      f"min={a_det.min().item():.4f}, max={a_det.max().item():.4f}, "
      f"mean={a_det.mean().item():.4f}, nan={nan_det}")
print(f"  stochastic    action: shape={tuple(a_sto.shape)}, "
      f"min={a_sto.min().item():.4f}, max={a_sto.max().item():.4f}, "
      f"mean={a_sto.mean().item():.4f}, nan={nan_sto}")
print(f"  SANITY: {'PASS' if sanity_pass else 'FAIL'}")

# Persist sanity result
with open(os.path.join(out_dir, "sanity.txt"), 'w') as f:
    f.write(f"sanity_pass={sanity_pass}\n")
    f.write(f"a_det min/max/mean = {a_det.min().item():.6f} {a_det.max().item():.6f} {a_det.mean().item():.6f}\n")
    f.write(f"a_sto min/max/mean = {a_sto.min().item():.6f} {a_sto.max().item():.6f} {a_sto.mean().item():.6f}\n")
    f.write(f"nan_det={nan_det} nan_sto={nan_sto}\n")
    f.write(f"final_bc_loss={loss_history[-1][1] if loss_history else None}\n")

# ── Convergence plot ──────────────────────────────────────────────────────────
if loss_history:
    steps = [s for s, _ in loss_history]
    losses = [l for _, l in loss_history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, 'b-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('BC loss (-log pi(a|s))')
    ax.set_title(f'BC training: data={args.data} seed={args.seed}')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=120)
    plt.close()

print(f"[BC] Done in {time.time()-t0:.0f}s. Log: {csv_path}")
