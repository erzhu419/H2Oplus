"""
train_offline_only.py
=====================
Pure offline RL on SUMO data only. No SIM rollout.
Verifies that the offline data + architecture can converge.

Uses the same AWR-style pretrain as h2oplus_bus.py:
  - Q-function regression on (s, a, r, s') transitions
  - V-function quantile regression
  - Policy: exponential advantage weighted regression

Periodically evaluates action distribution to verify policy learns to
hold more when headway deviates from target.

Run:
    cd H2Oplus/SimpleSAC
    conda run -n LSTM-RL python train_offline_only.py --n_steps 100000
"""

import os, sys, time, math, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from bus_replay_buffer import BusMixedReplayBuffer
from h2oplus_bus import H2OPlusBus
from model import (
    EmbeddingLayer, BusEmbeddingPolicy,
    BusEmbeddingQFunction, BusEmbeddingVFunction, BusSamplerPolicy,
)
from common.data_utils import set_route_length, build_edge_linear_map

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument('--n_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--eval_every', type=int, default=5000)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Output ──
out_dir = os.path.join(_H2O_ROOT, "experiment_output", "offline_only")
os.makedirs(out_dir, exist_ok=True)

# ── Route length ──
edge_xml = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    route_length = max(edge_map.values()) if edge_map else 13119.0
else:
    route_length = 13119.0
set_route_length(route_length)

# ── Load offline data ──
print("Loading offline data...")
ds_file = os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5")
replay_buffer = BusMixedReplayBuffer(
    state_dim=17, action_dim=2, context_dim=30,
    dataset_file=ds_file, device=args.device,
    buffer_ratio=1.0,  # no online needed
    reward_scale=1.0, reward_bias=0.0,
    action_scale=1.0, action_bias=0.0,
)
print(f"Loaded {replay_buffer.fixed_dataset_size:,} offline transitions")
r_mean, r_std = replay_buffer.get_reward_stats()
print(f"Reward stats: mean={r_mean:.2f}, std={r_std:.2f}")

# ── Build networks (same arch as h2o+_bus_main.py) ──
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
qf1 = BusEmbeddingQFunction(
    num_inputs=state_dim, num_actions=action_dim,
    hidden_size=hidden_size, embedding_layer=embedding_template.clone(),
)
target_qf1 = deepcopy(qf1)
qf2 = BusEmbeddingQFunction(
    num_inputs=state_dim, num_actions=action_dim,
    hidden_size=hidden_size, embedding_layer=embedding_template.clone(),
)
target_qf2 = deepcopy(qf2)
vf = BusEmbeddingVFunction(
    num_inputs=state_dim, hidden_size=hidden_size,
    embedding_layer=embedding_template.clone(),
)

# H2O+ config (offline-only: no disc needed)
config = H2OPlusBus.get_default_config()
config.device = args.device

h2o = H2OPlusBus(
    config, policy, qf1, qf2, target_qf1, target_qf2, vf,
    replay_buffer, discriminator=None,
)
h2o.torch_to_device(args.device)

sampler_policy = BusSamplerPolicy(policy, args.device)

# ── CSV logger ──
csv_path = os.path.join(out_dir, "train_log.csv")
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['step', 'policy_loss', 'qf1_loss', 'qf2_loss', 'vf_loss',
                      'vf_pred', 'alpha', 'wall_sec'])

# ── Training ──
print(f"\nStarting offline training for {args.n_steps} steps, batch_size={args.batch_size}")
t0 = time.time()
metrics_history = []

for step in range(1, args.n_steps + 1):
    m = h2o.train(args.batch_size, pretrain_steps=args.n_steps)  # always pretrain mode

    if step % 100 == 0:
        csv_writer.writerow([
            step, m.get('policy_loss', 0), m.get('real_qf1_loss', 0),
            m.get('real_qf2_loss', 0), m.get('vf_loss', 0),
            m.get('vf_pred', 0), 0.0, time.time() - t0,
        ])
        csv_file.flush()

    if step % 1000 == 0:
        print(f"  Step {step:6d}/{args.n_steps}: "
              f"pi_loss={m.get('policy_loss', 0):.4f}, "
              f"qf1={m.get('real_qf1_loss', 0):.2f}, "
              f"vf={m.get('vf_loss', 0):.4f}, "
              f"vf_pred={m.get('vf_pred', 0):.2f}, "
              f"wall={time.time()-t0:.0f}s")

    if step % args.eval_every == 0:
        metrics_history.append({
            'step': step,
            'policy_loss': m.get('policy_loss', 0),
            'qf1_loss': m.get('real_qf1_loss', 0),
            'vf_loss': m.get('vf_loss', 0),
        })

        # ── Action distribution check ──
        # Sample 10K obs, run policy, analyze actions by headway
        with torch.no_grad():
            batch = replay_buffer.sample(min(10000, replay_buffer.fixed_dataset_size), scope="real")
            obs = batch["observations"]
            actions_pred, log_pi = policy(obs)

            # Convert to numpy
            obs_np = obs.cpu().numpy()
            act_np = actions_pred.cpu().numpy()

            # Map raw tanh to hold/speed
            hold_raw = act_np[:, 0]  # tanh output
            hold_sec = 30.0 * hold_raw + 30.0  # [0, 60]

            fwd_hw = obs_np[:, 5]  # forward headway
            target_hw = obs_np[:, 8]  # target headway
            hw_gap = fwd_hw - target_hw  # positive = bus ahead is far, negative = bunching

        # Bin by headway gap
        bins = [(-999, -200), (-200, -100), (-100, -50), (-50, 0),
                (0, 50), (50, 100), (100, 200), (200, 999)]
        print(f"\n  [Step {step}] Hold time by headway gap:")
        print(f"    {'hw_gap':>12s}  {'mean_hold':>10s}  {'std_hold':>10s}  {'count':>6s}")
        for lo, hi in bins:
            mask = (hw_gap >= lo) & (hw_gap < hi)
            if mask.sum() > 0:
                h = hold_sec[mask]
                print(f"    [{lo:>5d},{hi:>5d})  {h.mean():>10.1f}  {h.std():>10.1f}  {mask.sum():>6d}")

csv_file.close()

# ── Save final model ──
ckpt_path = os.path.join(out_dir, "offline_final.pt")
torch.save({
    'policy': policy.state_dict(),
    'qf1': qf1.state_dict(),
    'qf2': qf2.state_dict(),
    'vf': vf.state_dict(),
    'step': args.n_steps,
}, ckpt_path)
print(f"\nFinal model saved: {ckpt_path}")

# ── Plot convergence ──
if metrics_history:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = [m['step'] for m in metrics_history]
    axes[0].plot(steps, [m['policy_loss'] for m in metrics_history], 'b-')
    axes[0].set_title('Policy Loss'); axes[0].set_xlabel('Step')
    axes[1].plot(steps, [m['qf1_loss'] for m in metrics_history], 'r-')
    axes[1].set_title('Q1 Loss'); axes[1].set_xlabel('Step')
    axes[2].plot(steps, [m['vf_loss'] for m in metrics_history], 'g-')
    axes[2].set_title('V Loss'); axes[2].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close()
    print(f"Convergence plot: {out_dir}/convergence.png")

print(f"\nDone in {time.time()-t0:.0f}s. Log: {csv_path}")
