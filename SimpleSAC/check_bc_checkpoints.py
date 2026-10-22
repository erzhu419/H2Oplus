"""
check_bc_checkpoints.py
=======================
Load each BC checkpoint and verify policy.forward(s) produces valid actions
(no NaNs, in [-1, 1]).
"""

import os
import sys
import json

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

# Setup route length
edge_xml = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    set_route_length(max(edge_map.values()) if edge_map else 13119.0)
else:
    set_route_length(13119.0)

# Load a small batch of obs from ep39 (used as sanity test inputs for both)
print("[Check] Loading 1024 sanity obs from ep39 data ...")
ds_dir = os.path.join(_BUS_H2O, "datasets_v2")
buf = BusMixedReplayBuffer(
    state_dim=17, action_dim=2, context_dim=30,
    dataset_dir=ds_dir, dataset_glob="sumo_sac_seed42.h5",
    device='cpu', buffer_ratio=0.0, skip_snapshots=True,
)
batch = buf.sample(1024, scope="real")
obs = batch["observations"]
print(f"[Check] Obs tensor shape: {obs.shape}")

# Build policy template
cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
cat_code_dict = {
    'line_id':     {i: i for i in range(12)},
    'bus_id':      {i: i for i in range(389)},
    'station_id':  {i: i for i in range(1)},
    'time_period': {i: i for i in range(1)},
    'direction':   {0: 0, 1: 1},
}

def build_policy():
    embedding_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
    state_dim = embedding_template.output_dim + (17 - len(cat_cols))
    return BusEmbeddingPolicy(
        num_inputs=state_dim, num_actions=2,
        hidden_size=48, embedding_layer=embedding_template.clone(),
        action_range=1.0,
    )

results = []
SEEDS = [42, 123, 456, 789, 2024]
for data in ['full', 'ep39']:
    for seed in SEEDS:
        ckpt_path = os.path.join(
            _H2O_ROOT, "experiment_output", f"bc_{data}_seed{seed}", "bc_final.pt"
        )
        entry = {"data": data, "seed": seed, "ckpt_path": ckpt_path}
        if not os.path.exists(ckpt_path):
            entry.update({"status": "MISSING"})
            results.append(entry)
            print(f"  [{data}/{seed}] MISSING ckpt {ckpt_path}")
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            policy = build_policy()
            policy.load_state_dict(ckpt['policy'])
            policy.eval()
            with torch.no_grad():
                a_det, _ = policy(obs, deterministic=True)
                a_sto, _ = policy(obs, deterministic=False)
            stats = {
                "final_bc_loss": ckpt.get("final_bc_loss"),
                "step": ckpt.get("step"),
                "a_det_min": float(a_det.min()),
                "a_det_max": float(a_det.max()),
                "a_det_mean": float(a_det.mean()),
                "a_sto_min": float(a_sto.min()),
                "a_sto_max": float(a_sto.max()),
                "a_sto_mean": float(a_sto.mean()),
                "nan_det": bool(torch.isnan(a_det).any()),
                "nan_sto": bool(torch.isnan(a_sto).any()),
            }
            sanity = (not stats["nan_det"]) and (not stats["nan_sto"]) and \
                     (stats["a_det_min"] >= -1.0 - 1e-3) and (stats["a_det_max"] <= 1.0 + 1e-3) and \
                     (stats["a_sto_min"] >= -1.0 - 1e-3) and (stats["a_sto_max"] <= 1.0 + 1e-3)
            entry.update({"status": "PASS" if sanity else "FAIL", **stats})
            print(f"  [{data}/{seed}] {entry['status']}: "
                  f"loss={stats['final_bc_loss']:.4f}, "
                  f"a_det range=[{stats['a_det_min']:.3f},{stats['a_det_max']:.3f}], "
                  f"a_sto range=[{stats['a_sto_min']:.3f},{stats['a_sto_max']:.3f}]")
        except Exception as e:
            entry.update({"status": "ERROR", "error": str(e)})
            print(f"  [{data}/{seed}] ERROR: {e}")
        results.append(entry)

# Save results
out = os.path.join(_H2O_ROOT, "experiment_output", "bc_sanity_results.json")
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved sanity results to {out}")

n_pass = sum(1 for r in results if r.get("status") == "PASS")
print(f"SUMMARY: {n_pass}/{len(results)} checkpoints pass sanity check.")
