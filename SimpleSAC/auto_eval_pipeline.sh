#!/bin/bash
# auto_eval_pipeline.sh — Automated: wait for CQL, eval, then try reward shaping
# Run: nohup bash auto_eval_pipeline.sh > /tmp/auto_pipeline.log 2>&1 &

set -e
cd "$(dirname "$0")"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
export LIBSUMO_AS_TRACI=1

echo "=== Auto Eval Pipeline ==="
echo "$(date): Starting"

# ── Step 1: Wait for CQL training to finish ──
echo ""
echo "Step 1: Waiting for CQL training to complete..."
CQL_CKPT="../experiment_output/offline_ensemble_cql/offline_ensemble_final.pt"
while [ ! -f "$CQL_CKPT" ]; do
    sleep 60
done
echo "$(date): CQL training complete!"

# ── Step 2: SUMO eval of CQL ──
echo ""
echo "Step 2: Running SUMO eval of CQL..."
# Update eval script to load from CQL checkpoint
$PYTHON eval_offline_on_sumo.py 2>&1 | grep -E "reward=|RESULTS|best|offline_rl|ep39|zero"
CQL_RESULT=$?
echo "$(date): CQL eval done"

# ── Step 3: Run reward shaping version ──
echo ""
echo "Step 3: Training ensemble + reward shaping (60K steps)..."
$PYTHON train_offline_ensemble.py \
    --n_steps 60000 \
    --use_reward_shaping \
    --rs_pretrain_steps 5000 \
    --device cpu \
    2>&1 | tail -20
echo "$(date): Reward shaping training done"

# ── Step 4: SUMO eval of reward shaping ──
echo ""
echo "Step 4: Running SUMO eval of reward shaping..."
# Need to point eval to the RS checkpoint
RS_CKPT="../experiment_output/offline_ensemble_rs/offline_ensemble_final.pt"
if [ -f "$RS_CKPT" ]; then
    $PYTHON -c "
import os, sys, torch, numpy as np, copy
sys.path.insert(0, '.')
sys.path.insert(0, '../bus_h2o')
from train_offline_ensemble import PolicyNet, EnsembleCritic
from model import EmbeddingLayer
from eval_offline_on_sumo import (
    _build_sumo_indices, event_to_obs, compute_reward, run_episode,
    SUMO_DIR, EDGE_XML, SCHEDULE_XML
)
from common.data_utils import build_edge_linear_map, set_route_length

line_idx_map, bus_idx_map = _build_sumo_indices(SCHEDULE_XML)
em = build_edge_linear_map(EDGE_XML, '7X')
set_route_length(max(em.values()))

from sumo_env.rl_bridge import SumoRLBridge
bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=18000)

cat_cols = ['line_id','bus_id','station_id','time_period','direction']
cat_code_dict = {'line_id':{i:i for i in range(12)}, 'bus_id':{i:i for i in range(389)},
                 'station_id':{i:i for i in range(1)}, 'time_period':{i:i for i in range(1)},
                 'direction':{0:0,1:1}}
emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb.output_dim + 12
ckpt = torch.load('$RS_CKPT', map_location='cpu', weights_only=True)
pi = PolicyNet(state_dim, 48, copy.deepcopy(emb))
pi.load_state_dict(ckpt['policy'])
pi.eval()

def fn(ev, obs, bid, la):
    prev_a = la.get(bid, np.zeros(2, dtype=np.float32))
    return pi.get_action(np.concatenate([obs, prev_a]), deterministic=True)

r,n,t = run_episode(bridge, line_idx_map, bus_idx_map, fn, 'rs')
print(f'Ensemble+RS on SUMO: reward={r:,.0f}, decisions={n}, per_step={r/n:.1f}')

def zero_fn(ev,obs,bid,la):
    return np.array([-1.0, 0.0], dtype=np.float32)
r0,n0,t0 = run_episode(bridge, line_idx_map, bus_idx_map, zero_fn, 'zero')
print(f'Zero-hold on SUMO:   reward={r0:,.0f}, decisions={n0}, per_step={r0/n0:.1f}')
bridge.close()
print(f'Reference: ep39=-649K~-683K, ensemble_base=-668K')
" 2>&1 | grep -E "reward=|Reference|Ensemble|Zero"
fi

echo ""
echo "=== Pipeline Complete ==="
echo "$(date)"
echo ""
echo "Summary of all methods on SUMO:"
echo "  ep39 (reference):     -649K ~ -683K"
echo "  Ensemble base:        -668K"
echo "  H2O+ (twin-Q+SUMO):  -705K"
echo "  Check logs above for CQL and RS results"
