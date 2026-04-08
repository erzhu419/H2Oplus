#!/bin/bash
# eval_all_ablations.sh — Evaluate all ablation checkpoints on SUMO
#
# Finds all checkpoint_best.pt in the ablation directory, evaluates each
# on a full 18000-step SUMO episode, and produces a summary table.
#
# Usage:
#   bash eval_all_ablations.sh [ablation_dir] [--parallel N]
#
# Note: SUMO (libsumo) only allows 1 session at a time, so parallel>1
#       requires sequential SUMO eval. The parallel flag controls how many
#       non-SUMO evaluations run simultaneously.

set -e

ABLATION_DIR="${1:-$(ls -dt ../experiment_output/ablation_* 2>/dev/null | head -1)}"
PARALLEL="${2:-1}"

if [ -z "$ABLATION_DIR" ] || [ ! -d "$ABLATION_DIR" ]; then
    # Fallback: eval all experiment dirs
    ABLATION_DIR="../experiment_output"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
RESULTS_FILE="$ABLATION_DIR/results_summary.csv"

echo "═══════════════════════════════════════════════════════════════"
echo "Evaluating all ablation checkpoints on SUMO"
echo "═══════════════════════════════════════════════════════════════"
echo "Directory: $ABLATION_DIR"
echo ""

# Find all checkpoint_best.pt files
CHECKPOINTS=$(find "$ABLATION_DIR" -name "checkpoint_best.pt" -o -name "checkpoint_epoch50.pt" | sort)

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found in $ABLATION_DIR"
    exit 1
fi

echo "name,sumo_reward,decisions,per_step,checkpoint" > "$RESULTS_FILE"

for ckpt in $CHECKPOINTS; do
    dir=$(dirname "$ckpt")
    name=$(basename "$dir")
    ckpt_name=$(basename "$ckpt")

    echo "Evaluating: $name ($ckpt_name)..."

    result=$(SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON -c "
import os, sys, torch, numpy as np
sys.path.insert(0, '$SCRIPT_DIR')
sys.path.insert(0, '$SCRIPT_DIR/../bus_h2o')
from model import EmbeddingLayer, BusEmbeddingPolicy
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
cat_code_dict = {'line_id':{i:i for i in range(12)},'bus_id':{i:i for i in range(389)},
                 'station_id':{i:i for i in range(1)},'time_period':{i:i for i in range(1)},
                 'direction':{0:0,1:1}}
emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb.output_dim + 12
policy = BusEmbeddingPolicy(state_dim, 2, 48, emb.clone(), action_range=1.0)
ckpt = torch.load('$ckpt', map_location='cpu', weights_only=True)
policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()

def fn(ev, obs, bid, la):
    prev_a = la.get(bid, np.zeros(2, dtype=np.float32))
    obs_aug = np.concatenate([obs, prev_a])
    with torch.no_grad():
        action, _ = policy(torch.FloatTensor(obs_aug).unsqueeze(0), deterministic=True)
    return action.cpu().numpy()[0]

r, n, t = run_episode(bridge, line_idx_map, bus_idx_map, fn, 'eval')
print(f'{r:.0f},{n},{r/n:.1f}')
bridge.close()
" 2>/dev/null | tail -1)

    if [ -n "$result" ]; then
        reward=$(echo "$result" | cut -d',' -f1)
        decisions=$(echo "$result" | cut -d',' -f2)
        per_step=$(echo "$result" | cut -d',' -f3)
        echo "  reward=$reward, decisions=$decisions, per_step=$per_step"
        echo "$name,$reward,$decisions,$per_step,$ckpt" >> "$RESULTS_FILE"
    else
        echo "  FAILED"
        echo "$name,FAILED,0,0,$ckpt" >> "$RESULTS_FILE"
    fi
done

# Also eval baselines
echo ""
echo "Evaluating baselines..."

# Zero-hold
result=$(SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON -c "
import os, sys, numpy as np
sys.path.insert(0, '$SCRIPT_DIR'); sys.path.insert(0, '$SCRIPT_DIR/../bus_h2o')
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
import numpy as np
def zero_fn(ev, obs, bid, la): return np.array([-1.0, 0.0], dtype=np.float32)
r, n, t = run_episode(bridge, line_idx_map, bus_idx_map, zero_fn, 'zero')
print(f'{r:.0f},{n},{r/n:.1f}')
bridge.close()
" 2>/dev/null | tail -1)
echo "zero_hold,$result,baseline" >> "$RESULTS_FILE"
echo "  Zero-hold: $result"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
echo ""
printf "%-45s %12s %10s\n" "Name" "SUMO Reward" "Per-Step"
printf "%-45s %12s %10s\n" "----" "-----------" "--------"
sort -t',' -k2 -rn "$RESULTS_FILE" | while IFS=',' read name reward decisions per_step ckpt; do
    [ "$name" = "name" ] && continue
    printf "%-45s %12s %10s\n" "$name" "$reward" "$per_step"
done
echo ""
echo "Reference: ep39=-649K~-683K"
echo "═══════════════════════════════════════════════════════════════"
