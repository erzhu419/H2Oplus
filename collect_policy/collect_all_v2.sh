#!/bin/bash
# collect_all_v2.sh — Full-network aligned data collection (v2)
# Uses per-line edge_maps, correct passenger_num_n, raw snapshot storage
# 5 policies × 5 seeds × 10 episodes = 250 episodes total

# set -e  # disabled: individual worker failures should not kill the whole batch
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
WORKER="$SCRIPT_DIR/collect_worker.py"
OUT_DIR="$SCRIPT_DIR/../bus_h2o/datasets_v2"
LOG_DIR="$SCRIPT_DIR/logs_v2"

mkdir -p "$OUT_DIR" "$LOG_DIR"

POLICIES=("zero" "random" "heuristic_best" "heuristic_weak" "sac")
SEEDS=(42 123 456 789 1024)
N_EPISODES=10
MAX_STEPS=40000
MAX_PARALLEL=25

echo "=== Full-Network Aligned Data Collection (v2) ==="
echo "Fixes applied:"
echo "  - Per-line edge_maps for all 12 lines"
echo "  - Correct waiting passenger attribute (passenger_num_n)"
echo "  - Raw snapshot storage for future z rebuild"
echo "  - Fractional route normalization in z"
echo "Policies: ${POLICIES[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Episodes per seed: $N_EPISODES"
echo "Max parallel: $MAX_PARALLEL"
echo "Output dir: $OUT_DIR"
echo ""

running=0
total=0

for policy in "${POLICIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out_file="$OUT_DIR/sumo_${policy}_seed${seed}.h5"
        log_file="$LOG_DIR/${policy}_seed${seed}.log"

        echo "[$(date +%H:%M:%S)] Starting: policy=$policy seed=$seed → $out_file"

        $PYTHON -u "$WORKER" \
            --policy "$policy" \
            --seed "$seed" \
            --n_episodes "$N_EPISODES" \
            --max_steps "$MAX_STEPS" \
            --out "$out_file" \
            > "$log_file" 2>&1 &

        running=$((running + 1))
        total=$((total + 1))

        # Throttle to MAX_PARALLEL
        if [ "$running" -ge "$MAX_PARALLEL" ]; then
            echo "  Waiting for a slot (${running}/${MAX_PARALLEL} running)..."
            wait -n 2>/dev/null || true
            running=$((running - 1))
        fi
    done
done

echo ""
echo "All $total workers launched. Waiting for completion..."
wait
echo ""
echo "=== All done! ==="
echo "Output files:"
ls -lh "$OUT_DIR"/sumo_*.h5 2>/dev/null || echo "  (none found)"
