#!/bin/bash
# collect_all.sh — 并行启动多策略 SUMO 数据采集
# 5 策略 × 5 seeds × 10 episodes each = 250 episodes total
# 最多 10 个并行 worker

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
WORKER="$SCRIPT_DIR/collect_worker.py"
OUT_DIR="$SCRIPT_DIR/../bus_h2o/datasets"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

POLICIES=("zero" "random" "heuristic_best" "heuristic_weak" "sac")
SEEDS=(42 123 456 789 1024)
N_EPISODES=10
MAX_STEPS=40000
MAX_PARALLEL=10

echo "=== Multi-Policy SUMO Offline Data Collection ==="
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
