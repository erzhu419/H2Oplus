#!/bin/bash
# recollect_sac.sh — Re-collect SAC policy data with norm bug fix (update=False)
# Only re-collects the "sac" policy, not other policies (they don't use norm)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
WORKER="$SCRIPT_DIR/collect_worker.py"
OUT_DIR="$SCRIPT_DIR/../bus_h2o/datasets_v2"
LOG_DIR="$SCRIPT_DIR/logs_v2"

mkdir -p "$OUT_DIR" "$LOG_DIR"

SEEDS=(42 123 456 789 1024)
N_EPISODES=10
MAX_STEPS=40000

echo "=== Re-collecting SAC policy data (norm bug fix) ==="
echo "Fix: state_norm(x, update=False) instead of state_norm(x)"
echo ""

for seed in "${SEEDS[@]}"; do
    out_file="$OUT_DIR/sumo_sac_seed${seed}.h5"
    log_file="$LOG_DIR/sac_seed${seed}_fixed.log"

    # Backup old file
    if [ -f "$out_file" ]; then
        mv "$out_file" "${out_file}.bak_norm_bug"
        echo "  Backed up old: $out_file → ${out_file}.bak_norm_bug"
    fi

    echo "[$(date +%H:%M:%S)] Collecting: sac seed=$seed → $out_file"

    LIBSUMO_AS_TRACI=1 $PYTHON -u "$WORKER" \
        --policy sac \
        --seed "$seed" \
        --n_episodes "$N_EPISODES" \
        --max_steps "$MAX_STEPS" \
        --out "$out_file" \
        > "$log_file" 2>&1

    echo "  Done: $(wc -l < "$log_file") log lines"
done

echo ""
echo "=== All seeds done! ==="
echo "New files:"
ls -lh "$OUT_DIR"/sumo_sac_seed*.h5 2>/dev/null
echo ""
echo "Next step: re-merge with merge_v2_lazy.py"
