#!/bin/bash
# deploy/jtl110gpu.sh — Run on jtl110gpu (port 22916) after SSH-ing in.
#
# Task: SUMO group, seed 123 only (~27h, libsumo serial)
#
# Usage from local:
#   rsync code/data to jtl110gpu (Mutagen handles this)
#   ssh jtl110gpu
#   cd /path/to/sumo-rl/H2Oplus/SimpleSAC
#   bash deploy/jtl110gpu.sh           # launches in background, exits
#   bash deploy/jtl110gpu.sh --check   # dry-run only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/../experiment_output/deploy_logs"
mkdir -p "$LOG_DIR"

# jtl110gpu has shared LSTM-RL env at /home/huiwei/anaconda3 (Python 3.10, full stack)
PYTHON_BIN="${PYTHON_BIN:-/home/huiwei/anaconda3/envs/LSTM-RL/bin/python}"
# eclipse-sumo bundles SUMO inside site-packages
export SUMO_HOME="${SUMO_HOME:-/home/huiwei/anaconda3/envs/LSTM-RL/lib/python3.10/site-packages/sumo}"
# viskit lives at H2Oplus root, not pip-installed
export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"
# Disable wandb (no API key on server; logs are written locally regardless)
export WANDB_MODE=disabled

echo "═══════════════════════════════════════════════════"
echo "jtl110gpu deployment — SUMO seed 123"
echo "═══════════════════════════════════════════════════"
echo "  Python    : $PYTHON_BIN"
echo "  Workdir   : $SCRIPT_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "FAIL: Python not found: $PYTHON_BIN"
    echo "      Override with: PYTHON_BIN=/path/to/python bash deploy/jtl110gpu.sh"
    exit 1
fi

DATA_FILE="$SCRIPT_DIR/../bus_h2o/datasets_v2/merged_all_v2.h5"
CKPT_FILE="$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt"
[ -f "$DATA_FILE" ] || { echo "FAIL: missing $DATA_FILE"; exit 1; }
[ -f "$CKPT_FILE" ] || { echo "FAIL: missing $CKPT_FILE"; exit 1; }
echo "  Dataset   : OK ($(du -h "$DATA_FILE" | cut -f1))"
echo "  Checkpoint: OK ($(du -h "$CKPT_FILE" | cut -f1))"

if [ "${1:-}" = "--check" ]; then
    cd "$SCRIPT_DIR"
    bash run_full_experiments_H2O+.sh --group sumo --seeds "123" \
        --python_bin "$PYTHON_BIN" --dry-run
    exit 0
fi

echo "Launching SUMO seed 123 (~27h)..."
echo "Log: $LOG_DIR/jtl110gpu_sumo_123.log"

cd "$SCRIPT_DIR"
nohup bash run_full_experiments_H2O+.sh \
    --group sumo --seeds "123" \
    --python_bin "$PYTHON_BIN" \
    > "$LOG_DIR/jtl110gpu_sumo_123.log" 2>&1 &

PID=$!
disown $PID
sleep 2
if kill -0 $PID 2>/dev/null; then
    echo "✓ PID=$PID running. Safe to disconnect SSH."
    echo "Watch: ssh jtl110gpu 'tail -f $LOG_DIR/jtl110gpu_sumo_123.log'"
else
    echo "✗ Process died. Check log."
    exit 1
fi
