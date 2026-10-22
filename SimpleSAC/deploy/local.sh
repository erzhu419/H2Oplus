#!/bin/bash
# deploy/local.sh — Run on local WSL machine.
#
# Task: SUMO group, seed 42 only (~27h, libsumo serial)
#
# Other seeds covered by:
#   - jtl110gpu  : seed 123  (./deploy/jtl110gpu.sh)
#   - jtl110gpu2 : seed 456 + all non-SUMO groups (./deploy/jtl110gpu2.sh)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/../experiment_output/deploy_logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python}"
# Apt-installed SUMO at /usr/share/sumo
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
# viskit lives at H2Oplus root, not pip-installed
export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"

# ── Sanity checks ─────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "Local WSL deployment — SUMO seed 42"
echo "═══════════════════════════════════════════════════"
echo "  Python    : $PYTHON_BIN"
echo "  Workdir   : $SCRIPT_DIR"
echo "  Log dir   : $LOG_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "FAIL: Python not found: $PYTHON_BIN"
    exit 1
fi

DATA_FILE="$SCRIPT_DIR/../bus_h2o/datasets_v2/merged_all_v2.h5"
CKPT_FILE="$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt"

if [ ! -f "$DATA_FILE" ]; then
    echo "FAIL: offline dataset missing: $DATA_FILE"
    exit 1
fi
if [ ! -f "$CKPT_FILE" ]; then
    echo "FAIL: ensemble checkpoint missing: $CKPT_FILE"
    echo "      run train_offline_ensemble.py first to produce it"
    exit 1
fi
echo "  Dataset   : $DATA_FILE ($(du -h "$DATA_FILE" | cut -f1))"
echo "  Checkpoint: $CKPT_FILE ($(du -h "$CKPT_FILE" | cut -f1))"

if ! command -v sumo >/dev/null 2>&1; then
    echo "WARN: 'sumo' not in PATH — make sure SUMO_HOME=/usr/share/sumo is correct"
fi

echo "All checks passed."
echo ""

# ── Dry-run preview ───────────────────────────────────────────
if [ "${1:-}" = "--check" ]; then
    cd "$SCRIPT_DIR"
    bash run_full_experiments_H2O+.sh --group sumo --seeds "42" \
        --python_bin "$PYTHON_BIN" --dry-run
    echo ""
    echo "[dry-run] No jobs launched. Re-run without --check to start."
    exit 0
fi

# ── Launch ────────────────────────────────────────────────────
echo "Launching SUMO seed 42 (9 configs sequential, ~27h wall)..."
echo "Output: $LOG_DIR/local_sumo_42.log"
echo "Use 'tail -f $LOG_DIR/local_sumo_42.log' to watch progress."
echo ""

cd "$SCRIPT_DIR"
nohup bash run_full_experiments_H2O+.sh \
    --group sumo --seeds "42" \
    --python_bin "$PYTHON_BIN" \
    > "$LOG_DIR/local_sumo_42.log" 2>&1 &

PID=$!
echo "Launched (PID=$PID). Detaching..."
disown $PID
sleep 2

# Verify it's running
if kill -0 $PID 2>/dev/null; then
    echo "✓ Process running. Job will continue after shell exits."
else
    echo "✗ Process died immediately. Check $LOG_DIR/local_sumo_42.log"
    exit 1
fi
