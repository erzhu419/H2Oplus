#!/bin/bash
# deploy/jtl110gpu2.sh — Run on jtl110gpu2 (port 23035) after SSH-ing in.
#
# Tasks:
#   1. SUMO group, seed 456 (in background, ~27h, libsumo serial)
#   2. All non-SUMO groups, all 5 seeds, par=6 (foreground, ~10h total)
#
# Resource budget on jtl110gpu2 (20GB GPU + sufficient RAM):
#   SUMO bg job        : 2 cores, ~6 GB RAM,  ~2 GB VRAM
#   6 × non-SUMO jobs  : 12 cores, ~24 GB RAM, ~12 GB VRAM
#   Total              : 14 cores, ~30 GB RAM, ~14 GB VRAM (under 20 GB GPU cap)
#
# Usage:
#   ssh jtl110gpu2
#   cd /path/to/sumo-rl/H2Oplus/SimpleSAC
#   bash deploy/jtl110gpu2.sh             # full launch
#   bash deploy/jtl110gpu2.sh --check     # dry-run only
#   bash deploy/jtl110gpu2.sh --no-sumo   # skip SUMO bg, only non-SUMO

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/../experiment_output/deploy_logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/erzhu419/miniconda3/envs/csbapr/bin/python}"
NON_SUMO_PARALLEL="${NON_SUMO_PARALLEL:-6}"   # adjust if VRAM tighter
# eclipse-sumo bundles SUMO inside site-packages (after pip install eclipse-sumo)
export SUMO_HOME="${SUMO_HOME:-/home/erzhu419/miniconda3/envs/csbapr/lib/python3.11/site-packages/sumo}"
# viskit lives at H2Oplus root, not pip-installed
export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"
# Disable wandb (no API key on server; logs are written locally regardless)
export WANDB_MODE=disabled

MODE="${1:-full}"   # full, --check, --no-sumo

echo "═══════════════════════════════════════════════════"
echo "jtl110gpu2 deployment"
echo "═══════════════════════════════════════════════════"
echo "  Python              : $PYTHON_BIN"
echo "  Workdir             : $SCRIPT_DIR"
echo "  Non-SUMO parallel   : $NON_SUMO_PARALLEL"
echo "  Mode                : $MODE"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "FAIL: Python not found: $PYTHON_BIN"
    exit 1
fi

DATA_FILE="$SCRIPT_DIR/../bus_h2o/datasets_v2/merged_all_v2.h5"
CKPT_FILE="$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt"
[ -f "$DATA_FILE" ] || { echo "FAIL: missing $DATA_FILE"; exit 1; }
[ -f "$CKPT_FILE" ] || { echo "FAIL: missing $CKPT_FILE"; exit 1; }
echo "  Dataset             : OK"
echo "  Checkpoint          : OK"

# GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | head -1)
    echo "  GPU                 : ${GPU_MEM}MB total, ${GPU_FREE}MB free"
    if [ "$GPU_FREE" -lt 14000 ]; then
        echo "  WARN: Free VRAM <14GB. Consider reducing NON_SUMO_PARALLEL."
    fi
fi

# ── Dry-run ───────────────────────────────────────────────────
if [ "$MODE" = "--check" ]; then
    cd "$SCRIPT_DIR"
    echo ""
    echo "── Would launch SUMO bg ──"
    bash run_full_experiments_H2O+.sh --group sumo --seeds "456" \
        --python_bin "$PYTHON_BIN" --dry-run | head -15
    echo ""
    echo "── Would launch non-SUMO groups ──"
    for grp in offline sim data_ablation data_size; do
        echo "[$grp]"
        bash run_full_experiments_H2O+.sh --group $grp --parallel $NON_SUMO_PARALLEL \
            --python_bin "$PYTHON_BIN" --dry-run | grep -cE "^\[" \
            | awk '{print "  count: "$1}'
    done
    exit 0
fi

cd "$SCRIPT_DIR"

# ── Launch SUMO bg (unless --no-sumo) ────────────────────────
if [ "$MODE" != "--no-sumo" ]; then
    echo ""
    echo "[1/2] Launching SUMO seed 456 in background..."
    nohup bash run_full_experiments_H2O+.sh \
        --group sumo --seeds "456" \
        --python_bin "$PYTHON_BIN" \
        > "$LOG_DIR/jtl110gpu2_sumo_456.log" 2>&1 &
    SUMO_PID=$!
    disown $SUMO_PID
    sleep 3
    if ! kill -0 $SUMO_PID 2>/dev/null; then
        echo "✗ SUMO bg died. Check log."
        exit 1
    fi
    echo "✓ SUMO bg PID=$SUMO_PID running. Log: $LOG_DIR/jtl110gpu2_sumo_456.log"
fi

# ── Launch non-SUMO groups foreground ────────────────────────
echo ""
echo "[2/2] Running non-SUMO groups (5 seeds, par=$NON_SUMO_PARALLEL)..."
echo "Estimated wall: ~10h"
echo ""

for grp in offline sim data_ablation data_size; do
    echo "── Group: $grp ──"
    LOG="$LOG_DIR/jtl110gpu2_${grp}.log"
    bash run_full_experiments_H2O+.sh \
        --group $grp \
        --parallel $NON_SUMO_PARALLEL \
        --python_bin "$PYTHON_BIN" \
        2>&1 | tee "$LOG"
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "All non-SUMO groups complete."
if [ "$MODE" != "--no-sumo" ]; then
    if kill -0 $SUMO_PID 2>/dev/null; then
        echo "SUMO bg (PID=$SUMO_PID) still running. Check:"
        echo "  tail -f $LOG_DIR/jtl110gpu2_sumo_456.log"
    else
        echo "SUMO bg complete."
    fi
fi
echo "═══════════════════════════════════════════════════"
