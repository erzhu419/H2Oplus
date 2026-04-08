#!/bin/bash
# run_experiments.sh — Comprehensive H2O+ ablation experiments
#
# Experiment matrix:
#   1. Pure Offline (ensemble, no online env)
#   2. SUMO Online (H2O+ with SUMO as online env)
#   3. SIM Online (H2O+ with SIM as online env)
#   4. SIM→SUMO transfer (train on SIM, eval on SUMO)
#
# Each group tests: ± IS weighting, ± Cal-QL, ± KL regularization
# All use ensemble Q (E=5) from same offline checkpoint.
#
# Usage:
#   bash run_experiments.sh [--parallel N] [--epochs E] [--dry-run]
#
# Examples:
#   bash run_experiments.sh --parallel 1              # sequential (default)
#   bash run_experiments.sh --parallel 2 --epochs 200 # 2 parallel, 200 epochs
#   bash run_experiments.sh --dry-run                 # just print commands

set -e

# ── Defaults ──
PARALLEL=1
EPOCHS=200
EVAL_PERIOD=10
CHECKPOINT_PERIOD=50
BATCH_SIZE=2048
TRAIN_STEPS=100
ROLLOUT_EVENTS=100
BUFFER_RATIO=1.5
WARMUP_EPISODES=5
WARMUP_COLLECT=20
DRY_RUN=false
SEED=42

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
MAIN="$SCRIPT_DIR/h2o+_bus_main.py"
OFFLINE_CKPT="$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt"
LOG_DIR="$SCRIPT_DIR/../experiment_output/ablation_$(date +%m%d_%H%M)"

mkdir -p "$LOG_DIR"

# ── Common args (shared across ALL experiments) ──
COMMON="--seed=$SEED --device=cpu \
  --use_ensemble_q --ensemble_size 5 \
  --ensemble_ckpt $OFFLINE_CKPT \
  --save_model=True \
  --n_epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --n_train_step_per_epoch=$TRAIN_STEPS \
  --n_rollout_events_per_epoch=$ROLLOUT_EVENTS \
  --eval_period=$EVAL_PERIOD \
  --eval_n_trajs=1 \
  --warmup_episodes=$WARMUP_EPISODES \
  --checkpoint_period=$CHECKPOINT_PERIOD \
  --buffer_ratio=$BUFFER_RATIO \
  --pretrain_steps=0 \
  --nouse_snapshot_reset --nouse_jtt"

# ── Experiment definitions ──
# Format: NAME|EXTRA_FLAGS
declare -a EXPERIMENTS=(
    # ═══════════════════════════════════════════════════
    # Group 1: Pure Offline (no online env, pretrain only)
    # ═══════════════════════════════════════════════════
    # Pure offline = just eval the offline checkpoint (no training loop)
    # We handle this separately via train_offline_ensemble.py

    # ═══════════════════════════════════════════════════
    # Group 2: SUMO Online
    # ═══════════════════════════════════════════════════
    "sumo_baseline|--use_sumo_online --disable_is_weighting --nouse_cal_ql --kl_coeff=0"
    "sumo_calql|--use_sumo_online --disable_is_weighting --use_cal_ql --kl_coeff=0"
    "sumo_is_contrastive|--use_sumo_online --use_contrastive_disc --nouse_cal_ql --kl_coeff=0"
    "sumo_is_calql|--use_sumo_online --use_contrastive_disc --use_cal_ql --kl_coeff=0"
    "sumo_is_calql_kl|--use_sumo_online --use_contrastive_disc --use_cal_ql --kl_coeff=0.5 --warmup_collect_epochs=$WARMUP_COLLECT"

    # ═══════════════════════════════════════════════════
    # Group 3: SIM Online
    # ═══════════════════════════════════════════════════
    "sim_baseline|--nouse_sumo_online --disable_is_weighting --nouse_cal_ql --kl_coeff=0"
    "sim_calql|--nouse_sumo_online --disable_is_weighting --use_cal_ql --kl_coeff=0"
    "sim_is_contrastive|--nouse_sumo_online --use_contrastive_disc --nouse_cal_ql --kl_coeff=0"
    "sim_is_calql|--nouse_sumo_online --use_contrastive_disc --use_cal_ql --kl_coeff=0"
    "sim_is_calql_kl|--nouse_sumo_online --use_contrastive_disc --use_cal_ql --kl_coeff=0.5 --warmup_collect_epochs=$WARMUP_COLLECT"
)

echo "═══════════════════════════════════════════════════════════════"
echo "H2O+ Ablation Experiments"
echo "═══════════════════════════════════════════════════════════════"
echo "Parallel: $PARALLEL"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Log dir: $LOG_DIR"
echo "Offline checkpoint: $OFFLINE_CKPT"
echo ""

# Check offline checkpoint exists
if [ ! -f "$OFFLINE_CKPT" ]; then
    echo "ERROR: Offline checkpoint not found: $OFFLINE_CKPT"
    echo "Run train_offline_ensemble.py first."
    exit 1
fi

# ── Run experiments ──
running=0
total=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name flags <<< "$exp"
    exp_log="$LOG_DIR/${name}.log"
    exp_dir_flag="--name_str=${name}_seed${SEED}"

    # For SUMO online, need env vars
    if [[ "$flags" == *"use_sumo_online"* ]]; then
        env_prefix="SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1"
    else
        env_prefix=""
    fi

    cmd="$env_prefix $PYTHON $MAIN $COMMON $flags $exp_dir_flag"

    total=$((total + 1))
    echo "[$total/${#EXPERIMENTS[@]}] $name"

    if $DRY_RUN; then
        echo "  CMD: $cmd"
        echo ""
        continue
    fi

    echo "  Log: $exp_log"
    eval "nohup $cmd > $exp_log 2>&1 &"
    running=$((running + 1))

    # Throttle to PARALLEL limit
    if [ "$running" -ge "$PARALLEL" ]; then
        echo "  Waiting for a slot ($running/$PARALLEL running)..."
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
done

if ! $DRY_RUN; then
    echo ""
    echo "All $total experiments launched. Waiting for completion..."
    wait
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "All experiments complete!"
    echo "Logs in: $LOG_DIR"
    echo ""
    echo "Next: run eval_all_ablations.sh to evaluate all checkpoints on SUMO"
    echo "═══════════════════════════════════════════════════════════════"
fi
