#!/bin/bash
# run_full_experiments.sh — Full experiment suite for paper
#
# Groups:
#   A. Offline baselines (no online env): Ensemble, CQL, IQL, BC
#   B. SUMO Online (H2O+ with SUMO): ± IS/CalQL/KL
#   C. SIM Online (H2O+ with SIM): ± IS/CalQL/KL
#   D. Data ablation: different offline data compositions
#
# All × 3 seeds (42, 123, 456)
# SUMO experiments must be sequential (libsumo single session)
# SIM experiments can be parallelized
#
# Usage:
#   bash run_full_experiments.sh [--parallel N] [--epochs E] [--seeds "42 123 456"]
#   bash run_full_experiments.sh --group sim --parallel 3   # only SIM group
#   bash run_full_experiments.sh --group offline             # only offline baselines
#   bash run_full_experiments.sh --dry-run                   # just print

set -e

# ── Defaults ──
PARALLEL=1
EPOCHS=200
OFFLINE_STEPS=60000
SEEDS="42 123 456"
GROUP="all"  # all, sumo, sim, offline, data_ablation
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
MAIN="$SCRIPT_DIR/h2o+_bus_main.py"
OFFLINE_TRAIN="$SCRIPT_DIR/train_offline_ensemble.py"
OFFLINE_CKPT="$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt"
TIMESTAMP=$(date +%m%d_%H%M)
LOG_DIR="$SCRIPT_DIR/../experiment_output/paper_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# Common H2O+ args
COMMON="--device=cpu --use_ensemble_q --ensemble_size 5 \
  --ensemble_ckpt $OFFLINE_CKPT --save_model=True \
  --n_epochs=$EPOCHS --batch_size=2048 --n_train_step_per_epoch=100 \
  --n_rollout_events_per_epoch=100 --eval_period=10 --eval_n_trajs=1 \
  --warmup_episodes=5 --checkpoint_period=50 --buffer_ratio=1.5 \
  --pretrain_steps=0 --nouse_snapshot_reset --nouse_jtt"

running=0
total=0

launch() {
    local name="$1"
    local cmd="$2"
    local log="$LOG_DIR/${name}.log"
    total=$((total + 1))
    echo "[$total] $name"
    if $DRY_RUN; then
        echo "  CMD: $cmd" | head -c 200
        echo "..."
        return
    fi
    echo "  Log: $log"
    eval "nohup $cmd > $log 2>&1 &"
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then
        wait -n 2>/dev/null || true
        running=$((running - 1))
    fi
}

echo "═══════════════════════════════════════════════════"
echo "Full Paper Experiment Suite"
echo "═══════════════════════════════════════════════════"
echo "Group: $GROUP | Parallel: $PARALLEL | Epochs: $EPOCHS"
echo "Seeds: $SEEDS | Log: $LOG_DIR"
echo ""

# ═══════════════════════════════════════════════════
# A. Offline Baselines
# ═══════════════════════════════════════════════════
if [[ "$GROUP" == "all" || "$GROUP" == "offline" ]]; then
    echo "── Group A: Offline Baselines ──"
    for seed in $SEEDS; do
        # A1. Ensemble offline (RE-SAC style) — already have seed=42
        launch "offline_ensemble_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=cpu"

        # A2. Ensemble + CQL
        launch "offline_ensemble_cql_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=cpu --use_cql --cql_alpha=5.0"

        # A3. Behavior Cloning (AWR with high temperature = pure imitation)
        launch "offline_bc_s${seed}" \
            "$PYTHON $SCRIPT_DIR/train_offline_only.py --n_steps=$OFFLINE_STEPS --seed=$seed --device=cpu"
    done
fi

# ═══════════════════════════════════════════════════
# B. SUMO Online (sequential only — libsumo constraint)
# ═══════════════════════════════════════════════════
if [[ "$GROUP" == "all" || "$GROUP" == "sumo" ]]; then
    echo ""
    echo "── Group B: SUMO Online (sequential) ──"
    for seed in $SEEDS; do
        # B1. SUMO baseline (no IS, no CalQL)
        launch "sumo_baseline_s${seed}" \
            "SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --disable_is_weighting --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_baseline_s${seed}"

        # B2. SUMO + CalQL
        launch "sumo_calql_s${seed}" \
            "SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --disable_is_weighting --use_cal_ql \
            --kl_coeff=0 --name_str=sumo_calql_s${seed}"

        # B3. SUMO + Contrastive IS
        launch "sumo_is_s${seed}" \
            "SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_contrastive_disc --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_is_s${seed}"

        # B4. SUMO + IS + CalQL
        launch "sumo_is_calql_s${seed}" \
            "SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0 --name_str=sumo_is_calql_s${seed}"

        # B5. SUMO + IS + CalQL + KL (full)
        launch "sumo_full_s${seed}" \
            "SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0.5 --warmup_collect_epochs=20 --name_str=sumo_full_s${seed}"
    done
fi

# ═══════════════════════════════════════════════════
# C. SIM Online (parallelizable)
# ═══════════════════════════════════════════════════
if [[ "$GROUP" == "all" || "$GROUP" == "sim" ]]; then
    echo ""
    echo "── Group C: SIM Online ──"
    for seed in $SEEDS; do
        # C1. SIM baseline
        launch "sim_baseline_s${seed}" \
            "$PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --disable_is_weighting --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_baseline_s${seed}"

        # C2. SIM + CalQL
        launch "sim_calql_s${seed}" \
            "$PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --disable_is_weighting --use_cal_ql \
            --kl_coeff=0 --name_str=sim_calql_s${seed}"

        # C3. SIM + Contrastive IS
        launch "sim_is_s${seed}" \
            "$PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_contrastive_disc --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_is_s${seed}"

        # C4. SIM + IS + CalQL
        launch "sim_is_calql_s${seed}" \
            "$PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0 --name_str=sim_is_calql_s${seed}"

        # C5. SIM + IS + CalQL + KL (full)
        launch "sim_full_s${seed}" \
            "$PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0.5 --warmup_collect_epochs=20 --name_str=sim_full_s${seed}"
    done
fi

# ═══════════════════════════════════════════════════
# D. Data Ablation (offline, different compositions)
# ═══════════════════════════════════════════════════
if [[ "$GROUP" == "all" || "$GROUP" == "data_ablation" ]]; then
    echo ""
    echo "── Group D: Data Ablation ──"
    DS_DIR="$SCRIPT_DIR/../bus_h2o/datasets_v2"
    for seed in $SEEDS; do
        # D1. SAC-only data (expert)
        launch "data_sac_only_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=cpu \
            --dataset_dir=$DS_DIR --dataset_glob='sumo_sac_*.h5'"

        # D2. Heuristic-only data
        launch "data_heuristic_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=cpu \
            --dataset_dir=$DS_DIR --dataset_glob='sumo_heuristic_best_*.h5'"

        # D3. Zero+Random only (worst quality)
        launch "data_weak_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=cpu \
            --dataset_dir=$DS_DIR --dataset_glob='sumo_zero_*.h5,sumo_random_*.h5'"
    done
fi

if ! $DRY_RUN; then
    echo ""
    echo "All $total experiments launched. Waiting..."
    wait
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "Done! Results in: $LOG_DIR"
    echo "Next: bash eval_all_ablations.sh $LOG_DIR"
    echo "═══════════════════════════════════════════════════"
fi
