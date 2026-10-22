#!/bin/bash
# run_full_experiments.sh — Full experiment suite for paper
#
# Groups:
#   A. Offline baselines (no online env): Ensemble, CQL, BC                    (3 configs)
#   B. SUMO Online: vanilla/DARC/DynDARC/Contrastive × ± Q-floor × ± KL        (9 configs)
#   C. SIM  Online: vanilla/DARC/DynDARC/Contrastive × ± Q-floor × ± KL        (9 configs)
#   D. Data ablation: different offline data compositions                      (3 configs)
#
# B and C include three head-to-head baselines requested by paper reviewers:
#   - `darc`:    default TransitionDiscriminator (produces w_IS ≈ 0.28)
#   - `dyn`:     DynamicsDiscriminator (factored DARC with temperature flag)
#   - `rescale`: DARC + explicit sim-reward rescale (cheap alternative to Q-floor)
# Plus the Contrastive variants (`is`, `is_calql`, `full`) for our method.
#
# Default: 5 seeds (42 123 456 789 2024) — override with --seeds for faster runs.
# SUMO experiments forced sequential (libsumo single session).
# SIM experiments parallelized via hardware-aware auto-detect.
#
# Usage:
#   bash run_full_experiments_H2O+.sh [--parallel N|auto] [--epochs E] [--seeds "42 123 456"]
#   bash run_full_experiments_H2O+.sh --group sim --parallel 3       # only SIM group
#   bash run_full_experiments_H2O+.sh --group offline                 # only offline baselines
#   bash run_full_experiments_H2O+.sh --seeds "42 123 456" --dry-run  # 3-seed dry run

set -e

# ── Defaults ──
PARALLEL=auto  # auto = detect from hardware
EPOCHS=200
OFFLINE_STEPS=60000
SEEDS="42 123 456 789 2024"
GROUP="all"  # all, sumo, sim, offline, data_ablation, data_size
DRY_RUN=false
DATA_SIZES="100000 200000 400000 0"  # 0 = full buffer (~675K)
PYTHON_BIN_OVERRIDE=""  # if set (via --python_bin or PYTHON_BIN env), overrides hardcoded conda path

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --python_bin) PYTHON_BIN_OVERRIDE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ═══════════════════════════════════════════════════
# Auto-detect parallelism from hardware
# ═══════════════════════════════════════════════════
detect_parallel() {
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    local mem_gb=$(free -g 2>/dev/null | awk '/Mem:/{print $2}' || echo 16)

    # GPU detection
    local gpu_count=0
    local gpu_mem_gb=0
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        gpu_mem_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}')
    fi

    # Each experiment needs:
    #   CPU mode: ~4GB RAM, ~2 CPU cores (ensemble Q is compute-heavy)
    #   GPU mode: ~2GB VRAM per experiment (batch_size=2048, E=5 ensemble)
    local mem_per_job=4  # GB RAM
    local cores_per_job=2
    local vram_per_job=2  # GB VRAM

    # Constraints
    local max_by_ram=$((mem_gb / mem_per_job))
    local max_by_cpu=$((cpu_cores / cores_per_job))

    if [ "$gpu_count" -gt 0 ] && [ "$gpu_mem_gb" -gt 0 ]; then
        # GPU mode: limited by VRAM per GPU
        # Too many concurrent GPU jobs cause memory thrashing → actually slower
        # Sweet spot: 1-2 jobs per GPU for our model size
        local max_by_gpu=$((gpu_count * (gpu_mem_gb / vram_per_job)))
        # Cap at 2 per GPU to avoid contention
        local gpu_cap=$((gpu_count * 2))
        max_by_gpu=$((max_by_gpu < gpu_cap ? max_by_gpu : gpu_cap))

        local par=$((max_by_ram < max_by_cpu ? max_by_ram : max_by_cpu))
        par=$((par < max_by_gpu ? par : max_by_gpu))
    else
        # CPU only
        local par=$((max_by_ram < max_by_cpu ? max_by_ram : max_by_cpu))
    fi

    # Global caps
    [ "$par" -lt 1 ] && par=1
    [ "$par" -gt 8 ] && par=8  # diminishing returns beyond 8

    echo "Hardware detection:" >&2
    echo "  CPU cores: $cpu_cores" >&2
    echo "  RAM: ${mem_gb}GB" >&2
    if [ "$gpu_count" -gt 0 ]; then
        echo "  GPUs: $gpu_count × ${gpu_mem_gb}GB VRAM" >&2
    else
        echo "  GPUs: none (CPU mode)" >&2
    fi
    echo "  Per-job: ~${mem_per_job}GB RAM, ~${cores_per_job} cores" >&2
    echo "  Recommended parallel: $par" >&2

    echo "$par"
}

if [ "$PARALLEL" = "auto" ]; then
    PARALLEL=$(detect_parallel)
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Python binary resolution: --python_bin > PYTHON_BIN env > hardcoded default
if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    PYTHON="$PYTHON_BIN_OVERRIDE"
elif [ -n "$PYTHON_BIN" ]; then
    PYTHON="$PYTHON_BIN"
else
    PYTHON="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python binary not found: $PYTHON"
    echo "Set via --python_bin /path/to/python or PYTHON_BIN env var"
    exit 1
fi
echo "Using Python: $PYTHON"
MAIN="$SCRIPT_DIR/h2o+_bus_main.py"
OFFLINE_TRAIN="$SCRIPT_DIR/train_offline_ensemble.py"
OFFLINE_CKPT="$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt"
TIMESTAMP=$(date +%m%d_%H%M)
LOG_DIR="$SCRIPT_DIR/../experiment_output/paper_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# Auto-detect device: use cuda if available, else cpu
DEVICE="cpu"
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        DEVICE="cuda"
    fi
fi

# Common H2O+ args
COMMON="--device=$DEVICE --use_ensemble_q --ensemble_size 5 \
  --ensemble_ckpt $OFFLINE_CKPT --save_model=True \
  --n_epochs=$EPOCHS --batch_size=2048 --n_train_step_per_epoch=100 \
  --n_rollout_events_per_epoch=100 --eval_period=10 --eval_n_trajs=1 \
  --warmup_episodes=5 --checkpoint_period=50 --buffer_ratio=1.5 \
  --pretrain_steps=0 --nouse_snapshot_reset --nouse_jtt"

running=0
total=0
gpu_slot=0  # round-robin GPU assignment

launch() {
    local name="$1"
    local cmd="$2"
    local log="$LOG_DIR/${name}.log"
    total=$((total + 1))

    # GPU round-robin: assign CUDA_VISIBLE_DEVICES if multiple GPUs
    local gpu_prefix=""
    if [ "$GPU_COUNT" -gt 1 ]; then
        gpu_prefix="CUDA_VISIBLE_DEVICES=$gpu_slot"
        gpu_slot=$(( (gpu_slot + 1) % GPU_COUNT ))
    fi

    echo "[$total] $name${gpu_prefix:+ (GPU $((gpu_slot == 0 ? GPU_COUNT-1 : gpu_slot-1)))}"
    if $DRY_RUN; then
        echo "  CMD: ${gpu_prefix:+$gpu_prefix }$cmd" | head -c 250
        echo "..."
        return
    fi
    echo "  Log: $log"
    # Use `env` to handle inline VAR=val pairs before the binary
    # (nohup VAR=val cmd does NOT parse env assignments; only env or bash -c does)
    eval "nohup env ${gpu_prefix:+$gpu_prefix }$cmd > $log 2>&1 &"
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
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE"

        # A2. Ensemble + CQL
        launch "offline_ensemble_cql_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE --use_cql --cql_alpha=5.0"

        # A3. Behavior Cloning (AWR with high temperature = pure imitation)
        launch "offline_bc_s${seed}" \
            "$PYTHON $SCRIPT_DIR/train_offline_only.py --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE"
    done
fi

# ═══════════════════════════════════════════════════
# B. SUMO Online (sequential only — libsumo constraint)
# ═══════════════════════════════════════════════════
if [[ "$GROUP" == "all" || "$GROUP" == "sumo" ]]; then
    # Force sequential for SUMO — libsumo only allows one session
    SAVED_PARALLEL=$PARALLEL
    PARALLEL=1
    echo ""
    echo "── Group B: SUMO Online (forced sequential — libsumo constraint) ──"
    for seed in $SEEDS; do
        # B1. SUMO baseline (no IS, no CalQL)
        launch "sumo_baseline_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --disable_is_weighting --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_baseline_s${seed}"

        # B2. SUMO + CalQL (no IS)
        launch "sumo_calql_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --disable_is_weighting --use_cal_ql \
            --kl_coeff=0 --name_str=sumo_calql_s${seed}"

        # B3a. SUMO + DARC IS (TransitionDiscriminator, no CalQL) — reviewer-demanded baseline
        launch "sumo_darc_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_darc_s${seed}"

        # B3b. SUMO + DARC IS + Q-floor — head-to-head DARC vs Contrastive with Q-floor
        launch "sumo_darc_calql_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_cal_ql \
            --kl_coeff=0 --name_str=sumo_darc_calql_s${seed}"

        # B3c. SUMO + DynamicsDiscriminator (factored DARC, temp=1.0) — tuned DARC variant
        launch "sumo_dyn_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_dynamics_disc --dynamics_disc_temp=1.0 --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_dyn_s${seed}"

        # B3d. SUMO + DARC IS + sim-reward rescale (ρ=5.0) — cheap alternative to Q-floor
        launch "sumo_rescale_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --sim_reward_rescale=5.0 --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_rescale_s${seed}"

        # B4. SUMO + Contrastive IS (no CalQL)
        launch "sumo_is_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_contrastive_disc --nouse_cal_ql \
            --kl_coeff=0 --name_str=sumo_is_s${seed}"

        # B5. SUMO + Contrastive IS + CalQL
        launch "sumo_is_calql_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0 --name_str=sumo_is_calql_s${seed}"

        # B6. SUMO + Contrastive IS + CalQL + KL (full)
        launch "sumo_full_s${seed}" \
            "WANDB_MODE=disabled LIBSUMO_AS_TRACI=1 $PYTHON $MAIN $COMMON \
            --seed=$seed --use_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0.5 --warmup_collect_epochs=20 --name_str=sumo_full_s${seed}"
    done
    # Wait for all SUMO jobs before restoring parallelism
    if ! $DRY_RUN; then wait; fi
    running=0
    PARALLEL=$SAVED_PARALLEL
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
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --disable_is_weighting --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_baseline_s${seed}"

        # C2. SIM + CalQL (no IS)
        launch "sim_calql_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --disable_is_weighting --use_cal_ql \
            --kl_coeff=0 --name_str=sim_calql_s${seed}"

        # C3a. SIM + DARC IS (TransitionDiscriminator, no CalQL) — reviewer-demanded baseline
        launch "sim_darc_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_darc_s${seed}"

        # C3b. SIM + DARC IS + Q-floor — head-to-head vs Contrastive+Q-floor
        launch "sim_darc_calql_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_cal_ql \
            --kl_coeff=0 --name_str=sim_darc_calql_s${seed}"

        # C3c. SIM + DynamicsDiscriminator (factored DARC, temp=1.0)
        launch "sim_dyn_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_dynamics_disc --dynamics_disc_temp=1.0 --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_dyn_s${seed}"

        # C3d. SIM + DynamicsDiscriminator (factored DARC, temp=0.5) — DARC tuning sweep
        launch "sim_dyn_t05_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_dynamics_disc --dynamics_disc_temp=0.5 --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_dyn_t05_s${seed}"

        # C3e. SIM + DynamicsDiscriminator (factored DARC, temp=2.0) — DARC tuning sweep
        launch "sim_dyn_t20_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_dynamics_disc --dynamics_disc_temp=2.0 --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_dyn_t20_s${seed}"

        # C3f. SIM + DARC IS + sim-reward rescale (ρ=5.0) — cheap alternative to Q-floor
        launch "sim_rescale_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --sim_reward_rescale=5.0 --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_rescale_s${seed}"

        # C4. SIM + Contrastive IS (no CalQL)
        launch "sim_is_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_contrastive_disc --nouse_cal_ql \
            --kl_coeff=0 --name_str=sim_is_s${seed}"

        # C5. SIM + Contrastive IS + CalQL
        launch "sim_is_calql_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
            --seed=$seed --nouse_sumo_online --use_contrastive_disc --use_cal_ql \
            --kl_coeff=0 --name_str=sim_is_calql_s${seed}"

        # C6. SIM + Contrastive IS + CalQL + KL (full)
        launch "sim_full_s${seed}" \
            "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
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
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE \
            --dataset_dir=$DS_DIR --dataset_glob='sumo_sac_*.h5'"

        # D2. Heuristic-only data
        launch "data_heuristic_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE \
            --dataset_dir=$DS_DIR --dataset_glob='sumo_heuristic_best_*.h5'"

        # D3. Zero+Random only (worst quality)
        launch "data_weak_s${seed}" \
            "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE \
            --dataset_dir=$DS_DIR --dataset_glob='sumo_zero_*.h5,sumo_random_*.h5'"
    done
fi

# ═══════════════════════════════════════════════════
# E. Data-size scaling: Pure offline vs H2O+ at varying offline buffer sizes
# ═══════════════════════════════════════════════════
if [[ "$GROUP" == "all" || "$GROUP" == "data_size" ]]; then
    echo ""
    echo "── Group E: Data-size scaling (Pure-offline vs H2O+) ──"
    for sz in $DATA_SIZES; do
        sz_tag=$([ "$sz" -eq 0 ] && echo "full" || echo "${sz}")
        for seed in $SEEDS; do
            # E1. Pure offline (Cal-QL via use_cql) at this size
            launch "size_offline_${sz_tag}_s${seed}" \
                "$PYTHON $OFFLINE_TRAIN --n_steps=$OFFLINE_STEPS --seed=$seed --device=$DEVICE \
                --use_cql --cql_alpha=5.0 --max_offline_samples=$sz \
                --name_str=size_offline_${sz_tag}_s${seed}"

            # E2. H2O+ SIM-online (full method) at this size
            launch "size_h2oplus_${sz_tag}_s${seed}" \
                "WANDB_MODE=disabled $PYTHON $MAIN $COMMON \
                --seed=$seed --nouse_sumo_online --use_contrastive_disc --use_cal_ql \
                --kl_coeff=0.5 --warmup_collect_epochs=20 \
                --max_offline_samples=$sz \
                --name_str=size_h2oplus_${sz_tag}_s${seed}"
        done
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
