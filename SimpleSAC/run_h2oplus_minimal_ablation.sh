#!/usr/bin/env bash
# Minimal H2O+ improvement screen for scarce GPU settings.
#
# Stage 1 ("proxy") runs a sequential additive ladder with one seed and short
# training. Use the proxy result only to eliminate weak ideas.
# Stage 2 ("promote") is intentionally manual: rerun only the best 1-2 arms
# with more seeds/epochs by passing --arms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON_BIN:-/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python}"
MAIN="$SCRIPT_DIR/h2o+_bus_main.py"
OFFLINE_CKPT="${OFFLINE_CKPT:-$SCRIPT_DIR/../experiment_output/offline_ensemble/offline_ensemble_final.pt}"

STAGE="proxy"
SEEDS="42"
EPOCHS=""
TRAIN_STEPS=""
ROLLOUT_EVENTS=50
EVAL_PERIOD=10
DEVICE="cuda"
ONLINE_ENV="sim"
ARMS="base p1_ratio p2_adaptive p3_jtt p4_floor_ess"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --train_steps) TRAIN_STEPS="$2"; shift 2 ;;
        --rollout_events) ROLLOUT_EVENTS="$2"; shift 2 ;;
        --eval_period) EVAL_PERIOD="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --online_env) ONLINE_ENV="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --python_bin) PYTHON="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$STAGE" == "promote" ]]; then
    : "${EPOCHS:=200}"
    : "${TRAIN_STEPS:=200}"
else
    : "${EPOCHS:=80}"
    : "${TRAIN_STEPS:=100}"
fi

COMMON=(
    --device="$DEVICE"
    --use_ensemble_q
    --ensemble_ckpt="$OFFLINE_CKPT"
    --pretrain_steps=0
    --n_epochs="$EPOCHS"
    --n_train_step_per_epoch="$TRAIN_STEPS"
    --n_rollout_events_per_epoch="$ROLLOUT_EVENTS"
    --eval_period="$EVAL_PERIOD"
    --eval_n_trajs=2
    --checkpoint_period=20
    --warmup_episodes=20
    --jtt_warmup_epochs=20
)

if [[ "$ONLINE_ENV" == "sumo" ]]; then
    COMMON+=(--use_sumo_online)
elif [[ "$ONLINE_ENV" != "sim" ]]; then
    echo "Unknown --online_env: $ONLINE_ENV (expected sim or sumo)" >&2
    exit 1
fi

arm_flags() {
    case "$1" in
        base)
            echo "--use_contrastive_disc --contrastive_ratio_mode=cosine --nouse_cal_ql --name_str=h2op_min_base"
            ;;
        p1_ratio)
            echo "--use_contrastive_disc --contrastive_ratio_mode=nce --nouse_cal_ql --name_str=h2op_min_p1_ratio"
            ;;
        p2_adaptive)
            echo "--use_contrastive_disc --contrastive_ratio_mode=nce --adaptive_sim_ratio --nouse_cal_ql --name_str=h2op_min_p2_adaptive"
            ;;
        p3_jtt)
            echo "--use_contrastive_disc --contrastive_ratio_mode=nce --adaptive_sim_ratio --use_jtt --nouse_cal_ql --name_str=h2op_min_p3_jtt"
            ;;
        p4_floor_ess)
            echo "--use_contrastive_disc --contrastive_ratio_mode=nce --adaptive_sim_ratio --use_jtt --use_cal_ql --cal_ql_mode=batch_quantile --ess_sim_loss_scale --name_str=h2op_min_p4_floor_ess"
            ;;
        clean_rlpd_guard)
            echo "--disable_is_weighting --nouse_cal_ql --name_str=h2op_min_clean_rlpd_guard"
            ;;
        *)
            echo "Unknown arm: $1" >&2
            return 1
            ;;
    esac
}

echo "Python: $PYTHON"
echo "Stage: $STAGE"
echo "Online env: $ONLINE_ENV"
echo "Seeds: $SEEDS"
echo "Arms: $ARMS"

for seed in $SEEDS; do
    for arm in $ARMS; do
        read -r -a EXTRA <<< "$(arm_flags "$arm")"
        CMD=("$PYTHON" "$MAIN" "${COMMON[@]}" --seed="$seed" "${EXTRA[@]}")
        echo
        echo "==> seed=$seed arm=$arm"
        printf ' %q' "${CMD[@]}"
        echo
        if [[ "$DRY_RUN" != "true" ]]; then
            "${CMD[@]}"
        fi
    done
done
