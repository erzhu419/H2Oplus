#!/usr/bin/env bash
# run_multiseed_eval.sh
# =====================
# Multi-seed × multi-OD evaluation harness for the H2O+ paper (Issue 4).
#
# Loops over 6 representative methods × 3 SUMO seeds × 3 OD scales = 54 runs.
# Each combo is a single libsumo process (~5-15 min) producing one JSON
# under experiment_output/multiseed_eval/{method}_sumo{S}_od{D}.json.
#
# OD scales: {0.6, 0.8, 1.0}. SUMO --scale > 1.0 duplicates passenger persons
# (e.g. workday6_7468.0.1) but the env's passenger_obj_dic is built from the
# original route XML, so the duplicates KeyError on lookup. Up-scaling demand
# requires re-generating the passenger route file, which is out of scope. We
# instead sweep DOWN from 1.0 — equally informative for stress sensitivity.
#
# This script is INVOKED PER-(method,seed,scale) — that is the unit the
# scheduler dispatches. It selects the right eval entrypoint per method:
#   * h2oplus_full / h2oplus_darc_calql / pure_online_sac → eval_with_metrics.py
#   * bc_full                                              → eval_with_metrics.py (BC ckpt)
#   * daganzo                                              → eval_daganzo.py
#   * zero_hold                                            → eval_with_metrics.py --zero_hold
#
# Usage:
#   bash run_multiseed_eval.sh <method> <sumo_seed> <od_scale>
# Example:
#   bash run_multiseed_eval.sh h2oplus_full 1001 1.0

set -euo pipefail

METHOD="${1:?usage: run_multiseed_eval.sh <method> <sumo_seed> <od_scale>}"
SUMO_SEED="${2:?seed required}"
OD_SCALE="${3:?od_scale required}"

# ── Resolve paths ────────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H2O_ROOT="$(cd "$HERE/.." && pwd)"
OUT_DIR="$H2O_ROOT/experiment_output/multiseed_eval"
mkdir -p "$OUT_DIR"

OUT_JSON="$OUT_DIR/${METHOD}_sumo${SUMO_SEED}_od${OD_SCALE}.json"

latest_checkpoint() {
  local dir_pattern="$1"
  local ckpt_name="${2:-checkpoint_best.pt}"
  find "$H2O_ROOT/experiment_output" -maxdepth 2 \
    -path "$H2O_ROOT/experiment_output/${dir_pattern}/${ckpt_name}" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

# ── Method → checkpoint table (canonical reps; see issue4_multiseed_summary.md) ──
REQUIRE_CKPT=0
case "$METHOD" in
  h2oplus_full)
    # SUMO-online H2O+ full (DARC + CalQL + IS + dyn-rescale + dyn-classifier)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_26-04-30-17-43-02/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  h2oplus_darc_calql)
    # SUMO-online H2O+ DARC + CalQL ablation
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_26-04-30-13-39-52/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  pure_online_sac)
    # Pure online SAC trained on simulator only
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_26-05-01-00-02-28/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  bc_full)
    # Behaviour cloning on the merged offline data
    CKPT="$H2O_ROOT/experiment_output/bc_full_seed42/bc_final.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  daganzo)
    CKPT=""  # not used
    SCRIPT="eval_daganzo.py"
    EXTRA_ARGS=(--alpha 0.6)
    ;;
  daganzo_a03)
    CKPT=""  # not used
    SCRIPT="eval_daganzo.py"
    EXTRA_ARGS=(--alpha 0.3)
    ;;
  daganzo_a04)
    CKPT=""  # not used
    SCRIPT="eval_daganzo.py"
    EXTRA_ARGS=(--alpha 0.4)
    ;;
  zero_hold)
    CKPT=""  # not used
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=(--zero_hold)
    ;;
  ep39)
    CKPT=""  # legacy ckpt loaded internally via --ep39 flag
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=(--ep39)
    ;;
  pure_online_sac_1000ep_s42)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed42_pure_online_1000ep_s42/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  pure_online_sac_1000ep_s123)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed123_pure_online_1000ep_s123/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  pure_online_sac_1000ep_s789)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_pure_online_1000ep_s789/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  sim_contrastive_s42)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed42_r2_sim_contrastive_s42/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  sim_contrastive_s123)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed123_r2_sim_contrastive_s123/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  sim_contrastive_s789)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_r2_sim_contrastive_s789/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"
    EXTRA_ARGS=()
    ;;
  # ── R3 #10 baselines (3 seeds each) ─────────────────────────────────────────
  bc_ep39_s42)
    CKPT="$H2O_ROOT/experiment_output/bc_ep39_seed42/bc_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  bc_ep39_s123)
    CKPT="$H2O_ROOT/experiment_output/bc_ep39_seed123/bc_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  bc_ep39_s789)
    CKPT="$H2O_ROOT/experiment_output/bc_ep39_seed789/bc_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  iql_s42)
    CKPT="$H2O_ROOT/experiment_output/iql_seed42/iql_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  iql_s123)
    CKPT="$H2O_ROOT/experiment_output/iql_seed123/iql_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  iql_s789)
    CKPT="$H2O_ROOT/experiment_output/iql_seed789/iql_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  awac_s42)
    CKPT="$H2O_ROOT/experiment_output/awac_seed42/awac_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  awac_s123)
    CKPT="$H2O_ROOT/experiment_output/awac_seed123/awac_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  awac_s789)
    CKPT="$H2O_ROOT/experiment_output/awac_seed789/awac_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  td3bc_s42)
    CKPT="$H2O_ROOT/experiment_output/td3bc_seed42/td3bc_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  td3bc_s123)
    CKPT="$H2O_ROOT/experiment_output/td3bc_seed123/td3bc_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  td3bc_s789)
    CKPT="$H2O_ROOT/experiment_output/td3bc_seed789/td3bc_final.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  rlpd_s42)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed42_r3_rlpd_s42/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  rlpd_s123)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed123_r3_rlpd_s123/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  rlpd_s789)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_r3_rlpd_s789/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  wsrl_s42)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed42_r3_wsrl_s42/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  wsrl_s123)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed123_r3_wsrl_s123/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  wsrl_s789)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_r3_wsrl_s789/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  # ── R3 #10 nosnap (revised: --nouse_snapshot_reset for fair vs H2O+ Contrastive) ─────
  rlpd_nosnap_s42)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed42_r3_rlpd_nosnap_s42/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  rlpd_nosnap_s123)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed123_r3_rlpd_nosnap_s123/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  rlpd_nosnap_s789)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_r3_rlpd_nosnap_s789/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  wsrl_nosnap_s42)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed42_r3_wsrl_nosnap_s42/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  wsrl_nosnap_s123)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed123_r3_wsrl_nosnap_s123/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  wsrl_nosnap_s789)
    CKPT="$H2O_ROOT/experiment_output/h2oplus_bus_seed789_r3_wsrl_nosnap_s789/checkpoint_best.pt"
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  # ── P4 snapshot-reset variants (latest clean rerun; old shared-dir runs are ignored) ──
  p4_nojtt_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_best.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_nojtt_ep20_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_epoch20.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_nojtt_ep40_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_epoch40.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_nojtt_ep60_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_epoch60.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_nojtt_ep80_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_epoch80.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_nojtt_final_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' model_final.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_nojtt_hs09_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_best.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=(--hold_scale 0.9)
    ;;
  p4_nojtt_hs08_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' checkpoint_best.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=(--hold_scale 0.8)
    ;;
  p4_rescale5_nojtt_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_rescale5_nojtt_seed42_*_pid*' checkpoint_best.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_jtt_late_soft_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_jtt_late_soft_seed42_*_pid*' checkpoint_best.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_jtt_cons_ema_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_jtt_cons_ema_mix_seed42_*_pid*' checkpoint_best.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  p4_jtt_cons_ema_ep80_s42)
    CKPT="$(latest_checkpoint 'h2op_snap_p4_jtt_cons_ema_mix_seed42_*_pid*' checkpoint_epoch80.pt)"
    REQUIRE_CKPT=1
    SCRIPT="eval_with_metrics.py"; EXTRA_ARGS=()
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    echo "Valid: h2oplus_full h2oplus_darc_calql pure_online_sac{,_1000ep_s{42,123,789}} bc_full daganzo{,_a03,_a04} zero_hold ep39 p4_nojtt_s42 p4_nojtt_ep{20,40,60,80}_s42 p4_nojtt_final_s42 p4_nojtt_hs{09,08}_s42 p4_rescale5_nojtt_s42 p4_jtt_late_soft_s42 p4_jtt_cons_ema_s42 p4_jtt_cons_ema_ep80_s42" >&2
    exit 2
    ;;
esac

# ── Validate checkpoint exists for ckpt-based methods ─────────────────────────
# When a baseline-eval task is dispatched before its training task has produced a
# checkpoint, we exit with a recognisable "not-yet-ready" code (75 = EX_TEMPFAIL).
# The scheduler watcher classifies non-zero exits without a stack-trace as transient,
# and re-queues with backoff until the ckpt appears. This avoids a hard-fail
# escalation while training is still running.
if [[ "$REQUIRE_CKPT" == "1" && -z "$CKPT" ]]; then
  echo "[run_multiseed_eval] Checkpoint not yet present for method $METHOD (will be retried later)" >&2
  exit 75
fi
if [[ -n "$CKPT" && ! -f "$CKPT" ]]; then
  echo "[run_multiseed_eval] Checkpoint not yet present: $CKPT (will be retried later)" >&2
  exit 75
fi

echo "── multiseed eval ──────────────────────────────────────────────"
echo "  method     : $METHOD"
echo "  script     : $SCRIPT"
echo "  ckpt       : ${CKPT:-<none>}"
echo "  sumo_seed  : $SUMO_SEED"
echo "  od_scale   : $OD_SCALE"
echo "  out_json   : $OUT_JSON"
echo "────────────────────────────────────────────────────────────────"

# ── Run via libsumo + LSTM-RL conda env ───────────────────────────────────────
export SUMO_HOME=/usr/share/sumo
export LIBSUMO_AS_TRACI=1

cd "$HERE"

# Idempotency: skip if the target JSON already exists (manual re-runs become no-ops).
if [[ -f "$OUT_JSON" ]]; then
  echo "[run_multiseed_eval] SKIP — output already present: $OUT_JSON"
  exit 0
fi

# Use absolute python from the conda env directly — `bash -lc` (how scheduler launches us)
# does NOT read ~/.bashrc, so the `conda` CLI is not in PATH.
CMD=(/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python "$SCRIPT"
     --sumo_seed "$SUMO_SEED" --od_scale "$OD_SCALE"
     --output "$OUT_JSON" --method_tag "$METHOD"
     "${EXTRA_ARGS[@]}")

if [[ -n "$CKPT" ]]; then
  CMD+=(--checkpoint "$CKPT")
fi

echo "[run_multiseed_eval] CMD: ${CMD[*]}"
"${CMD[@]}"

echo "[run_multiseed_eval] DONE: $OUT_JSON"
