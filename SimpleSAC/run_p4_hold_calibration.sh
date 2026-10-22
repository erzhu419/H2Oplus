#!/usr/bin/env bash
# Select P4 no-JTT checkpoint/hold_scale on a validation SUMO seed, then
# evaluate the frozen choice on the paper default test seeds.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H2O_ROOT="$(cd "$HERE/.." && pwd)"
EXP_ROOT="$H2O_ROOT/experiment_output"
VAL_DIR="$EXP_ROOT/calibration_eval"
OUT_DIR="$EXP_ROOT/multiseed_eval"
PY="${PY:-/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python}"
VAL_SEED="${VAL_SEED:-2001}"
TEST_SEEDS="${TEST_SEEDS:-1001 1002 1003}"
OD_SCALE="${OD_SCALE:-1.0}"

mkdir -p "$VAL_DIR" "$OUT_DIR"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="${LIBSUMO_AS_TRACI:-1}"

latest_p4_dir() {
  find "$EXP_ROOT" -maxdepth 1 -type d \
    -name 'h2op_snap_p4_floor_ess_nojtt_seed42_*_pid*' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

P4_DIR="$(latest_p4_dir)"
if [[ -z "$P4_DIR" ]]; then
  echo "[calib] No clean P4 no-JTT directory found under $EXP_ROOT" >&2
  exit 75
fi

declare -A CKPTS=(
  [best]="$P4_DIR/checkpoint_best.pt"
  [ep60]="$P4_DIR/checkpoint_epoch60.pt"
  [ep80]="$P4_DIR/checkpoint_epoch80.pt"
)
SCALES=(0.70 0.75 0.80 0.85 0.90 1.00)

cd "$HERE"

echo "[calib] P4 dir: $P4_DIR"
echo "[calib] validation seed: $VAL_SEED"
echo "[calib] test seeds: $TEST_SEEDS"

for ckpt_label in best ep60 ep80; do
  ckpt="${CKPTS[$ckpt_label]}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[calib] Missing checkpoint: $ckpt" >&2
    exit 75
  fi
  for scale in "${SCALES[@]}"; do
    scale_tag="${scale/./}"
    method="p4_calibgrid_${ckpt_label}_hs${scale_tag}_s42"
    out_json="$VAL_DIR/${method}_sumo${VAL_SEED}_od${OD_SCALE}.json"
    if [[ -f "$out_json" ]]; then
      echo "[calib] SKIP validation existing: $out_json"
      continue
    fi
    echo "[calib] validation method=$method seed=$VAL_SEED od=$OD_SCALE ckpt=$ckpt"
    "$PY" eval_with_metrics.py \
      --sumo_seed "$VAL_SEED" --od_scale "$OD_SCALE" \
      --output "$out_json" --method_tag "$method" \
      --hold_scale "$scale" --checkpoint "$ckpt"
  done
done

SEL_JSON="$VAL_DIR/p4_hold_calibration_val${VAL_SEED}_selected.json"
VAL_DIR="$VAL_DIR" VAL_SEED="$VAL_SEED" OD_SCALE="$OD_SCALE" "$PY" - <<'PY'
import glob
import json
import os
import re

val_dir = os.environ["VAL_DIR"]
val_seed = os.environ["VAL_SEED"]
od_scale = os.environ["OD_SCALE"]
pattern = os.path.join(val_dir, f"p4_calibgrid_*_s42_sumo{val_seed}_od{od_scale}.json")
rows = []
for path in glob.glob(pattern):
    with open(path) as f:
        data = json.load(f)
    method = data.get("method_tag") or os.path.basename(path).split("_sumo", 1)[0]
    m = re.match(r"p4_calibgrid_(?P<ckpt>.+)_hs(?P<scale_tag>[0-9]+)_s42$", method)
    if not m:
        continue
    rows.append({
        "method": method,
        "checkpoint_label": m.group("ckpt"),
        "checkpoint": data["checkpoint"],
        "hold_scale": float(data.get("hold_scale", 1.0)),
        "validation_seed": int(data.get("sumo_seed", val_seed)),
        "od_scale": float(data.get("od_scale", od_scale)),
        "validation_reward": float(data["cumulative_reward"]),
        "validation_per_step_reward": float(data.get("per_step_reward", 0.0)),
        "validation_passenger_wait": data.get("passenger_wait_mean"),
        "validation_hold_mean": (data.get("action") or {}).get("hold_mean"),
        "validation_json": path,
    })

if not rows:
    raise SystemExit("no validation rows found")

rows.sort(key=lambda r: (r["validation_reward"], -r["hold_scale"]), reverse=True)
best = rows[0]
out_path = os.path.join(val_dir, f"p4_hold_calibration_val{val_seed}_selected.json")
with open(out_path, "w") as f:
    json.dump({"selected": best, "candidates": rows}, f, indent=2)
print(f"[calib] selected: {best['method']} reward={best['validation_reward']:.0f} "
      f"ckpt={best['checkpoint_label']} hold_scale={best['hold_scale']}")
print(f"[calib] selected_json: {out_path}")
PY

CKPT_PATH="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$SEL_JSON")"
HOLD_SCALE="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["hold_scale"])' "$SEL_JSON")"
CKPT_LABEL="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint_label"])' "$SEL_JSON")"
VAL_REWARD="$("$PY" -c 'import json,sys; print(round(json.load(open(sys.argv[1]))["selected"]["validation_reward"], 3))' "$SEL_JSON")"

FROZEN_METHOD="p4_calib_val${VAL_SEED}_s42"
echo "[calib] frozen method=$FROZEN_METHOD ckpt=$CKPT_LABEL hold_scale=$HOLD_SCALE val_reward=$VAL_REWARD"

for seed in $TEST_SEEDS; do
  out_json="$OUT_DIR/${FROZEN_METHOD}_sumo${seed}_od${OD_SCALE}.json"
  if [[ -f "$out_json" ]]; then
    echo "[calib] SKIP frozen existing: $out_json"
    continue
  fi
  echo "[calib] frozen eval method=$FROZEN_METHOD seed=$seed od=$OD_SCALE"
  "$PY" eval_with_metrics.py \
    --sumo_seed "$seed" --od_scale "$OD_SCALE" \
    --output "$out_json" --method_tag "$FROZEN_METHOD" \
    --hold_scale "$HOLD_SCALE" --checkpoint "$CKPT_PATH"
done

"$PY" aggregate_multiseed.py
