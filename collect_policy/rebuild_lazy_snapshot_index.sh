#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$BASE/bus_h2o/datasets_v2"
PY="${PYTHON_BIN:-/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python}"

TMP="$DATA/merged_all_v2.lazy_tmp.h5"
TMP_MAN="$DATA/file_manifest.lazy_tmp.json"
OUT="$DATA/merged_all_v2.h5"
MAN="$DATA/file_manifest.json"
READY="$DATA/.merged_all_v2_lazy_snapshot_ready"

rm -f "$READY" "$TMP" "$TMP_MAN"

cd "$BASE"
"$PY" collect_policy/merge_v2_lazy.py \
    --input_dir "$DATA" \
    --output "$TMP" \
    --manifest "$TMP_MAN" \
    --snapshot_key snapshot_T1

"$PY" collect_policy/validate_lazy_snapshot_index.py \
    --merged "$TMP" \
    --manifest "$TMP_MAN" \
    --archive_dir "$DATA" \
    --snapshot_key snapshot_T1 \
    --samples 5

ts="$(date +%y%m%d_%H%M%S)"
if [[ -f "$OUT" ]]; then
    mv "$OUT" "$OUT.pre_lazy_$ts"
fi
if [[ -f "$MAN" ]]; then
    cp "$MAN" "$MAN.pre_lazy_$ts"
fi

mv "$TMP" "$OUT"
mv "$TMP_MAN" "$MAN"

"$PY" collect_policy/validate_lazy_snapshot_index.py \
    --merged "$OUT" \
    --manifest "$MAN" \
    --archive_dir "$DATA" \
    --snapshot_key snapshot_T1 \
    --samples 5

printf 'ready %s\n' "$(date -Iseconds)" > "$READY"
echo "READY $READY"
