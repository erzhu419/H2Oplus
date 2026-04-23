#!/usr/bin/env bash
# =============================================================================
# H2O+ Full Experiment Runner
# =============================================================================
# Runs the complete experiment matrix:
#   3 environments x 3 data sources x 3 dynamics types x 2 variety values = 54
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh            # run all 54 experiments serially
#   ./run_experiments.sh --dry-run  # print commands without executing
# =============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTRY_POINT="${PROJECT_DIR}/SimpleSAC/h2o+_main.py"
OUTPUT_DIR="${PROJECT_DIR}/experiment_output"
LOG_FILE="${OUTPUT_DIR}/experiment_runner.log"

ENVS=("HalfCheetah-v2" "Walker2d-v2" "Hopper-v2")
DATA_SOURCES=("medium_replay" "medium" "medium_expert")
DYNAMICS=("gravity" "density" "friction")
VARIETIES=("2.0" "0.5")

N_EPOCHS=1000
SEED=42
DEVICE="cuda"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ---- Helpers ----------------------------------------------------------------

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# Build a short tag used to check whether this experiment already ran.
# Matches the directory naming pattern produced by viskit/logging.
result_tag() {
    local env="$1" ds="$2" dyn="$3" var="$4"
    echo "${env}_${ds}_${dyn}x${var}_seed${SEED}"
}

# Check if results already exist for this experiment.
# Looks for any progress.csv inside output dirs whose name contains the tag.
experiment_done() {
    local tag="$1"
    # The logger creates dirs like default_<timestamp>-s-<seed>--<uuid>.
    # We look for a progress.csv in any subdir that matches the run's wandb
    # name pattern, or simply check for a marker file we create on completion.
    local marker="${OUTPUT_DIR}/.done_${tag}"
    [[ -f "$marker" ]]
}

mark_done() {
    local tag="$1"
    touch "${OUTPUT_DIR}/.done_${tag}"
}

# ---- Main loop --------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

total=0
skipped=0
succeeded=0
failed=0

log "=========================================="
log "H2O+ experiment sweep started"
log "Matrix: ${#ENVS[@]} envs x ${#DATA_SOURCES[@]} data x ${#DYNAMICS[@]} dynamics x ${#VARIETIES[@]} varieties"
log "=========================================="

for env in "${ENVS[@]}"; do
    for ds in "${DATA_SOURCES[@]}"; do
        for dyn in "${DYNAMICS[@]}"; do
            for var in "${VARIETIES[@]}"; do
                total=$((total + 1))
                tag=$(result_tag "$env" "$ds" "$dyn" "$var")

                # --- Skip if already completed ---
                if experiment_done "$tag"; then
                    log "[SKIP] ${tag} -- results exist"
                    skipped=$((skipped + 1))
                    continue
                fi

                # --- Build command ---
                cmd=(
                    python "$ENTRY_POINT"
                    --env_list "$env"
                    --data_source "$ds"
                    --unreal_dynamics "$dyn"
                    --variety_list "$var"
                    --n_epochs "$N_EPOCHS"
                    --seed "$SEED"
                    --device "$DEVICE"
                    --logging.output_dir "$OUTPUT_DIR"
                    --logging.online false
                )

                if $DRY_RUN; then
                    echo "[DRY-RUN] ${cmd[*]}"
                    continue
                fi

                log "[START] ${tag}"
                start_ts=$(date +%s)

                # Run from the project root so relative imports work.
                if (cd "$PROJECT_DIR" && \
                    PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" \
                    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"); then
                    end_ts=$(date +%s)
                    elapsed=$(( end_ts - start_ts ))
                    log "[DONE]  ${tag}  (${elapsed}s)"
                    mark_done "$tag"
                    succeeded=$((succeeded + 1))
                else
                    end_ts=$(date +%s)
                    elapsed=$(( end_ts - start_ts ))
                    log "[FAIL]  ${tag}  (${elapsed}s)"
                    failed=$((failed + 1))
                    # Continue to next experiment rather than aborting.
                fi
            done
        done
    done
done

log "=========================================="
log "Sweep complete: ${total} total, ${succeeded} succeeded, ${skipped} skipped, ${failed} failed"
log "=========================================="
