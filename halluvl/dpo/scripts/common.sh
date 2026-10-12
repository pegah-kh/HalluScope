set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS_DIR/config.sh"
cd "$SCRIPTS_DIR/.."

IMAGES="$DATA_ROOT/images"
PROMPTS="$DATA_ROOT/prompts"
ANSWERS="$DATA_ROOT/answers/$MODEL"
PAIRS="$DATA_ROOT/pairs/$MODEL"
DATASET="$DATA_ROOT/dataset"

# Runs the commands in JOBS, one queue per GPU.
run_on_gpus() {
    local pids=() failed=0
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        (
            for ((j = gpu; j < ${#JOBS[@]}; j += NUM_GPUS)); do
                echo "[GPU $gpu] ${JOBS[$j]}"
                CUDA_VISIBLE_DEVICES=$gpu bash -c "${JOBS[$j]}"
            done
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    return $failed
}
