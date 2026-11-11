#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_RUN_ID="$1"
mkdir -p results checkpoints predictor-state
finish() {
    TASK_EXIT=$?
    trap - EXIT
    printf '{"exit_code": %s}\n' "$TASK_EXIT" > "$TASK_ROOT/results/worker-exit.json"
    if ! tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results ||
       ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar" ||
       ! tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" checkpoints ||
       ! mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar"; then exit 74; fi
    exit "$TASK_EXIT"
}
trap finish EXIT
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
tar -xf source.tar
cd source
tar -xf "$TASK_ROOT/decoder-runtime.tar" .venv
export PATH="$PWD/.venv/bin:/opt/project/.venv/bin:$PATH" PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
uv pip freeze --python .venv/bin/python > "$TASK_ROOT/results/environment.txt"
diff -u experiments/capsaicin/decoder_env/installed.txt "$TASK_ROOT/results/environment.txt"
python -m cluster.decoder_environment
tar -xf "$TASK_ROOT/decoder-assets.tar" -C "$TASK_ROOT"
python -m robojepa_worlds.decoder_prepare --out "$TASK_ROOT/assets" --verify-only
tar -xf "$TASK_ROOT/capsaicin-data.tar" -C "$TASK_ROOT"
for TASK_ARCHIVE in "$TASK_ROOT"/features-*.tar; do tar -xf "$TASK_ARCHIVE" -C "$TASK_ROOT"; done
# Only the frozen predictor is needed for the final RGB rollout; no encoder loading.
tar -xf "$TASK_ROOT/robojepa-weights.tar" -C "$TASK_ROOT" weights/predictor.pt
tar -xf "$TASK_ROOT/predictor-checkpoints.tar" -C "$TASK_ROOT/predictor-state" checkpoints/best.pt
export TORCH_HOME="$TASK_ROOT/assets/torch" VJEPA2_HUB_REPO="$PWD/vendor/vjepa2"
export WANDB_MODE=online NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results"
python -m robojepa_worlds.decoder_train \
    --data "$TASK_ROOT/data" --features "$TASK_ROOT/features" \
    --assets "$TASK_ROOT/assets" --weights "$TASK_ROOT/weights" \
    --predictor-checkpoint "$TASK_ROOT/predictor-state/checkpoints/best.pt" \
    --out "$TASK_ROOT/results" --checkpoints "$TASK_ROOT/checkpoints" --run-id "$TASK_RUN_ID"
