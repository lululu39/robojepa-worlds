#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
mkdir -p results assets predictor-state
finish() {
    TASK_EXIT=$?
    trap - EXIT
    printf '{"exit_code": %s}\n' "$TASK_EXIT" > "$TASK_ROOT/results/worker-exit.json"
    if ! tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results ||
       ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar"; then exit 74; fi
    exit "$TASK_EXIT"
}
trap finish EXIT
tar -xf source.tar
cd source
tar -xf "$TASK_ROOT/runtime.tar"
export PATH="/opt/project/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled
export UV_CACHE_DIR="$TASK_ROOT/uv-cache" UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --project experiments/capsaicin/decoder_env --frozen --python /opt/project/.venv/bin/python
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD"
uv pip freeze --python .venv/bin/python > "$TASK_ROOT/results/environment.txt"
python -m robojepa_worlds.decoder_prepare --out "$TASK_ROOT/assets"
tar -cf "$TASK_ROOT/decoder-assets.tar.tmp" -C "$TASK_ROOT" assets
mv "$TASK_ROOT/decoder-assets.tar.tmp" "$TASK_ROOT/decoder-assets.tar"
tar -cf "$TASK_ROOT/decoder-runtime.tar.tmp" .venv experiments/capsaicin/decoder_env
mv "$TASK_ROOT/decoder-runtime.tar.tmp" "$TASK_ROOT/decoder-runtime.tar"
tar -xf "$TASK_ROOT/capsaicin-data.tar" -C "$TASK_ROOT"
tar -xf "$TASK_ROOT/robojepa-weights.tar" -C "$TASK_ROOT"
for TASK_ARCHIVE in "$TASK_ROOT"/features-*.tar; do tar -xf "$TASK_ARCHIVE" -C "$TASK_ROOT"; done
tar -xf "$TASK_ROOT/checkpoints.tar" -C "$TASK_ROOT/predictor-state" checkpoints/best.pt
export TORCH_HOME="$TASK_ROOT/assets/torch"
export VJEPA2_HUB_REPO="$PWD/vendor/vjepa2"
python -m robojepa_worlds.decoder_gate \
    --data "$TASK_ROOT/data" --features "$TASK_ROOT/features" \
    --weights "$TASK_ROOT/weights" --assets "$TASK_ROOT/assets" \
    --predictor-checkpoint "$TASK_ROOT/predictor-state/checkpoints/best.pt" \
    --out "$TASK_ROOT/results"
