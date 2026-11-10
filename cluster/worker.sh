#!/bin/bash
set -euo pipefail
TASK_ROOT="$PWD"
TASK_STAGE="$1"
mkdir -p results checkpoints
TASK_EXIT=0
finish() {
    TASK_EXIT=$?
    trap - EXIT
    if ! tar -cf "$TASK_ROOT/results.tar.tmp" -C "$TASK_ROOT" results ||
       ! mv "$TASK_ROOT/results.tar.tmp" "$TASK_ROOT/results.tar"; then exit 74; fi
    if [ "$TASK_STAGE" = train ] || [ "$TASK_STAGE" = gate ]; then
        if ! tar -cf "$TASK_ROOT/checkpoints.tar.tmp" -C "$TASK_ROOT" checkpoints ||
           ! mv "$TASK_ROOT/checkpoints.tar.tmp" "$TASK_ROOT/checkpoints.tar"; then exit 74; fi
    fi
    exit "$TASK_EXIT"
}
trap finish EXIT
if [ -f results.tar ]; then tar -xf results.tar; fi
if [ -f checkpoints.tar ]; then tar -xf checkpoints.tar; fi
tar -xf source.tar
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export UV_CACHE_DIR="$TASK_ROOT/uv-cache" UV_LINK_MODE=copy
export PATH="/opt/project/.venv/bin:$PATH"
export WANDB_MODE=disabled
cd source
if [ "$TASK_STAGE" = prepare ]; then
    export OMP_NUM_THREADS=2 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1
    TASK_ENV_STARTED=$SECONDS
    echo "ENVIRONMENT_PREPARATION_START"
    uv sync --frozen --python /opt/project/.venv/bin/python
    .venv/bin/python -m pip --version >/dev/null 2>&1 || true
    uv pip freeze --python .venv/bin/python > "$TASK_ROOT/results/environment.txt"
    echo "ENVIRONMENT_PREPARATION_COMPLETE seconds=$((SECONDS - TASK_ENV_STARTED))"
    .venv/bin/python -m robojepa_worlds.prepare --out "$TASK_ROOT/prepared"
    tar -cf "$TASK_ROOT/runtime.tar" .venv pyproject.toml uv.lock
    mv "$TASK_ROOT/prepared/capsaicin-data.tar" "$TASK_ROOT/"
    mv "$TASK_ROOT/prepared/robojepa-source-weights.tar" "$TASK_ROOT/"
    cp "$TASK_ROOT/prepared/preparation.json" "$TASK_ROOT/results/"
else
    tar -xf "$TASK_ROOT/runtime.tar"
    if [ "$TASK_STAGE" = gate ]; then
        tar -xf "$TASK_ROOT/robojepa-source-weights.tar" -C "$TASK_ROOT"
    else
        tar -xf "$TASK_ROOT/robojepa-weights.tar" -C "$TASK_ROOT"
    fi
    if [ -f "$TASK_ROOT/capsaicin-data.tar" ]; then tar -xf "$TASK_ROOT/capsaicin-data.tar" -C "$TASK_ROOT"; fi
    export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD"
    export TORCH_HOME="$TASK_ROOT/torch-cache"
    export VJEPA2_HUB_REPO="$PWD/vendor/vjepa2"
    if [ "$TASK_STAGE" = gate ]; then
        python -m robojepa_worlds.gate --data "$TASK_ROOT/data" --weights "$TASK_ROOT/weights" --out "$TASK_ROOT/results/gate.json" --weights-archive "$TASK_ROOT/robojepa-weights.tar"
    elif [ "$TASK_STAGE" = encode ]; then
        python -m robojepa_worlds.encode --data "$TASK_ROOT/data" --weights "$TASK_ROOT/weights" --out "$TASK_ROOT/features" --shard "$2" --shards 4 --report "$TASK_ROOT/results/encode.json"
        tar -cf "$TASK_ROOT/features.tar.tmp" -C "$TASK_ROOT" features
        mv "$TASK_ROOT/features.tar.tmp" "$TASK_ROOT/features.tar"
    elif [ "$TASK_STAGE" = train ]; then
        for TASK_ARCHIVE in "$TASK_ROOT"/features-*.tar; do tar -xf "$TASK_ARCHIVE" -C "$TASK_ROOT"; done
        export WANDB_MODE=online NETRC="$TASK_ROOT/.netrc" WANDB_DIR="$TASK_ROOT/results"
        python -m robojepa_worlds.train --data "$TASK_ROOT/data" --features "$TASK_ROOT/features" --weights "$TASK_ROOT/weights" --out "$TASK_ROOT/results" --checkpoints "$TASK_ROOT/checkpoints" --run-id "$2"
    else
        echo "Unknown stage" >&2; exit 2
    fi
fi
