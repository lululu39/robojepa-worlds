#!/bin/bash
# Source in an HTCondor interactive GPU shell, from its transferred scratch directory.
# Preparation failures return to the shell; model failures remain inside pdb.
export DECODER_DEBUG_ROOT="$PWD"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PATH="/opt/project/.venv/bin:$PATH"

decoder_prepare_session() (
    set -euo pipefail
    cd "$DECODER_DEBUG_ROOT"
    mkdir -p results predictor-state
    if [ ! -f source/.debug-source-ready ]; then
        tar -xf source.tar
        touch source/.debug-source-ready
    fi
    cd source
    if [ ! -f .debug-runtime-ready ]; then
        tar -xf "$DECODER_DEBUG_ROOT/decoder-runtime.tar" .venv
        touch .debug-runtime-ready
    fi
    export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD"
    uv pip freeze --python .venv/bin/python > "$DECODER_DEBUG_ROOT/results/environment.txt"
    diff -u experiments/capsaicin/decoder_env/installed.txt "$DECODER_DEBUG_ROOT/results/environment.txt"
    python -m cluster.decoder_environment
    for name in decoder-assets capsaicin-data robojepa-weights features-0 features-1 features-2 features-3; do
        if [ ! -f "$DECODER_DEBUG_ROOT/.unpacked-$name" ]; then
            tar -xf "$DECODER_DEBUG_ROOT/$name.tar" -C "$DECODER_DEBUG_ROOT"
            touch "$DECODER_DEBUG_ROOT/.unpacked-$name"
        fi
    done
    if [ ! -f "$DECODER_DEBUG_ROOT/predictor-state/checkpoints/best.pt" ]; then
        tar -xf "$DECODER_DEBUG_ROOT/checkpoints.tar" -C "$DECODER_DEBUG_ROOT/predictor-state" checkpoints/best.pt
    fi
    python -m robojepa_worlds.decoder_prepare --out "$DECODER_DEBUG_ROOT/assets" --verify-only
    printf 'DECODER_DEBUG_READY\n'
)

decoder_activate() {
    cd "$DECODER_DEBUG_ROOT/source" || return
    export PATH="$PWD/.venv/bin:/opt/project/.venv/bin:$PATH" PYTHONPATH="$PWD"
    export TORCH_HOME="$DECODER_DEBUG_ROOT/assets/torch" VJEPA2_HUB_REPO="$PWD/vendor/vjepa2"
}

decoder_save() (
    set -euo pipefail
    cd "$DECODER_DEBUG_ROOT"
    mkdir -p results
    if [ -d source/robojepa_worlds ]; then
        tar -cf results/debug-source.tar -C source robojepa_worlds cluster experiments/capsaicin/decoder-gate.json source.json
    fi
    tar -cf results.tar.tmp results
    mv results.tar.tmp results.tar
)

decoder_run() {
    decoder_activate || return
    local decoder_attempt
    decoder_attempt=$(date -u +%Y%m%dT%H%M%SZ)
    python -m pdb -c continue -m robojepa_worlds.decoder_gate \
        --data "$DECODER_DEBUG_ROOT/data" --features "$DECODER_DEBUG_ROOT/features" \
        --weights "$DECODER_DEBUG_ROOT/weights" --assets "$DECODER_DEBUG_ROOT/assets" \
        --predictor-checkpoint "$DECODER_DEBUG_ROOT/predictor-state/checkpoints/best.pt" \
        --out "$DECODER_DEBUG_ROOT/results/$decoder_attempt"
    decoder_save
}

trap decoder_save EXIT
printf 'Run decoder_prepare_session, then decoder_run. Failures stay in this allocation.\n'
