#!/bin/bash
# Source from the transferred scratch directory in an explicit interactive GPU job.
export CONTROL_ROOT="$PWD"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PATH="/opt/project/.venv/bin:$PATH"

control_prepare() (
    set -euo pipefail
    cd "$CONTROL_ROOT"
    mkdir -p results predictor-state decoder-state
    if [ ! -f source/.eval-source-ready ]; then
        tar -xf source.tar
        touch source/.eval-source-ready
    fi
    cd source
    if [ ! -f .eval-runtime-ready ]; then
        tar -xf "$CONTROL_ROOT/decoder-runtime.tar" .venv
        touch .eval-runtime-ready
    fi
    export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD"
    uv pip freeze --python .venv/bin/python > "$CONTROL_ROOT/results/environment.txt"
    diff -u experiments/capsaicin/decoder_env/installed.txt "$CONTROL_ROOT/results/environment.txt"
    python -m cluster.decoder_environment
    for name in decoder-assets capsaicin-data features-0 features-1 features-2 features-3; do
        if [ ! -f "$CONTROL_ROOT/.unpacked-$name" ]; then
            tar -xf "$CONTROL_ROOT/$name.tar" -C "$CONTROL_ROOT"
            touch "$CONTROL_ROOT/.unpacked-$name"
        fi
    done
    if [ ! -f "$CONTROL_ROOT/weights/predictor.pt" ]; then
        tar -xf "$CONTROL_ROOT/robojepa-weights.tar" -C "$CONTROL_ROOT" weights/predictor.pt
    fi
    if [ ! -f "$CONTROL_ROOT/predictor-state/checkpoints/best.pt" ]; then
        tar -xf "$CONTROL_ROOT/checkpoints.tar" -C "$CONTROL_ROOT/predictor-state" checkpoints/best.pt
    fi
    if [ ! -f "$CONTROL_ROOT/decoder-state/checkpoints/best.pt" ]; then
        tar -xf "$CONTROL_ROOT/decoder-train-1e866f14-checkpoints.tar" -C "$CONTROL_ROOT/decoder-state" checkpoints/best.pt
    fi
    python -m robojepa_worlds.decoder_prepare --out "$CONTROL_ROOT/assets" --verify-only
    printf 'CONTROL_EVAL_READY\n'
)

control_activate() {
    cd "$CONTROL_ROOT/source" || return
    export PATH="$PWD/.venv/bin:/opt/project/.venv/bin:$PATH" PYTHONPATH="$PWD"
    export TORCH_HOME="$CONTROL_ROOT/assets/torch" VJEPA2_HUB_REPO="$PWD/vendor/vjepa2"
}

control_save() (
    set -euo pipefail
    cd "$CONTROL_ROOT"
    mkdir -p results
    if [ -d source/robojepa_worlds ]; then
        tar -cf results/evaluation-source.tar -C source robojepa_worlds cluster experiments/capsaicin/control-eval.json source.json
    fi
    tar -cf results.tar.tmp results
    mv results.tar.tmp results.tar
)

control_run() {
    control_activate || return
    python -m pdb -c continue -m robojepa_worlds.control_eval \
        --data "$CONTROL_ROOT/data" --features "$CONTROL_ROOT/features" \
        --weights "$CONTROL_ROOT/weights" --assets "$CONTROL_ROOT/assets" \
        --predictor-checkpoint "$CONTROL_ROOT/predictor-state/checkpoints/best.pt" \
        --decoder-checkpoint "$CONTROL_ROOT/decoder-state/checkpoints/best.pt" \
        --out "$CONTROL_ROOT/results/outputs"
    control_save
}

trap control_save EXIT
printf 'Run control_prepare, then control_run. This session performs no training.\n'
