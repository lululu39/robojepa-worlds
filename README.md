# RoboJEPA Worlds

Adapt **pretrained RoboJEPA** to tasks defined by existing interactive-world-model work. The task's original code supplies the data, action, inference and evaluation contracts; the model adaptation retains RoboJEPA weights. Other world-model generator checkpoints are not used.

## Layout

- `robojepa_worlds/`: data adapters, model interface, training, evaluation and recovery.
- `experiments/capsaicin/`: first-person action control, fixed configuration and small reports.
- `cluster/`: independent single-GPU HTCondor jobs with bounded execution.
- `vendor/`: pinned official RoboJEPA, V-JEPA 2 and Micro-World code.
- `tests/`: stdlib contracts; real tensor/gradient/recovery checks run in the compute-node gate.

Clone with `git clone --recurse-submodules`. Dependencies are pinned in `uv.lock`. Follow [AGENTS.md](AGENTS.md): reference the original task implementation first, document required adaptations, and never substitute a latent-loss result for video-generation evaluation.

## Current milestone

The first task is Micro-World 2's Capsaicin image-plus-keyboard/mouse controlled video generation. Current code implements a **latent-adaptation preflight**, including direct upstream action parsing, a frozen RoboJEPA 100M backbone with new task conditioning, and first-frame-conditioned latent rollouts. The RGB decoder and video-level evaluation are not integrated yet. `formal=false` and `auto_train=false` prevent the old latent pilot from silently becoming a formal video-generation experiment.

The controller currently runs preparation and one GPU correctness gate, then exits. It does not launch the earlier two-branch full-predictor/random-control proposal. No formal run has started. The prior ceiling of 2,000 optimizer updates / 6 training GPU-hours is retained in configuration, not expanded.

See [Capsaicin](experiments/capsaicin/README.md) and the [reference-code audit](experiments/capsaicin/REFERENCE_AUDIT.md) for exact reuse, corrections and remaining differences. Formal runs, when the task path is complete, use W&B `yibozhong657-none/robojepa-capsaicin`; diagnostics create no run. New subtasks use separate projects.

Credentials, data, features and checkpoints stay outside Git. Configuration, source commits, dataset revision, dependency versions and small result reports are retained. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream licenses and attribution.
