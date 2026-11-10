# RoboJEPA Worlds

Fine-tune pretrained **RoboJEPA** for action-conditioned prediction in interactive environments. Other projects supply datasets and task definitions; their world-model checkpoints are not used.

## Layout

- `robojepa_worlds/`: shared data, model, training, evaluation and checkpoint code.
- `experiments/capsaicin/`: first-person navigation experiment, fixed configuration and small reports.
- `cluster/`: independent single-GPU HTCondor jobs and resumable transfers.
- `vendor/`: pinned official RoboJEPA and V-JEPA 2 source submodules, under their respective licenses.
- `tests/`: data contracts, causality, gradient and recovery checks.

Clone with `git clone --recurse-submodules`. Resolve dependencies with the committed `uv.lock`. GPU work runs on compute nodes; the submit node only edits code, submits jobs and reads reports.

## First experiment

Capsaicin 200-frame videos, one view at 240×320, frozen V-JEPA 2.1 encoder and the released 100M RoboJEPA predictor. A new embodiment receives keyboard and mouse commands, with zero proprioception. Collision and invalid-jump outcomes are **never inputs**. The first milestone is latent dynamics prediction, not RGB generation or a playable simulator.

Initial bounded pilot: one pretrained-predictor fine-tune and one randomly initialized predictor control, identical frozen encoder and training data, each capped at 2,000 optimizer updates and 6 GPU-hours of training. No automatic larger-model run or sweep. The GPU correctness gate must pass first. These pilot limits are engineering choices, not a claim of convergence.

Formal runs use W&B entity `yibozhong657-none`, project `robojepa-capsaicin`; diagnostic jobs never create W&B runs. Future subtasks get separate projects. Credentials, datasets, features and checkpoints stay outside Git. Small JSON metrics and sanitized job summaries are versioned under the experiment.

See [Capsaicin](experiments/capsaicin/README.md) for data semantics, splits and commands. The repository records the exact source commits, upstream dataset revision, configuration, dependency versions and GPU used. It does not perform content-hash scans.

## Licenses

Original orchestration and adaptation code is MIT licensed. `vendor/robo_jepa` and `vendor/vjepa2` retain upstream licenses, including non-commercial model/source restrictions where applicable. Dataset and pretrained-weight licenses remain applicable independently. No pretrained tensors or dataset media are redistributed here.
