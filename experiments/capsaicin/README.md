# Capsaicin navigation pilot

Use only `200f/video/` and matching `200f/metadata-detection/` from `amd/Micro-World-Capsaicin`. The dataset revision and exact files/sizes are frozen in `dataset-manifest.json` before submission. Original metadata is retained; corrected labels are generated without modifying source files.

## Task and actions

Predict future visual latents from past visual latents and commanded controls. Per-frame keyboard fields `ws`, `ad`, `scs` and mouse `pitch_delta`, `yaw_delta` are converted into nine channels: W, S, A, D, jump, sneak, sprint, normalized pitch, normalized yaw. No collision or invalid-jump outcomes enter the model. Mouse degree values are normalized with the recorded `mouse_normalized_to_mc` factors (normally 1/15), matching the source convention without an additional sign flip.

Apply the official Micro-World W/S correction once, unless `_sprite_ws_refixed` is already set. Action index `t` is interpreted as the command taking frame `t` to `t+1`; source timestamps and decoded frame counts are checked during preprocessing. At stride 4, four **ordered** nine-channel commands are concatenated (36 values) and padded to RoboJEPA's 64 input slots, preserving order and held keys. Frames without the complete command interval are dropped. This is commanded-action conditioning, not conditioning on realized displacement.

Use a deterministic 240×320 resize and ImageNet normalization, identical for all splits. Encode every fourth frame independently (`T=1`) with the official V-JEPA 2.1 image path, then layer-normalize tokens as in the released predictor. This caching policy has no random augmentation and is recorded as such. Expected cache size is about 22 GB at BF16, rather than 88 GB for all source frames.

Split by scene prefix preceding `_aNNN_vNN`, keeping all camera anchors and action variants for each scene together. At least one scene is held out for validation and one for test; test is evaluated only at completion. Report both scene and trajectory counts. A small number of scenes makes this a pilot, not a broad generalization benchmark.

## Model and budget

Official 100M predictor, frozen pretrained encoder, one new `capsaicin` embodiment. Existing predictor tensors load strictly; only the added embodiment entries are new. No unrelated generator checkpoint is used. Zero proprioception is used at both training and inference. Full 100M predictor fine-tuning plus a higher learning rate for the action/state/embodiment interface. Random-initialized predictor control shares data, encoder, schedule and seed.

Each branch: at most 2,000 updates and 6 GPU-hours of training, global batch 8 (microbatch 2 × accumulation 4), 4 context frames plus a 2-step differentiable rollout. Evaluate 1/2/5/10-step latent L1/cosine errors, persistence, zero-action and time-reversed-action controls. These controls measure conditioning sensitivity, not counterfactual simulator correctness. Longer-horizon decoding and RGB generation are outside this first pilot. The random predictor control measures learning under the same small budget, not convergence from scratch.

## Cluster stages

1. CPU preparation: locked environment, selective downloads, source sizes, decoded frames and actions, scene split, archives.
2. Single-GPU gate: strict checkpoint loading, encoder preprocessing, causal prefix, finite gradients, action sensitivity, numerical recovery and throughput.
3. Four independent single-GPU encoding shards, each transferred back as a persistent archive. No distributed gradient synchronization.
4. Two independent single-GPU training jobs after successful gate and encoding, with 1800-second exit-85 checkpoint transfers; same W&B run ID across restarts.

Only `latest` and `best` training checkpoints persist. Checkpoints contain model, optimizer, RNG, optimizer step, accumulated training time and deterministic sampler position. Output archive failure overrides exit 85 with a failure code. Failure stops the workflow; there is no blind automatic resubmission.

Submit from the repository with `python3 -m cluster.submit prepare`. The bounded controller is `python3 -m cluster.workflow`; `--once` performs one observation/transition. It collects only complete JSON reports, commits/pushes them, and advances through the four stages above. A `STOPPED` marker under the external jobs directory stops future submissions; stopping existing jobs also requires removing their recorded Condor IDs. No secrets or raw stderr are published. It prefers H100/H200; after one continuous idle hour, encoding/training can also use A100 80GB if the passed gate records the runtime's compiled Ampere SDPA support. Blackwell is not enabled by this runtime.

Current results belong under `reports/`. Static tests do not establish GPU correctness; a successful `gate.json` is required before encoding or formal training. Formal W&B starts only in the training worker. Transient W&B connection failure falls back to local logs with the same run ID, returned in the results archive for later sync.

W&B: https://wandb.ai/yibozhong657-none/robojepa-capsaicin

References: [RoboJEPA](https://github.com/facebookresearch/robo_jepa), [Micro-World data tooling](https://github.com/AMD-AGI/Micro-World), [Capsaicin](https://huggingface.co/datasets/amd/Micro-World-Capsaicin).
