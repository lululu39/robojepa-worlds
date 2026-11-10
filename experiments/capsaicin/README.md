# Capsaicin: Micro-World task, RoboJEPA adaptation

The target is first-frame + keyboard/mouse controlled **video generation**, following [Micro-World 2](../../vendor/micro_world). Current execution is the latent-adaptation preflight. RGB generation and the original video metrics remain incomplete; there is no formal training or benchmark claim. Read [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) for the source comparison.

## Data and actions

Preparation requests 2 CPUs and 8 GB RAM, reuses the existing nanochat container, and builds the locked video runtime once. Weight downloads and tar packing are streamed; the 30 GB encoder checkpoint is never loaded/rewritten by this CPU job. The already-queued GPU gate (48 GB host RAM, 160 GB scratch) extracts `target_encoder` once and publishes the compact shared weights archive. All four encoding jobs reuse it.

Only the fixed `200f/video/` and matching `200f/metadata-detection/` files from `amd/Micro-World-Capsaicin` are downloaded. The 440 clips occupy about 0.95 GB. Raw JSON is retained. Correct W/S once in memory using the upstream marker, then **call `VideoGameDataset._actions_to_tensors` directly**. Compare every clip's result against the small stdlib port before caching it.

Actions are incoming-frame controls: row `i` describes motion into frame `i`; the first frame has zero control. A transition from cached frame `l` to `r=l+4` consumes rows `l+1` through `r`, in order. Each row has W/S/A/D/jump/sneak/sprint/pitch/yaw; four rows give 36 values, padded to RoboJEPA's 64 slots. Mouse values use the upstream factor × 15 conversion into **degrees**. No extra inference-time pitch flip is applied.

`process_collisions=True` matches the reference trainer: collision clears movement labels and invalid jump clears `scs`. These are **processed control labels**, not guaranteed raw player commands; collision outcomes affect the labels even though they are not separate input channels. This corrects the previous incompatible claim that outcomes never affected inputs.

Use the reference stage-2 default 352×640 center crop, then downsample to 176×320 for the single-GPU preflight. ImageNet normalization and independent-frame V-JEPA 2.1 encoding are JEPA-specific changes. Cache every fourth frame, layer-normalized, BF16: about 16 GB. Raw 200-frame action rows remain in each cache for exact temporal-window/reversal processing. Schema 2 rejects the old shifted-action cache.

The source trainer samples 81-frame windows. Our windows cover the same 80 transition frames using 21 cached states; starts are restricted to stride-4 boundaries. The 0.5 temporal-reversal augmentation calls the upstream coupled frame/keyboard/mouse function, only for no-jump windows. Evaluation never augments.

Our conservative asset-family split is **not an upstream published benchmark split**. It groups camera/anchor variants into eight families: bistro, gallery, sponza, suntemple, hidden_alley, namaqualand, the_shed, verdant_trail. With seed 42: train 332 clips, validation 20 (sponza), test 88 (namaqualand). Unknown filenames fail rather than silently becoming new scenes.

## Model and evaluation

Load the official 100M predictor strictly, retaining all old parameters and adding a Capsaicin embodiment. Follow Micro-World's action-stage freezing strategy: freeze the encoder and predictor trunk; update only the new action projection/bias, task embedding and zero-proprioception bias. Gradient masks protect old embodiment rows. Weight decay is zero because the parameters share tables with frozen embodiments; the GPU gate verifies exact old-row/trunk preservation.

Keep RoboJEPA's MAE latent objective; cosine is diagnostic. The prior extra cosine penalty was removed. Teacher-forced prediction uses overlapping windows of at most 16 states; a two-step differentiable rollout starts from **one observed frame**. Longer rollouts slide a 16-state context; this is an explicit positional-context adaptation, not Micro-World's Wan KV-cache implementation.

Latent preflight reports 1/2/5/10/20-step errors, corresponding to 4/8/20/40/80 source-frame offsets, plus persistence and action sensitivity. These do not establish RGB quality or physical correctness. The eventual video task needs matched generated/reference videos, action-controllability evaluation and the visual/temporal metrics under upstream `test_metrics`. See `evaluation-protocol.json` for the incomplete task contract.

## Selected RGB decoder

The default RGB path reuses the pretrained V-JEPA2 decoder from [JEPA-WMs](https://github.com/facebookresearch/jepa-wms), with checkpoint [`facebook/jepa-wms/vjepa2_vitg_256_INet.pth.tar`](https://huggingface.co/facebook/jepa-wms/blob/main/vjepa2_vitg_256_INet.pth.tar). RoboJEPA remains the encoder and action-conditioned predictor.

The interface is `RoboJEPA features (1664) -> Linear(1664, 1408, bias=True) -> pretrained JEPA-WMs decoder -> RGB`. The adapter has 2,344,320 parameters. The decoder retains its pretrained input projection, 1024-wide internal representation and 12 transformer blocks (about 153M trainable parameters). Reference the upstream [`VisionTransformerDecoder`](https://github.com/facebookresearch/jepa-wms/blob/main/app/plan_common/models/decoder.py) and [V-JEPA2 decoder training configuration](https://github.com/facebookresearch/jepa-wms/blob/main/configs/vjepa_wm/vm2m/open_source_decs/step2_lpips_vm2m_vjepa2vitgopen_vgxf_vitldec_dup_256_vjtransf_norm_bs4_4n.yaml); pin the source and checkpoint revisions when integrating them.

For decoder training, freeze the RoboJEPA encoder and predictor. Reuse the cached real-frame features and their matching RGB targets, briefly warm up the adapter, then fine-tune the adapter and decoder with the upstream L2 pixel and LPIPS perceptual losses. This learns feature-distribution alignment as well as the channel mapping. Adapt the upstream square-grid resizing, positional encoding and pixel-patch assembly to the 176×320 output, preserving the pretrained learned weights. Evaluate real-latent reconstruction before predicted-latent rollouts, including temporal stability on the existing held-out scenes. Framewise decoding retains the cache's stride-4 cadence; it does not itself restore the original video frame rate or establish real-time throughput.

This records the selected design. Decoder loading, training and RGB rollout integration are not implemented or GPU-validated yet; the execution flags and existing training ceiling below are unchanged.

## Execution

`python3 -m cluster.submit prepare` submits preparation. Pre-submit the GPU gate with `python3 -m cluster.submit gate --defer`, then each of the four encoding jobs with `python3 -m cluster.submit encode --shard N --defer` (N=0..3). These jobs enter the real Condor queue held on their dependencies; they do not reserve a GPU while waiting. `python3 -m cluster.workflow` releases the same job IDs after successful dependencies, collects reports, and exits after all four encoding shards finish because `auto_train=false`. Artifact sizes, source compatibility and the GPU gate are checked before release; an unexpected hold/failure stops the workflow. `formal=false` also rejects direct formal submission. These are implementation-status checks, not a new user-approval requirement.

GPU validation covers real pretrained tensors, upstream action/reversal parity, causality, action gradients, frozen-weight preservation, an 81-source-frame training window, 20-step rollout and numerical checkpoint recovery. Static checks cannot substitute for this gate. The earlier full-predictor plus random-predictor pilot is not automatically launched.

The existing 2,000-step / 6-training-hour ceiling remains unchanged. When the RGB task path is implemented, the shared training/recovery code retains optimizer, RNG, sampler position and run ID; latest/best are separate and archive failure cannot return exit 85. Formal W&B is https://wandb.ai/yibozhong657-none/robojepa-capsaicin; preparation/gates have no W&B run.
