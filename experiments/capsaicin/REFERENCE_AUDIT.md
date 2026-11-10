# Micro-World code audit

Task reference: AMD-AGI/Micro-World, pinned submodule commit `ad8c653c896a48c19cebb3dd4ccf802b40421c7a`. Model reference: the pinned official RoboJEPA and V-JEPA 2 submodules. No Micro-World/Wan generator weights are used.

| Area | Source inspected | Correction / adaptation |
| --- | --- | --- |
| Action timing | `microworld/data/dataset_game_video.py`, `_actions_to_tensors`, `_reverse_temporal` | Incoming-frame convention. Corrected chunk rows from `[l:r]` to `[l+1:r+1]`; frame-zero control is zero. |
| Mouse units | Same `_actions_to_tensors` | Preserve the final factor × 15. Previous code incorrectly kept MC-normalized units, 15× smaller on Capsaicin. Workers now call the actual upstream function. |
| W/S labels | `tools/fix_ws_swap.py` | Idempotent in-memory repair with `_sprite_ws_refixed`; raw files preserved. |
| Collision policy | Dataset's `process_collisions`; stage-2 trainer defaults | Match processed movement labels. Previous raw-command-only policy was a task deviation; removed as the default. |
| Frame window / reversal | `scripts/wan2.2/train_causal_ti2v_action_capsaicin.sh`, `VideoGameDataset.get_batch` | Use 81 source frames and upstream 0.5 coupled reversal. Cached starts are stride-aligned; no-jump restriction follows the stated validity assumption. |
| Framing | Stage-2 constructor leaves `frame_scaling=crop` | Match 352×640 center crop, then halve spatial resolution. Previous 240×320 full-frame resize changed aspect ratio and field of view. |
| Trainable parameters | `train_causal_ti2v_action.py`, `requires_grad_(False)` / `trainable_modules=action` | Freeze pretrained trunk first. Optimize new JEPA task rows only; this is an interface adaptation, not Wan AdaLN reuse. |
| Predictor view geometry | RoboJEPA `configs/released/wm-100m/config.json`, `models/predictor.py`, `models/utils/modules.py` | Preserve released `num_views=2`, view groups and all 14 pretrained view embedding tables. Pass the single game camera as runtime V=1 with view ID 0 through upstream's mixed-view support. Constructing with `num_views=1` incorrectly removes checkpoint weights and changes the RoPE channel split. |
| Objective | Micro-World FM target; `robo_jepa/.../world_model.py:WorldModel.compute_loss` | Wan FM cannot be applied directly to a JEPA predictor. Use official RoboJEPA MAE; removed the ad-hoc cosine penalty. Short differentiable rollout is the documented JEPA adaptation. |
| Input / rollout | `examples/wan2.2/predict_causal_action_evalcase.py`, stage-2 `i2v_ratio=1` | One observed frame, with action-conditioned rollout. Previous four-frame conditioning was an easier, different task. |
| Splits | Fixed HF file manifest; no packaged reference split/CSV in this checkout | Previous suffix-only parser rejected valid clips and treated anchors as scenes. Eight conservative asset families now keep related captures together; this is a new held-out split, not an official result protocol. |
| Evaluation | `tools/extract_real_case.py`, `test_metrics/evaluate.py`, `test_metrics/visual_metrics.py` | Keep future RGB/action/video metrics explicit. Latent errors remain preflight diagnostics. |

## Remaining work and limits

The repository still lacks a connected RGB decoder and video-generation evaluation. The official RoboJEPA alternative is its V-JEPA-conditioned DiT + Cosmos head (`docs/decoder.md`, `app/robo_jepa/world_model_heads.py`), rather than replacing the model with Wan. Its pretrained checkpoint availability, non-robotic reconstruction quality and single-GPU cost must be verified before choosing a decoder/training stage; no decoder training run is added by this audit.

Micro-World's metrics additionally require external GameWorldScore/IDM assets. Its `visual_metrics.py` passes [0,1] frames to an LPIPS implementation while its own comment says [-1,1]; this needs an explicit correction/check before reuse. The Minecraft IDM's validity on Capsaicin must also be established rather than assumed. No video metrics have been measured here.

The retired random-initialization and full-predictor branches, their CLI options and trunk optimizer settings have been removed from the current code. Old versions remain in Git history. Formal submission remains disabled (`formal=false`, `auto_train=false`). Preparation, the corrected GPU preflight and four independent GPU encoding shards may proceed. The user explicitly requested queuing the downstream encoding jobs; this does not make latent validation a video-generation result. This is not a change in user authorization or a request for renewed permission: the implementation has not yet met the task definition. No training budget is increased.
