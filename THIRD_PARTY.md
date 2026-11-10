# Upstream sources

Git submodules retain their original licenses; this repository's MIT license does not relicense them.

| Source | Use | License location |
| --- | --- | --- |
| facebookresearch/robo_jepa | Predictor, pretrained model architecture/loading, MAE objective reference | `vendor/robo_jepa/LICENSE` |
| facebookresearch/vjepa2 | Frozen visual encoder | `vendor/vjepa2/LICENSE` |
| AMD-AGI/Micro-World | Capsaicin data/task protocol, direct action parser and temporal reversal calls | `vendor/micro_world/LICENSE.txt` |

The action-conversion reference port in `robojepa_worlds/capsaicin.py` follows `microworld/data/dataset_game_video.py` and `tools/fix_ws_swap.py` from Micro-World. The upstream dataset file credits VideoX-Fun (Apache 2.0); the Micro-World repository supplies its Open RAIL-MSD terms. Retain those notices and terms for that derived portion. Other original orchestration code remains under the root MIT license. Model and dataset terms apply separately. No pretrained tensors or media are redistributed.
