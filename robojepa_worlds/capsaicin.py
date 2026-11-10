"""Pure-Python Capsaicin action and split contracts (no GPU imports)."""
import math
import random
import re

ACTION_CHANNELS = ["forward", "backward", "left", "right", "jump", "sneak", "sprint", "pitch", "yaw"]


def scene_id(stem):
    match = re.fullmatch(r"(.+)_a\d+_v\d+", stem)
    if not match:
        raise ValueError(f"Unrecognized scene/anchor naming: {stem}")
    return match.group(1)


def make_splits(stems, seed=42):
    scenes = sorted({scene_id(s) for s in stems})
    if len(scenes) < 3:
        raise ValueError("Need at least three scene groups for disjoint train/val/test")
    random.Random(seed).shuffle(scenes)
    n = max(1, round(len(scenes) * .15))
    mapping = {s: ("test" if i < n else "val" if i < 2*n else "train") for i, s in enumerate(scenes)}
    return {stem: mapping[scene_id(stem)] for stem in sorted(stems)}, mapping


def frame_actions(metadata):
    corrected = bool(metadata.get("action_encoding", {}).get("_sprite_ws_refixed", False))
    factors = metadata.get("mouse_normalized_to_mc", {"pitch_factor": 1/15, "yaw_factor": 1/15})
    actions = metadata["actions"]
    keys = sorted(map(int, actions))
    if keys != list(range(len(keys))):
        raise ValueError("Action indices must be contiguous starting at zero")
    output = []
    for i in keys:
        a = actions[str(i)]
        ws, ad, scs = int(a["ws"]), int(a["ad"]), int(a["scs"])
        if ws not in (0,1,2) or ad not in (0,1,2) or scs not in (0,1,2,3):
            raise ValueError(f"Invalid keyboard enum at frame {i}")
        if not corrected:
            ws = {0:0, 1:2, 2:1}[ws]
        # Keep commanded motion even when collision=1. Outcomes are not commands.
        row = [float(ws==1), float(ws==2), float(ad==1), float(ad==2),
               float(scs==1), float(scs==2), float(scs==3),
               float(a["pitch_delta"])*float(factors["pitch_factor"]),
               float(a["yaw_delta"])*float(factors["yaw_factor"])]
        if not all(math.isfinite(x) for x in row):
            raise ValueError(f"Non-finite action at frame {i}")
        output.append(row)
    return output


def action_chunks(metadata, frame_count, stride=4):
    if stride*len(ACTION_CHANNELS) > 64:
        raise ValueError("Ordered action chunk exceeds 64 slots")
    rows = frame_actions(metadata)
    if len(rows) < frame_count-1:
        raise ValueError("Insufficient commands for decoded video")
    indices = list(range(0, frame_count, stride))
    chunks = []
    for left,right in zip(indices,indices[1:]):
        flat = [x for row in rows[left:right] for x in row]
        if len(flat) != stride*len(ACTION_CHANNELS):
            raise ValueError("Incomplete action interval")
        chunks.append(flat + [0.]*(64-len(flat)))
    return indices,chunks
