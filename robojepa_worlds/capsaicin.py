"""Small task adapters; upstream action parsing runs in reference.py on workers."""
# Action conversion semantics adapted from AMD-AGI/Micro-World's dataset_game_video.py.
# That upstream source and its license remain in vendor/micro_world; see THIRD_PARTY.md.
import copy
import math
import random
import re

ACTION_CHANNELS = ["forward", "backward", "left", "right", "jump", "sneak", "sprint", "pitch", "yaw"]


def scene_id(stem):
    # Asset families, NOT individual camera anchors. Some 200f clips have no _a/_v suffix.
    families=[('gallery',r'^allow_gallery_'),('sponza',r'^(allow_sponza_|cap_sp_)'),
              ('bistro',r'^(be_|bi_|bw_|cap_br_)'),('suntemple',r'^suntemple_')]
    families += [(name,rf'^polyamp_bistro_.*_{name}$')
                 for name in ['hidden_alley','namaqualand','the_shed','verdant_trail']]
    for family,pattern in families:
        if re.search(pattern,stem): return family
    raise ValueError(f'Unmapped Capsaicin asset family: {stem}')


def make_splits(stems, seed=42):
    scenes = sorted({scene_id(s) for s in stems})
    if len(scenes) < 3:
        raise ValueError("Need at least three scene groups for disjoint train/val/test")
    random.Random(seed).shuffle(scenes)
    n = max(1, round(len(scenes) * .15))
    mapping = {s: ("test" if i < n else "val" if i < 2*n else "train") for i, s in enumerate(scenes)}
    return {stem: mapping[scene_id(stem)] for stem in sorted(stems)}, mapping


def corrected_metadata(metadata):
    """In-memory equivalent of upstream tools/fix_ws_swap.py; retain raw files."""
    result=copy.deepcopy(metadata)
    encoding=result.setdefault('action_encoding',{})
    if not encoding.get('_sprite_ws_refixed'):
        for action in result['actions'].values(): action['ws']={1:2,2:1}.get(action['ws'],action['ws'])
        encoding['_sprite_ws_refixed']=True
    return result


def frame_actions(metadata,process_collisions=True):
    """Stdlib reference-port for AP checks. Workers use the actual upstream parser.

    Rows are incoming-frame controls in DEGREES. Row zero has no preceding frame.
    See vendor/micro_world/microworld/data/dataset_game_video.py:_actions_to_tensors.
    """
    corrected = bool(metadata.get("action_encoding", {}).get("_sprite_ws_refixed", False))
    factors=metadata.get('mouse_normalized_to_mc')
    if isinstance(factors,dict):
        pitch_factor=factors.get('pitch_factor',-1/15); yaw_factor=factors.get('yaw_factor',1/15)
    elif metadata.get('source') in ['Capsaicin','Blender']:
        pitch_factor=yaw_factor=1/15
    else:
        pitch_factor=yaw_factor=1.
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
        if process_collisions:
            if a.get('collision',0)==1: ws=ad=0
            if a.get('jump_invalid',0)==1: scs=0
        row = [float(ws==1), float(ws==2), float(ad==1), float(ad==2),
               float(scs==1), float(scs==2), float(scs==3),
               float(a['pitch_delta'])*float(pitch_factor)*15,
               float(a['yaw_delta'])*float(yaw_factor)*15]
        if not all(math.isfinite(x) for x in row):
            raise ValueError(f"Non-finite action at frame {i}")
        output.append(row if i else [0.]*9)
    return output


def chunk_rows(rows,frame_count,stride=4):
    if stride*len(ACTION_CHANNELS) > 64:
        raise ValueError("Ordered action chunk exceeds 64 slots")
    if len(rows) < frame_count:
        raise ValueError("Insufficient commands for decoded video")
    indices = list(range(0, frame_count, stride))
    chunks = []
    for left,right in zip(indices,indices[1:]):
        # Upstream row i describes the transition INTO frame i.
        flat = [x for row in rows[left+1:right+1] for x in row]
        if len(flat) != stride*len(ACTION_CHANNELS):
            raise ValueError("Incomplete action interval")
        chunks.append(flat + [0.]*(64-len(flat)))
    return indices,chunks


def action_chunks(metadata,frame_count,stride=4):
    return chunk_rows(frame_actions(metadata),frame_count,stride)
