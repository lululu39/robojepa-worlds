"""Call pinned Micro-World task code directly; import only on compute workers."""
import sys
import torch
from .common import ROOT
from .capsaicin import corrected_metadata,frame_actions


def reference_dataset():
    path=str(ROOT/'vendor/micro_world')
    if path not in sys.path: sys.path.insert(0,path)
    from microworld.data.dataset_game_video import VideoGameDataset
    dataset=VideoGameDataset.__new__(VideoGameDataset)
    dataset.process_collisions=True
    return dataset


def reference_actions(metadata,frame_count,start=0):
    corrected=corrected_metadata(metadata)
    # Reject missing labels rather than silently using upstream zero padding.
    own=torch.tensor(frame_actions(corrected),dtype=torch.float32)
    if len(own)<frame_count or not 0<=start<frame_count: raise ValueError('Invalid action window')
    mouse,keyboard=reference_dataset()._actions_to_tensors(corrected,start,frame_count-1,frame_count-start)
    actual=torch.cat([keyboard,mouse],dim=-1)
    expected=own[start:frame_count].clone(); expected[0]=0
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    return actual


def reverse_sample(features,rows):
    """Use the upstream coupled reversal; callers restrict it to no-jump samples."""
    return reference_dataset()._reverse_temporal(features,rows[:,7:],rows[:,:7])
