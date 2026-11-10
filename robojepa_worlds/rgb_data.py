"""Matched cached JEPA states and RGB frames, preserving Capsaicin preprocessing."""
from collections import OrderedDict
from pathlib import Path

import torch

from .common import read_json
from .encode import decode_frames


class RgbFeatureDataset:
    def __init__(self, data, features, split):
        self.data = Path(data)
        self.features = Path(features)
        index = read_json(self.data / 'index.json')
        if index['schema'] != 2 or index['frame_stride'] != 4:
            raise ValueError('Expected existing schema-2 stride-4 cache')
        self.rows = [r for r in index['clips'] if r['split'] == split]
        self.cache = OrderedDict()

    def clip(self, i):
        row = self.rows[i]
        key = row['id']
        if key not in self.cache:
            latent = torch.load(self.features / (key + '.pt'), map_location='cpu', mmap=True, weights_only=True)
            if (latent['schema'] != 2 or latent['id'] != key or latent['split'] != row['split']
                    or latent['indices'] != row['indices']):
                raise ValueError('Frame/latent identity mismatch')
            rgb = decode_frames(self.data / row['video'], row['indices'])
            if len(rgb) != len(latent['features']):
                raise ValueError('Frame/latent count mismatch')
            self.cache[key] = (latent, rgb)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self, clip_index, frame_indices):
        latent, rgb = self.clip(clip_index)
        return (latent['features'][frame_indices, None].cuda(),
                rgb[frame_indices, None].cuda().float() / 255.)
