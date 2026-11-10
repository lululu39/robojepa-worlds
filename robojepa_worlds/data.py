"""Deterministic, restartable batches from memory-mapped per-clip feature files."""
from collections import OrderedDict
from pathlib import Path
import random

import torch
from .common import read_json


class FeatureDataset:
    def __init__(self,data,features,split,frames,seed=42):
        self.rows=[r for r in read_json(Path(data)/'index.json')['clips'] if r['split']==split]
        self.root=Path(features); self.frames=frames; self.seed=seed; self.cache=OrderedDict()
        if not self.rows: raise ValueError('Empty split '+split)
        for r in self.rows:
            if len(r['indices'])<frames or not (self.root/(r['id']+'.pt')).is_file():
                raise ValueError('Missing/short cached clip: '+r['id'])

    def clip(self,i):
        key=self.rows[i]['id']
        if key not in self.cache:
            value=torch.load(self.root/(key+'.pt'),map_location='cpu',mmap=True,weights_only=True)
            if value['id']!=key or value['split']!=self.rows[i]['split']: raise ValueError('Cache identity mismatch')
            self.cache[key]=value
            if len(self.cache)>8: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self,position,batch_size):
        rng=random.Random(self.seed+position*1000003)
        features=[]; actions=[]
        for _ in range(batch_size):
            value=self.clip(rng.randrange(len(self.rows)))
            start=rng.randrange(len(value['features'])-self.frames+1)
            features.append(value['features'][start:start+self.frames])
            actions.append(value['actions'][start:start+self.frames-1])
        return torch.stack(features).cuda(),torch.stack(actions).cuda()

    def evaluation_windows(self,windows=2,limit=None):
        for i in range(min(len(self.rows),limit or len(self.rows))):
            value=self.clip(i); available=len(value['features'])-self.frames
            for j in range(windows):
                start=0 if windows==1 else round(j*available/(windows-1))
                yield value['features'][start:start+self.frames][None].cuda(),value['actions'][start:start+self.frames-1][None].cuda()
