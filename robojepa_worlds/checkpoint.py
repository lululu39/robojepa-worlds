"""Atomic complete training state; no diagnostic checkpoint reuse."""
import os
from pathlib import Path
import random
import numpy as np
import torch


def save(path,model,optimizer,**metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                 rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
                 rng_cuda=torch.cuda.get_rng_state_all(),**metadata)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def restore(path,model,optimizer,run_id,config):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['run_id']!=run_id or state['config']!=config:
        raise ValueError('Resume identity/config mismatch')
    model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda'])
    return {k:v for k,v in state.items() if k not in ['model','optimizer','rng_python','rng_numpy','rng_torch','rng_cuda']}
