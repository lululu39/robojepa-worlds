"""Real GPU/data gate. Never initializes W&B or supplies formal-run weights."""
import argparse
import gc
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from . import checkpoint
from .common import ROOT,read_json,write_json
from .encode import load_encoder,decode_frames,encode_tensor
from .model import NavigationModel,objective,make_optimizer,rollout
from .reference import reference_actions,reverse_sample


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--weights',required=True)
    p.add_argument('--out',required=True); a=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('One GPU required')
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42); np.random.seed(42)
    cfg=read_json(ROOT/'experiments/capsaicin/pilot.json'); report={'complete':False}
    write_json(a.out,report); begin=time.monotonic()
    index=read_json(Path(a.data)/'index.json')
    if index.get('schema')!=2: raise ValueError('Stale data contract')
    row=next(r for r in index['clips']
             if r['split']=='train' and any(any(value!=0 for value in command[:36]) for command in r['actions'][:5]))
    metadata=read_json(Path(a.data)/'200f/metadata-detection'/(row['id']+'.json'))
    incoming=reference_actions(metadata,row['frames'])
    torch.testing.assert_close(incoming,torch.tensor(row['frame_actions']),rtol=1e-6,atol=1e-6)
    window=reference_actions(metadata,81,start=4)
    if window[0].any(): raise AssertionError('Reference first-frame action must be zero')
    pixels=torch.arange(5.)[:,None]; controls=torch.zeros(5,9); controls[1:,7]=torch.arange(1.,5.)
    controls[1:,0]=1
    reversed_pixels,mouse,keyboard=reverse_sample(pixels,controls)
    torch.testing.assert_close(reversed_pixels,pixels.flip(0))
    torch.testing.assert_close(mouse[:,0],torch.tensor([0.,-4.,-3.,-2.,-1.]))
    torch.testing.assert_close(keyboard[:,1],controls[:,0].flip(0))
    report['reference_action_and_reversal_parity']=True
    states=(cfg['source_window_frames']-1)//cfg['frame_stride']+1
    frames=decode_frames(Path(a.data)/row['video'],row['indices'][:states])
    encoder=load_encoder(a.weights)
    features=encode_tensor(encoder,frames)[None].cuda()
    del encoder,frames; gc.collect(); torch.cuda.empty_cache()
    # Reloaded cached tensors must support ordinary training autograd.
    features=features.clone(); actions=torch.tensor(row['actions'][:states-1],device='cuda')[None]
    model=NavigationModel(a.weights,train_scope=cfg['train_scope']).cuda(); model.eval()
    report['load']=model.load_report
    # Numerical retention check, limited to the model already loaded for this gate.
    raw=torch.load(Path(a.weights)/'predictor.pt',map_location='cpu',mmap=True,weights_only=False)
    old={k.removeprefix('module.'):v for k,v in raw.get('model',raw.get('predictor')).items()}
    current=model.predictor.state_dict()
    for key,value in old.items():
        got=current[key].cpu()
        if key=='embodiment_assignment_tokens': got=got[:,:26]
        elif key.endswith('patch_embed.w'): got=got[:,:,:26]
        elif key.endswith('patch_embed.b'): got=got[:26]
        elif key.endswith('patch_embed.to_onehot'): got=got[:26,:26]
        if not torch.equal(got,value): raise AssertionError('Pretrained tensor changed: '+key)
    del old,raw,current,got
    report['pretrained_retained']=True
    before={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        original=model(features[:,:6],actions[:,:6])
        changed=features[:,:6].clone(); changed[:,4:]=torch.randn_like(changed[:,4:])
        changed_actions=actions[:,:6].clone(); changed_actions[:,4:]=torch.randn_like(changed_actions[:,4:])
        altered=model(changed,changed_actions)
        prefix=model(features[:,:4],actions[:,:4])
        causal_error=float((original[:,:4].float()-altered[:,:4].float()).abs().max())
        prefix_error=float((original[:,:4].float()-prefix.float()).abs().max())
        if causal_error>1e-5 or prefix_error>.04: raise AssertionError(f'Causality/prefix failed: {causal_error}, {prefix_error}')
        perturbed=actions[:,:4].clone(); perturbed[:,-1,:36]+=1
        action_delta=float((model(features[:,:4],perturbed)[:,-1].float()-prefix[:,-1].float()).abs().mean())
        if action_delta<=1e-7: raise AssertionError('Actions have no effect')
        long=rollout(model,features[:,:1],actions,20)
        if not torch.isfinite(long).all(): raise FloatingPointError('Non-finite long rollout')
    report.update(causal_max_error=causal_error,prefix_max_error=prefix_error,action_delta=action_delta,
                  long_rollout_finite=True)
    del original,changed,changed_actions,altered,prefix,perturbed,long
    model.train(); optimizer=make_optimizer(model,cfg)
    x=features.repeat(cfg['batch_size'],1,1,1,1,1)
    act=actions.repeat(cfg['batch_size'],1,1)
    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,_=objective(model,x,act,cfg['context_frames'],cfg['rollout_steps'])
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
        action_grad=model.predictor.action_encoder.patch_embed.w.grad[:,:,-1].float().abs().sum()
        if not torch.isfinite(action_grad) or action_grad<=0: raise AssertionError('Missing action gradient')
        optimizer.step(); torch.cuda.synchronize()
        return float(loss.detach()),float(gradient),float(action_grad)
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic(); first=update(); seconds=time.monotonic()-started
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'resume.pt'
        checkpoint.save(path,model,optimizer,step=1,data_position=1,run_id='gate',config=cfg)
        expected_rng=(random.random(),float(np.random.random()),float(torch.rand(1,device='cuda')))
        second=update(); expected={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        restored=checkpoint.restore(path,model,optimizer,'gate',cfg)
        actual_rng=(random.random(),float(np.random.random()),float(torch.rand(1,device='cuda')))
        if expected_rng!=actual_rng or restored['data_position']!=1: raise AssertionError('RNG/data resume failed')
        repeated=update()
        maximum=max(float((p.detach().cpu()-expected[n]).abs().max()) for n,p in model.named_parameters())
        if abs(second[0]-repeated[0])>1e-6 or maximum>2e-6: raise AssertionError(f'Optimizer resume failed: {maximum}')
    old_row_axes={'predictor.action_encoder.patch_embed.w':-1,'predictor.action_encoder.patch_embed.b':0,
                  'predictor.state_encoder.patch_embed.b':0,'predictor.embodiment_assignment_tokens':1}
    changed_task=False
    for name,p in model.named_parameters():
        actual=p.detach().cpu(); original=before[name]
        if name in old_row_axes:
            index=[slice(None)]*actual.ndim; index[old_row_axes[name]]=slice(0,model.embodiment_id)
            if not torch.equal(actual[tuple(index)],original[tuple(index)]): raise AssertionError('Old embodiment drift: '+name)
            changed_task=changed_task or not torch.equal(actual,original)
        elif not torch.equal(actual,original): raise AssertionError('Frozen trunk drift: '+name)
    if not changed_task: raise AssertionError('Task interface did not update')
    report.update(complete=True,first_loss=first[0],gradient_norm=first[1],action_gradient_l1=first[2],
                  resume_max_parameter_error=maximum,resume_loss_error=abs(second[0]-repeated[0]),
                  batch_size=cfg['batch_size'],seconds_per_microbatch=seconds,
                  peak_predictor_gpu_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__,cuda_architectures=torch.cuda.get_arch_list(),attention='sdpa',
                  frozen_trunk_and_old_embodiments_unchanged=True,source_window_frames=cfg['source_window_frames'],
                  seconds=time.monotonic()-begin,source=read_json(ROOT/'source.json'))
    write_json(a.out,report); print('GATE_COMPLETE',report,flush=True)


if __name__=='__main__': main()
