"""Bounded single-GPU fine-tuning with complete exit-85 recovery."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import checkpoint
from .common import ROOT,read_json,write_json
from .data import FeatureDataset
from .evaluate import evaluate
from .model import NavigationModel,objective,make_optimizer


def main():
    p=argparse.ArgumentParser()
    for name in ['data','features','weights','out','checkpoints','run-id']: p.add_argument('--'+name,required=True)
    p.add_argument('--initialization',choices=['pretrained','random'],required=True); a=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('One GPU required')
    torch.set_num_threads(4); cfg=read_json(ROOT/'experiments/capsaicin/pilot.json')
    cfg['initialization']=a.initialization
    if not cfg['formal']: raise RuntimeError('This config is a preflight; RGB task integration is not complete')
    source=read_json(ROOT/'source.json'); index=read_json(Path(a.data)/'index.json')
    cfg['source']=source; cfg['dataset_revision']=index['source']['revision']
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    out=Path(a.out); ckpt=Path(a.checkpoints); out.mkdir(exist_ok=True); ckpt.mkdir(exist_ok=True)
    train_frames=(cfg['source_window_frames']-1)//cfg['frame_stride']+1
    train=FeatureDataset(a.data,a.features,'train',train_frames,cfg['seed'],cfg['temporal_reverse_ratio'])
    val=FeatureDataset(a.data,a.features,'val',cfg['context_frames']+max(cfg['eval_horizons']))
    model=NavigationModel(a.weights,a.initialization,cfg['seed'],cfg['train_scope']).cuda(); optimizer=make_optimizer(model,cfg)
    step=position=0; previous_seconds=0.; best=None
    if (ckpt/'latest.pt').exists():
        restored=checkpoint.restore(ckpt/'latest.pt',model,optimizer,a.run_id,cfg)
        step=restored['step']; position=restored['data_position']; previous_seconds=restored['train_seconds']; best=restored['best_val']
    import wandb
    # Credentials are provided via NETRC by the worker; diagnostics never import this module.
    wandb_args=dict(entity=cfg['wandb_entity'],project=cfg['wandb_project'],id=a.run_id,
                    name='100m-'+a.initialization,config={**cfg,'gpu':torch.cuda.get_device_name(),
                    'torch':torch.__version__,'load_report':model.load_report},dir=str(out))
    try:
        run=wandb.init(**wandb_args,resume='allow',settings=wandb.Settings(init_timeout=90))
    except wandb.errors.CommError:
        print('WANDB_OFFLINE_FALLBACK',a.run_id,flush=True)
        run=wandb.init(**wandb_args,mode='offline')
        write_json(out/'wandb-offline.json',dict(run_id=a.run_id,needs_sync=True))
    begin=time.monotonic(); model.train()
    def elapsed(): return previous_seconds+time.monotonic()-begin
    def save(name='latest.pt'):
        checkpoint.save(ckpt/name,model,optimizer,run_id=a.run_id,config=cfg,step=step,
                        data_position=position,train_seconds=elapsed(),best_val=best)
    def log(values):
        item={'step':step,**values}
        with (out/'metrics.jsonl').open('a') as file: file.write(json.dumps(item,allow_nan=False)+'\n')
        run.log(values,step=step); print(json.dumps(item,allow_nan=False),flush=True)
    write_json(out/'run.json',dict(run_id=a.run_id,config=cfg,gpu=torch.cuda.get_device_name(),complete=False))
    print('FORMAL_TRAIN_START',a.run_id,'resume_step',step,flush=True)
    while step<cfg['max_steps'] and elapsed()<cfg['max_train_seconds']:
        started=time.monotonic(); optimizer.zero_grad(set_to_none=True); loss_sum=0.; teacher=0.; ar=0.
        progress=max(0.,(step-cfg['warmup_steps'])/max(1,cfg['max_steps']-cfg['warmup_steps']))
        scale=min(1.,(step+1)/cfg['warmup_steps'])*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups: group['lr']=group['base_lr']*scale
        for _ in range(cfg['grad_accum']):
            features,actions=train.batch(position,cfg['batch_size']); position+=1
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,parts=objective(model,features,actions,cfg['context_frames'],cfg['rollout_steps'])
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            (loss/cfg['grad_accum']).backward()
            loss_sum+=float(loss.detach())/cfg['grad_accum']; teacher+=parts['teacher_loss']/cfg['grad_accum']; ar+=parts['rollout_loss']/cfg['grad_accum']
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
        optimizer.step(); step+=1; torch.cuda.synchronize()
        log({'train/loss':loss_sum,'train/teacher':teacher,'train/rollout':ar,'train/grad_norm':float(gradient),
             'train/seconds_per_step':time.monotonic()-started,'train/elapsed_seconds':elapsed(),
             'train/lr':optimizer.param_groups[0]['lr'],'train/data_position':position,
             'train/visual_tokens_seen':position*cfg['batch_size']*train_frames*(cfg['height']//16)*(cfg['width']//16),
             'gpu/peak_memory_bytes':torch.cuda.max_memory_allocated()})
        if step%cfg['eval_every']==0:
            metrics=evaluate(model,val,cfg,limit=8); log({'dev/'+k:v for k,v in metrics.items()})
            score=metrics['rollout/h5/l1']
            if best is None or score<best: best=score; save('best.pt')
        if step%cfg['save_every']==0: save()
        if time.monotonic()-begin>=cfg['segment_seconds'] and step<cfg['max_steps'] and elapsed()<cfg['max_train_seconds']:
            save(); write_json(out/'status.json',dict(status='checkpoint',step=step,run_id=a.run_id,train_seconds=elapsed()))
            run.finish(); raise SystemExit(85)
    # Retain both the final state and the independently selected best development state.
    save()
    if not (ckpt/'best.pt').exists(): save('best.pt')
    selected=torch.load(ckpt/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True); selected_step=selected['step']; del selected
    final={}
    for split,dataset in [('dev',val),('test',FeatureDataset(a.data,a.features,'test',cfg['context_frames']+max(cfg['eval_horizons'])))]:
        metrics=evaluate(model,dataset,cfg,controls=True)
        final.update({split+'/'+k:v for k,v in metrics.items()})
    final['evaluation/checkpoint_step']=selected_step; log(final)
    result=dict(complete=True,run_id=a.run_id,step=step,selected_step=selected_step,train_seconds=elapsed(),
                stop_reason='steps' if step>=cfg['max_steps'] else 'time_budget',metrics=final,config=cfg)
    write_json(out/'completed.json',result); run.finish(); print('FORMAL_TRAIN_COMPLETE',a.run_id,step,flush=True)


if __name__=='__main__': main()
