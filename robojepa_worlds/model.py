"""Strict loading of the official predictor with one additional embodiment."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, upstream_path


def predictor_config():
    cfg=json.loads((ROOT/'vendor/robo_jepa/configs/released/wm-100m/config.json').read_text())
    cfg.pop('encoder_version')
    cfg.update(num_views=1,view_groups_v=[1],attn_impl='sdpa',use_activation_checkpointing=True)
    return cfg


class NavigationModel(nn.Module):
    def __init__(self,weights,initialization='pretrained',seed=42):
        super().__init__(); upstream_path()
        from app.robo_jepa.robo_jepa.models.predictor import vit_predictor
        torch.manual_seed(seed)
        cfg=predictor_config(); n=len(cfg['embodiment_list'])
        cfg['embodiment_list']=cfg['embodiment_list']+['capsaicin']
        self.predictor=vit_predictor(**cfg)
        self.embodiment_id=n; self.config=cfg
        self.load_report={'initialization':initialization,'new_embodiment_id':n}
        if initialization=='pretrained':
            checkpoint=torch.load(Path(weights)/'predictor.pt',map_location='cpu',mmap=True,weights_only=False)
            raw=checkpoint.get('model',checkpoint.get('predictor'))
            if not isinstance(raw,dict): raise ValueError('No official predictor state found')
            state={k.removeprefix('module.'):v for k,v in raw.items()}
            target=self.predictor.state_dict()
            if set(state)!=set(target):
                raise ValueError(f'Unexpected checkpoint keys: missing={set(target)-set(state)} extra={set(state)-set(target)}')
            expanded=[]
            for name,value in state.items():
                if value.shape==target[name].shape: continue
                new=target[name].clone()
                if name=='embodiment_assignment_tokens':
                    assert value.shape[1]==n; new[:,:n].copy_(value)
                elif name in ['action_encoder.patch_embed.w','state_encoder.patch_embed.w']:
                    assert value.shape[-1]==n; new[:,:,:n].copy_(value)
                elif name in ['action_encoder.patch_embed.b','state_encoder.patch_embed.b']:
                    assert value.shape[0]==n; new[:n].copy_(value)
                elif name in ['action_encoder.patch_embed.to_onehot','state_encoder.patch_embed.to_onehot']:
                    assert value.shape==(n,n) and torch.equal(value,torch.eye(n,dtype=value.dtype))
                    # Keep the new (n+1)x(n+1) identity, including its final diagonal.
                else:
                    raise ValueError(f'Unapproved shape change: {name} {value.shape} -> {new.shape}')
                state[name]=new; expanded.append(name)
            self.predictor.load_state_dict(state,strict=True)
            self.load_report.update(strict=True,tensors=len(state),expanded=expanded)
            del checkpoint,state,raw
        elif initialization!='random': raise ValueError(initialization)

    def forward(self,features,actions):
        b,t=features.shape[:2]
        if features.ndim!=6 or actions.shape!=(b,t,64):
            raise ValueError('Expected features B,T,1,H,W,C and aligned actions B,T,64')
        states=torch.zeros_like(actions)
        ids=torch.full((b,),self.embodiment_id,dtype=torch.long,device=features.device)
        views=torch.zeros((b,1),dtype=torch.long,device=features.device)
        return self.predictor(features,actions,states=states,embodiment_ids=ids,view_map=views)[0]


def latent_loss(pred,target):
    pred,target=pred.float(),target.float()
    return F.l1_loss(pred,target)+.1*(1-F.cosine_similarity(pred,target,dim=-1).mean())


def rollout(model,context,actions,steps):
    history=context
    predictions=[]
    for k in range(steps):
        t=history.shape[1]
        # Keep positions anchored while the full sequence remains below 16 frames.
        if t>16: raise ValueError('Pilot rollout exceeds positional context budget')
        y=model(history,actions[:,:t])[:,-1:]
        predictions.append(y)
        history=torch.cat([history,y],dim=1)
    return torch.cat(predictions,dim=1)


def objective(model,features,actions,context=4,steps=2):
    teacher=model(features[:,:-1],actions)
    one=latent_loss(teacher,features[:,1:])
    predicted=rollout(model,features[:,:context],actions,steps)
    ar=latent_loss(predicted,features[:,context:context+steps])
    return (one+ar)/2,{'teacher_loss':float(one.detach()),'rollout_loss':float(ar.detach())}


def make_optimizer(model,cfg):
    groups={'trunk':[],'interface':[]}
    for name,p in model.named_parameters():
        if p.requires_grad:
            interface=any(x in name for x in ['action_encoder','state_encoder','embodiment_assignment'])
            groups['interface' if interface else 'trunk'].append(p)
    return torch.optim.AdamW([
        {'params':groups['trunk'],'lr':cfg['learning_rate'],'base_lr':cfg['learning_rate']},
        {'params':groups['interface'],'lr':cfg['action_learning_rate'],'base_lr':cfg['action_learning_rate']}],
        weight_decay=cfg['weight_decay'])
