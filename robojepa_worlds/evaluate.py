"""Scene-held-out latent rollout metrics; action controls are not simulator truth."""
import torch
from torch.nn import functional as F
from .model import rollout


@torch.inference_mode()
def evaluate(model,dataset,cfg,limit=None,controls=False):
    was_training=model.training; model.eval()
    sums={}; count=0; context=cfg['context_frames']; horizons=cfg['eval_horizons']
    def record(prefix,pred,target):
        for h in horizons:
            p=pred[:,h-1].float(); t=target[:,h-1].float()
            values={'l1':(p-t).abs().mean(),'cosine':F.cosine_similarity(p,t,dim=-1).mean()}
            for metric,value in values.items():
                if not torch.isfinite(value): raise FloatingPointError('Non-finite evaluation')
                key=f'{prefix}/h{h}/{metric}'; sums[key]=sums.get(key,0.)+float(value)
    for features,actions in dataset.evaluation_windows(cfg['eval_windows_per_clip'],limit):
        target=features[:,context:context+max(horizons)]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=rollout(model,features[:,:context],actions,max(horizons))
            record('rollout',prediction,target)
            record('persistence',features[:,context-1:context].expand_as(target),target)
            if controls:
                # Preserve context controls, perturb the queried/future commands only.
                for name in ['zero_actions','reversed_actions']:
                    changed=actions.clone()
                    future=changed[:,context-1:]
                    if name=='zero_actions': future.zero_()
                    else: changed[:,context-1:]=future.flip(1).clone()
                    alternate=rollout(model,features[:,:context],changed,max(horizons))
                    record(name,alternate,target)
                    delta=float((alternate.float()-prediction.float()).abs().mean())
                    sums[name+'/prediction_delta']=sums.get(name+'/prediction_delta',0.)+delta
        count+=1
    if not count: raise ValueError('No evaluation windows')
    model.train(was_training)
    return {**{k:v/count for k,v in sums.items()},'windows':count}
