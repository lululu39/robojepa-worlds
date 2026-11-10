"""Single-GPU, independent-frame feature extraction with no W&B runs."""
import argparse
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import read_json, write_json, upstream_path


def load_encoder(weights):
    upstream_path()
    from app.robo_jepa.utils import build_encoder
    return build_encoder(filename=str(Path(weights)/'encoder.pt'),device='cuda').eval()


def decode_frames(path,indices,height=240,width=320):
    wanted=set(indices); frames=[]
    with av.open(str(path)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in wanted:
                im=frame.to_image().convert('RGB').resize((width,height),Image.Resampling.BICUBIC)
                frames.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1))
    if len(frames)!=len(indices): raise ValueError('Decoded frame selection mismatch')
    return torch.stack(frames)


@torch.inference_mode()
def encode_tensor(encoder,frames,batch_size=8):
    parts=[]
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for batch in frames.split(batch_size):
        image=(batch.cuda().float()/255.-mean)/std
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=encoder(image.unsqueeze(2))
        if features.shape[1:]!=(300,1664): raise ValueError(f'Unexpected encoder output {features.shape}')
        features=F.layer_norm(features.float(),(1664,)).to(torch.bfloat16)
        parts.append(features.cpu().reshape(-1,1,15,20,1664))
    out=torch.cat(parts)
    if not torch.isfinite(out).all(): raise ValueError('Non-finite encoder features')
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--weights',required=True)
    p.add_argument('--out',required=True); p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=4); p.add_argument('--report',required=True); a=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('One GPU required')
    torch.set_num_threads(4)
    data=Path(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=read_json(data/'index.json')['clips'][a.shard::a.shards]
    encoder=load_encoder(a.weights); begin=time.monotonic(); total=0
    for row in rows:
        frames=decode_frames(data/row['video'],row['indices'])
        features=encode_tensor(encoder,frames)
        result={'id':row['id'],'features':features,'actions':torch.tensor(row['actions']),
                'indices':row['indices'],'split':row['split'],'scene':row['scene']}
        tmp=out/(row['id']+'.pt.tmp'); torch.save(result,tmp); tmp.replace(out/(row['id']+'.pt'))
        total+=len(features)
        print('ENCODED',row['id'],len(features),flush=True)
    report=dict(complete=True,shard=a.shard,shards=a.shards,clips=len(rows),frames=total,
                seconds=time.monotonic()-begin,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                bytes=sum(p.stat().st_size for p in out.glob('*.pt')),
                preprocessing='240x320 bicubic RGB/ImageNet/T=1; FP32 layernorm then BF16',
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write_json(a.report,report); print(report,flush=True)


if __name__=='__main__': main()
