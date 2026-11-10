"""Single-GPU, independent-frame feature extraction with no W&B runs."""
import argparse
import gc
import os
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import read_json, write_json, upstream_path


def compact_encoder(weights):
    """One-time checkpoint extraction on the GPU gate's higher-memory worker."""
    weights=Path(weights); source=weights/'encoder-original.pt'; destination=weights/'encoder.pt'
    state=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    tensors=state.get('target_encoder')
    if not tensors or not all(isinstance(v,torch.Tensor) for v in tensors.values()):
        raise ValueError('Malformed target_encoder checkpoint')
    temporary=destination.with_suffix('.pt.tmp')
    with temporary.open('wb') as file:
        torch.save({'target_encoder':tensors},file); file.flush(); os.fsync(file.fileno())
    os.replace(temporary,destination)
    del state,tensors; gc.collect()
    source.unlink()
    print('ENCODER_COMPACTED',destination.stat().st_size,flush=True)
    return destination.stat().st_size


def load_encoder(weights):
    upstream_path()
    from app.robo_jepa.utils import build_encoder
    return build_encoder(filename=str(Path(weights)/'encoder.pt'),device='cuda').eval()


def decode_frames(path,indices,height=176,width=320):
    from torchvision.transforms.functional import center_crop
    wanted=set(indices); frames=[]
    with av.open(str(path)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in wanted:
                # Match stage-2 default CenterCrop, then halve its spatial cost for JEPA.
                im=center_crop(frame.to_image().convert('RGB'),[352,640])
                im=im.resize((width,height),Image.Resampling.BILINEAR)
                frames.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1))
    if len(frames)!=len(indices): raise ValueError('Decoded frame selection mismatch')
    return torch.stack(frames)


@torch.inference_mode()
def encode_tensor(encoder,frames,batch_size=8):
    parts=[]
    gh,gw=frames.shape[-2]//16,frames.shape[-1]//16
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    for batch in frames.split(batch_size):
        image=(batch.cuda().float()/255.-mean)/std
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=encoder(image.unsqueeze(2))
        if features.shape[1:]!=(gh*gw,1664): raise ValueError(f'Unexpected encoder output {features.shape}')
        features=F.layer_norm(features.float(),(1664,)).to(torch.bfloat16)
        parts.append(features.cpu().reshape(-1,1,gh,gw,1664))
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
    index=read_json(data/'index.json')
    if index.get('schema')!=2: raise ValueError('Stale action/data contract')
    rows=index['clips'][a.shard::a.shards]
    encoder=load_encoder(a.weights); begin=time.monotonic(); total=0
    for row in rows:
        frames=decode_frames(data/row['video'],row['indices'])
        features=encode_tensor(encoder,frames)
        result={'id':row['id'],'features':features,'actions':torch.tensor(row['actions']),
                'schema':2,'frame_actions':torch.tensor(row['frame_actions']),
                'indices':row['indices'],'split':row['split'],'scene':row['scene']}
        tmp=out/(row['id']+'.pt.tmp'); torch.save(result,tmp); tmp.replace(out/(row['id']+'.pt'))
        total+=len(features)
        print('ENCODED',row['id'],len(features),flush=True)
    report=dict(complete=True,shard=a.shard,shards=a.shards,clips=len(rows),frames=total,
                seconds=time.monotonic()-begin,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                bytes=sum(p.stat().st_size for p in out.glob('*.pt')),
                preprocessing='352x640 center crop -> 176x320 bilinear RGB/ImageNet/T=1; layernorm BF16',
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write_json(a.report,report); print(report,flush=True)


if __name__=='__main__': main()
