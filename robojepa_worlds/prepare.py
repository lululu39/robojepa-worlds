"""Worker-only downloads and decoded-frame/data-contract validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

from .common import ROOT, read_json, write_json
from .capsaicin import action_chunks, make_splits, scene_id


def download(url, path, expected=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and expected is not None and path.stat().st_size==expected:
        return
    tmp=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=120) as src, tmp.open('wb') as dst:
        shutil.copyfileobj(src,dst,4*1024*1024)
    if expected is not None and tmp.stat().st_size!=expected:
        raise ValueError(f"Size mismatch: {path.name}")
    os.replace(tmp,path)


def archive(path, entries):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    with tarfile.open(tmp,'w') as out:
        for source,name in entries:
            out.add(source,arcname=name)
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='prepared'); a=p.parse_args()
    out=Path(a.out); out.mkdir(exist_ok=True)
    manifest=read_json(ROOT/'experiments/capsaicin/dataset-manifest.json')
    data=out/'data'; data.mkdir(exist_ok=True)
    def fetch(item):
        download(f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{item['path']}",
                 data/item['path'],item['bytes'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,manifest['files']))
    print('DATA_DOWNLOAD_COMPLETE',flush=True)
    import av
    stems=[Path(x['path']).stem for x in manifest['files'] if x['path'].endswith('.mp4')]
    split,scenes=make_splits(stems)
    rows=[]
    for stem in sorted(stems):
        video=data/'200f/video'/f'{stem}.mp4'
        with av.open(str(video)) as container:
            stream=container.streams.video[0]; fps=float(stream.average_rate)
            pts=[]; count=0
            for frame in container.decode(stream):
                count+=1
                if frame.time is not None: pts.append(float(frame.time))
            if any(y<=x for x,y in zip(pts,pts[1:])):
                raise ValueError(f'Non-monotonic timestamps: {stem}')
        meta=read_json(data/'200f/metadata-detection'/f'{stem}.json')
        idx,actions=action_chunks(meta,count)
        rows.append(dict(id=stem,scene=scene_id(stem),split=split[stem],frames=count,fps=fps,
                         indices=idx,actions=actions,video=f'200f/video/{stem}.mp4'))
    if any(r['frames']!=200 for r in rows):
        raise ValueError('Unexpected source frame count; inspect before continuing')
    index={'source':manifest,'scenes':scenes,'clips':rows,'frame_stride':4,
           'action_alignment':'command[t] maps frame[t] to frame[t+1]',
           'ws_policy':'swap unless _sprite_ws_refixed', 'outcomes_are_inputs':False}
    write_json(data/'index.json',index)
    # Published encoder file includes auxiliary tensors. Preserve only target_encoder.
    weights=out/'weights'; weights.mkdir(exist_ok=True)
    urls=manifest['weights']
    for name,spec in urls.items():
        download(spec['url'],weights/name,spec['bytes'])
    import torch
    source=weights/'encoder-original.pt'
    state=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    if 'target_encoder' not in state: raise ValueError('Missing target encoder')
    tensors=state['target_encoder']
    if not tensors or not all(isinstance(v,torch.Tensor) for v in tensors.values()):
        raise ValueError('Malformed encoder state')
    torch.save({'target_encoder':tensors},weights/'encoder.pt')
    del state,tensors
    source.unlink()
    archive(out/'capsaicin-data.tar',[(data,'data')])
    archive(out/'robojepa-weights.tar',[(weights,'weights')])
    summary=dict(complete=True,created_at=time.time(),clips=len(rows),scenes=scenes,
                 split_counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']},
                 frames=sum(r['frames'] for r in rows),cached_frames=sum(len(r['indices']) for r in rows),
                 artifacts={p.name:p.stat().st_size for p in out.glob('*.tar')},
                 encoder_bytes=(weights/'encoder.pt').stat().st_size)
    write_json(out/'preparation.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
