"""Watch only the authorized predictor job, exiting after its first finite update."""
import argparse
import fcntl
import json
import math
import time

from robojepa_worlds.common import ROOT,read_json,write_json
from .submit import JOBS,command
from .workflow import query


def first_update(text,run_id):
    started=False; resume_step=0
    for line in text.splitlines():
        if line.startswith('FORMAL_TRAIN_START '):
            parts=line.split()
            started=len(parts)==4 and parts[1]==run_id and parts[2]=='resume_step'
            if started: resume_step=int(parts[3])
            continue
        if not started or not line.startswith('{'): continue
        try: item=json.loads(line)
        except json.JSONDecodeError: continue
        if 'train/loss' not in item: continue
        values=[item.get(k) for k in ['train/loss','train/teacher','train/rollout','train/grad_norm']]
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in values):
            raise RuntimeError('Non-finite/missing formal training metrics')
        if item.get('step',0)>resume_step: return item
    return None


def tick():
    root=JOBS/'train'
    if (JOBS/'STOPPED').exists(): return 'stopped'
    state=read_json(root/'state.json')
    status,ad=query(state)
    log=root/'logs/job.out'
    first=first_update(log.read_text(errors='replace') if log.exists() else '',state['run_id'])
    observation=dict(time=time.time(),job=state['job'],run_id=state['run_id'],job_status=status,
                     remote_host=ad.get('RemoteHost'),starts=ad.get('NumJobStarts',0))
    if first is not None:
        write_json(root/'first-update.json',{**observation,'status':'formal_training_started','metrics':first})
        return 'formal_training_started'
    if status==4: raise RuntimeError('Job completed without a finite formal update in its log')
    if status==1 and time.time()-ad['EnteredCurrentStatus']>=3600 and not state.get('a100_fallback'):
        gate=read_json(ROOT/'experiments/capsaicin/reports/gate.json')
        # Existing cu126 SDPA supports Ampere; do not enable unsupported Blackwell.
        if gate.get('attention')=='sdpa' and 'sm_80' in gate.get('cuda_architectures',[]):
            requirement='regexp("H100|H200|A100",DeviceName) && !regexp("MIG",DeviceName) && (GlobalMemoryMb >= 76000)'
            command(['condor_qedit',state['job'],'RequireGPUs',requirement])
            state['a100_fallback']=True; write_json(root/'state.json',state)
    write_json(root/'watch-status.json',{**observation,'status':'waiting_for_first_update'})
    return 'waiting_for_first_update'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--once',action='store_true'); args=parser.parse_args()
    root=JOBS/'train'
    with (root/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try: status=tick()
            except Exception as error:
                write_json(root/'watch-status.json',dict(time=time.time(),status='needs_attention',error=str(error)))
                raise
            print(time.strftime('%FT%TZ',time.gmtime()),status,flush=True)
            if args.once or status!='waiting_for_first_update': return
            time.sleep(60)


if __name__=='__main__': main()
