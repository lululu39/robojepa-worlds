"""Bounded stage controller: successful outputs advance; every failure stops."""
import argparse
import fcntl
import json
import re
import subprocess
import sys
import tarfile
import time

from robojepa_worlds.common import ROOT,read_json,write_json
from .submit import JOBS,command

REPORTS=ROOT/'experiments/capsaicin/reports'


def query(state):
    ads=json.loads(command(['condor_q',state['job'],'-json']) or '[]')
    if not ads:
        ads=json.loads(command(['condor_history',state['job'],'-limit','1','-json']) or '[]')
    if not ads: raise RuntimeError('Job absent from both queue and history: '+state['job'])
    ad=ads[0]; status=ad['JobStatus']
    if status in [3,5] or (status==4 and (ad.get('ExitBySignal') or ad.get('ExitCode')!=0)):
        raise RuntimeError(f"Job {state['job']} failed/held/removed (status={status}, exit={ad.get('ExitCode')})")
    return status,ad


def collect(state):
    archive=JOBS/state['key']/'results.tar'
    filename={'prepare':'preparation.json','gate':'gate.json','encode':'encode.json','train':'completed.json'}[state['stage']]
    with tarfile.open(archive) as tar:
        item=tar.getmember('results/'+filename)
        if not item.isfile() or item.size>2_000_000: raise ValueError('Unexpected report size/type')
        report=json.load(tar.extractfile(item))
        if state['stage']=='prepare':
            environment=tar.getmember('results/environment.txt')
            if not environment.isfile() or environment.size>100_000: raise ValueError('Unexpected environment report')
            versions=tar.extractfile(environment).read().decode().splitlines()
            if not all(re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',line) for line in versions):
                raise ValueError('Review non-version environment entries before publication')
            report['installed_dependencies']=versions
    if report.get('complete') is not True: raise RuntimeError('Incomplete report: '+state['key'])
    report['job_id']=state['job']; report.setdefault('source',state['source'])
    REPORTS.mkdir(exist_ok=True)
    target=REPORTS/(state['key']+'.json'); write_json(target,report)
    return target


def publish(paths):
    if not paths: return
    command(['git','add',*[str(p.relative_to(ROOT)) for p in paths]])
    if command(['git','diff','--cached','--name-only']):
        command(['git','diff','--cached','--check'])
        command(['git','commit','-m','Record successful Capsaicin stage reports'])
        command([sys.executable,'tools/publish.py'])


def submit(stage,*args):
    subprocess.run([sys.executable,'-m','cluster.submit',stage,*args],cwd=ROOT,check=True)


def tick():
    if (JOBS/'STOPPED').exists(): return 'stopped'
    if command(['git','status','--porcelain']): raise RuntimeError('Working tree changed; pause controller for review')
    states={p.parent.name:read_json(p) for p in JOBS.glob('*/state.json')}
    completed=set(); active=[]; updates=[]; observations={}
    for key,state in states.items():
        status,ad=query(state)
        observations[key]={'job':state['job'],'status':status,'starts':ad.get('NumJobStarts',0)}
        if status==1 and state['stage'] in ['encode','train'] and time.time()-ad['EnteredCurrentStatus']>=3600:
            gate=read_json(REPORTS/'gate.json')
            # cu126 SDPA has a compiled Ampere path. Blackwell is not enabled by this runtime.
            if gate.get('attention')=='sdpa' and 'sm_80' in gate.get('cuda_architectures',[]) and not state.get('a100_fallback'):
                requirement='regexp("H100|H200|A100",DeviceName) && !regexp("MIG",DeviceName) && (GlobalMemoryMb >= 76000)'
                command(['condor_qedit',state['job'],'RequireGPUs',requirement])
                state['a100_fallback']=True; write_json(JOBS/key/'state.json',state)
        if status==4:
            completed.add(key)
            if not (REPORTS/(key+'.json')).exists(): updates.append(collect(state))
        else: active.append(key)
    publish(updates)
    write_json(JOBS/'workflow-status.json',dict(time=time.time(),jobs=observations,completed=sorted(completed),active=active))
    if 'prepare' not in states: submit('prepare')
    elif 'prepare' in completed and 'gate' not in states: submit('gate')
    elif 'gate' in completed:
        for i in range(4):
            if f'encode-{i}' not in states: submit('encode','--shard',str(i))
        if all(f'encode-{i}' in completed for i in range(4)):
            for initialization in ['pretrained','random']:
                if 'train-'+initialization not in states: submit('train','--initialization',initialization)
    return 'complete' if all('train-'+i in completed for i in ['pretrained','random']) else 'waiting'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--once',action='store_true'); a=p.parse_args()
    JOBS.mkdir(exist_ok=True)
    with (JOBS/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                status=tick(); print(time.strftime('%FT%TZ',time.gmtime()),status,flush=True)
            except Exception as error:
                write_json(JOBS/'workflow-error.json',dict(time=time.time(),error=str(error)))
                (JOBS/'STOPPED').write_text('Controller stopped on failure; inspect before resuming.\n')
                raise
            if a.once or status in ['complete','stopped']: return
            time.sleep(45)


if __name__=='__main__': main()
