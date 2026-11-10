"""Submit held, inspect actual inputs/resources, then release. No retries/sweeps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import uuid

from robojepa_worlds.common import ROOT, read_json, write_json

STAGING=Path('/staging/y/yzhong239/robojepa-worlds/capsaicin')
JOBS=Path('/home/yzhong239/robojepa-capsaicin-jobs')
IMAGE='file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif'
INPUT_CODE=['robojepa_worlds','vendor','uv.lock','pyproject.toml','experiments/capsaicin/pilot.json',
            'experiments/capsaicin/dataset-manifest.json','cluster/worker.sh']


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=90).strip()


def pack_source(target):
    source=dict(commit=command(['git','rev-parse','HEAD']),upstream={
        n:command(['git','rev-parse','HEAD'],ROOT/'vendor'/n) for n in ['robo_jepa','vjepa2','micro_world']})
    files=command(['git','ls-files','--recurse-submodules']).splitlines()
    with tarfile.open(target,'w') as tar:
        for name in files:
            p=ROOT/name
            if p.is_file(): tar.add(p,arcname='source/'+name,recursive=False)
        import io
        payload=json.dumps(source,indent=2).encode()
        info=tarfile.TarInfo('source/source.json'); info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
    return source


def validate_inputs(stage,commit):
    """Check dependencies before matching a GPU; deferred submission does not bypass this."""
    if stage=='prepare': return
    preparation=read_json(ROOT/'experiments/capsaicin/reports/prepare.json')
    if not preparation.get('complete') or preparation.get('schema')!=2:
        raise RuntimeError('Successful schema-2 preparation required')
    if command(['git','diff','--name-only',preparation['source']['commit'],commit,'--',*INPUT_CODE]):
        raise RuntimeError('Preparation source is incompatible with the queued worker')
    for name in ['runtime.tar','capsaicin-data.tar','robojepa-weights.tar']:
        artifact=STAGING/name
        if not artifact.is_file() or artifact.stat().st_size<=0: raise RuntimeError('Missing preparation artifact: '+name)
        expected=preparation.get('artifacts',{}).get(name)
        if expected is not None and artifact.stat().st_size!=expected: raise RuntimeError('Artifact size mismatch: '+name)
    if stage in ['encode','train']:
        gate=read_json(ROOT/'experiments/capsaicin/reports/gate.json')
        if not gate.get('complete'): raise RuntimeError('GPU gate required')
        if command(['git','diff','--name-only',gate['source']['commit'],commit,'--',*INPUT_CODE]):
            raise RuntimeError('Queued worker differs from the GPU-validated implementation')
    if stage=='train':
        for i in range(4):
            if not read_json(ROOT/f'experiments/capsaicin/reports/encode-{i}.json').get('complete'):
                raise RuntimeError('All encoding shards must complete')


def release_deferred(state):
    """Release the same pre-submitted job after successful dependencies; never resubmit."""
    with (JOBS/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOBS/'STOPPED').exists(): raise RuntimeError('Workflow stopped')
        current=read_json(JOBS/state['key']/'state.json')
        if not current.get('waiting_dependencies'): return
        ad=json.loads(command(['condor_q',current['job'],'-json']))[0]
        if not (ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0 and ad.get('HoldReasonCode')==15
                and ad.get('RoboJepaWaitingDependencies') is True):
            raise RuntimeError('Unexpected deferred-job state; do not release')
        validate_inputs(current['stage'],current['source']['commit'])
        if command(['git','diff','--name-only',current['source']['commit'],'HEAD','--',*INPUT_CODE]):
            raise RuntimeError('Implementation changed after submission; review queued source')
        if 'nanochat-uv-v4.sif' not in ad['TransferInput']: raise RuntimeError('Container URI missing')
        current.update(waiting_dependencies=False,status='releasing',dependencies_ready_at=time.time())
        write_json(JOBS/current['key']/'state.json',current)
        command(['condor_qedit',current['job'],'RoboJepaWaitingDependencies','false'])
        command(['condor_release',current['job']])
        current['status']='queued'; write_json(JOBS/current['key']/'state.json',current)
        print('DEPENDENCIES_READY',current['key'],current['job'],flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','gate','encode','train'])
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    p.add_argument('--defer',action='store_true',help='Queue gate/encoding held until successful dependencies')
    a=p.parse_args()
    if a.defer and a.stage not in ['gate','encode']: p.error('--defer supports only gate/encode')
    if a.stage=='train':
        cfg=read_json(ROOT/'experiments/capsaicin/pilot.json')
        if not cfg['formal']: raise RuntimeError('Preflight only: complete the RGB task path before formal training')
    key=a.stage+(f'-{a.shard}' if a.stage=='encode' else '')
    JOBS.mkdir(exist_ok=True); STAGING.mkdir(parents=True,exist_ok=True)
    with (JOBS/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOBS/'STOPPED').exists(): raise RuntimeError('Workflow stopped')
        root=JOBS/key; root.mkdir(exist_ok=True)
        state_path=root/'state.json'
        if state_path.exists():
            print(json.dumps(read_json(state_path))); return
        if (root/'intent.json').exists(): raise RuntimeError('Prior submit intent exists; inspect, do not duplicate')
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'RoboJepaKey == "capsaicin-{key}"','-json']) or '[]')
        if active: raise RuntimeError('Existing marked job; inspect it instead')
        if command(['git','status','--porcelain']): raise RuntimeError('Commit reviewed code before submission')
        dependencies=['prepare'] if a.stage=='gate' else ['prepare','gate'] if a.stage=='encode' else []
        if a.defer:
            for parent in dependencies:
                if not (JOBS/parent/'state.json').exists(): raise RuntimeError('Submit dependency first: '+parent)
        else: validate_inputs(a.stage,command(['git','rev-parse','HEAD']))
        (root/'logs').mkdir(exist_ok=True)
        source=pack_source(root/'source.tar')
        shutil.copy2(ROOT/'cluster/worker.sh',root/'worker.sh'); os.chmod(root/'worker.sh',0o755)
        inputs=[str(root/'source.tar')]
        outputs=['results.tar']; remaps=[f'results.tar={root}/results.tar']
        args=a.stage
        if a.stage=='prepare':
            for name in ['runtime.tar','capsaicin-data.tar','robojepa-weights.tar']:
                outputs.append(name); remaps.append(f'{name}={STAGING}/{name}')
        else:
            for name in ['runtime.tar','capsaicin-data.tar','robojepa-weights.tar']:
                f=STAGING/name
                if not a.defer and (not f.is_file() or not f.stat().st_size):
                    raise RuntimeError('Preparation artifact missing: '+name)
                inputs.append('osdf:///chtc'+str(f))
        if a.stage in ['gate','train']:
            outputs.append('checkpoints.tar'); remaps.append(f'checkpoints.tar={root}/checkpoints.tar')
        if a.stage=='encode':
            args+=f' {a.shard}'; outputs.append('features.tar')
            remaps.append(f'features.tar={STAGING}/features-{a.shard}.tar')
        run_id=None
        if a.stage=='train':
            run_id=uuid.uuid4().hex[:8]; args+=f' {run_id}'
            for i in range(4):
                artifact=STAGING/f'features-{i}.tar'
                if not artifact.is_file() or not artifact.stat().st_size: raise RuntimeError('Missing feature archive')
                inputs.append('osdf:///chtc'+str(artifact))
            inputs.append('/home/yzhong239/.netrc')
        lines=['universe = vanilla',f'container_image = {IMAGE}','executable = worker.sh',f'arguments = {args}',
               'should_transfer_files = YES','when_to_transfer_output = ON_EXIT','stream_output = True','stream_error = True',
               'requirements = (HasCHTCStaging == true)','request_cpus = 4',
               'request_memory = '+('64GB' if a.stage=='prepare' else '48GB'),
               'request_disk = '+('150GB' if a.stage=='prepare' else '160GB' if a.stage=='train' else '100GB'),
               'transfer_input_files = '+','.join(inputs),'transfer_output_files = '+','.join(outputs),
               'transfer_output_remaps = "'+';'.join(remaps)+'"',
               'log = logs/job.log','output = logs/job.out','error = logs/job.err',
               f'+JobBatchName = "robojepa-capsaicin-{key}"',f'+RoboJepaKey = "capsaicin-{key}"',
               'rank = ifThenElse(regexp("^backfill",TARGET.Name),0.0,1.0)','hold = true']
        if a.defer: lines.append('+RoboJepaWaitingDependencies = true')
        if a.stage!='prepare':
            lines+=['request_gpus = 1','gpus_minimum_memory = 76000M',
                    'require_gpus = regexp("H100|H200",DeviceName) && !regexp("MIG",DeviceName) && (GlobalMemoryMb >= 76000)',
                    'rank = 100*ifThenElse(regexp("H100|H200",TARGET.GPUs_DeviceName),1.0,0.0) + ifThenElse(regexp("^backfill",TARGET.Name),0.0,1.0)',
                    '+WantGPULab = true','+GPUJobLength = "short"']
        if a.stage=='train':
            lines+=['checkpoint_exit_code = 85','transfer_checkpoint_files = checkpoints.tar,results.tar',
                    'encrypt_input_files = /home/yzhong239/.netrc','+is_resumable = true']
        lines+=['queue 1']
        (root/'job.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dryrun.ad','job.sub'],root)
        state=dict(stage=a.stage,key=key,source=source,run_id=run_id,submitted_at=time.time(),status='submitting',
                   waiting_dependencies=a.defer,dependencies=dependencies)
        write_json(root/'intent.json',state)
        job=command(['condor_submit','-terse','job.sub'],root).split()[0]
        state.update(job=job,status='held'); write_json(state_path,state)
        ad=json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestCpus']==4
        assert 'nanochat-uv-v4.sif' in ad['TransferInput']
        if a.stage!='prepare': assert ad['RequestGPUs']==1 and 'H100|H200' in ad['RequireGPUs']
        if a.stage=='train':
            assert ad['SuccessCheckpointExitCode']==85 and '.netrc' in ad['EncryptInputFiles']
        write_json(root/'verified.json',{k:ad.get(k) for k in ['ClusterId','ProcId','JobStatus','RequestCpus','RequestMemory','RequestDisk','RequestGPUs','RequireGPUs','TransferInput','TransferOutput','ContainerImage','SuccessCheckpointExitCode','HoldReasonCode','RoboJepaWaitingDependencies']})
        if a.defer:
            assert ad.get('RoboJepaWaitingDependencies') is True and ad.get('HoldReasonCode')==15
            state['status']='waiting_dependencies'
        else:
            command(['condor_release',job]); state['status']='queued'
        write_json(state_path,state)
        print(json.dumps(state,indent=2))


if __name__=='__main__': main()
