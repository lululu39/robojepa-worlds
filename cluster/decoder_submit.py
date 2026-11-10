"""Submit one bounded decoder integration gate, without a formal run or W&B."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from robojepa_worlds.common import ROOT, read_json, write_json
from .submit import JOBS, STAGING, IMAGE, command, pack_source, validate_inputs, changed_inputs

GATE_ROOT = JOBS/'decoder/gate'
CACHE_AND_PREDICTOR_CODE = [
    *['robojepa_worlds/'+n+'.py' for n in ['model','data','encode','reference','capsaicin','checkpoint','common']],
    'vendor/robo_jepa','vendor/vjepa2','vendor/micro_world','pyproject.toml','uv.lock',
    'experiments/capsaicin/pilot.json','experiments/capsaicin/dataset-manifest.json',
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-prepared',action='store_true')
    args=parser.parse_args()
    GATE_ROOT.mkdir(parents=True,exist_ok=True)
    with (GATE_ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOBS/'STOPPED').exists() or (GATE_ROOT/'STOPPED').exists(): raise RuntimeError('Stopped')
        if (GATE_ROOT/'state.json').exists():
            print(json.dumps(read_json(GATE_ROOT/'state.json'))); return
        if (GATE_ROOT/'intent.json').exists(): raise RuntimeError('Prior intent; inspect before submitting')
        active=json.loads(command(['condor_q','yzhong239','-constraint','RoboJepaKey == "capsaicin-decoder-gate"','-json']) or '[]')
        if active: raise RuntimeError('Existing decoder gate in queue')
        if command(['git','status','--porcelain']): raise RuntimeError('Commit reviewed source first')
        commit=command(['git','rev-parse','HEAD'])
        gate=read_json(ROOT/'experiments/capsaicin/reports/gate.json')
        validate_inputs('train',gate['source']['commit'])
        if changed_inputs(gate['source']['commit'],commit,CACHE_AND_PREDICTOR_CODE):
            raise RuntimeError('Existing predictor/cache contract changed')
        training=read_json(ROOT/'experiments/capsaicin/reports/train.json')
        cfg=read_json(ROOT/'experiments/capsaicin/decoder-gate.json')
        if not training['complete'] or training['selected_step']!=cfg['predictor_selected_step'] or cfg['formal']:
            raise RuntimeError('Completed selected predictor and diagnostic-only configuration required')
        checkpoint=JOBS/'train/checkpoints.tar'
        if not checkpoint.is_file() or not checkpoint.stat().st_size: raise RuntimeError('No persisted predictor')
        (GATE_ROOT/'logs').mkdir(exist_ok=True)
        source=pack_source(GATE_ROOT/'source.tar')
        shutil.copy2(ROOT/'cluster/decoder_worker.sh',GATE_ROOT/'worker.sh')
        os.chmod(GATE_ROOT/'worker.sh',0o755)
        inputs=[str(GATE_ROOT/'source.tar'),str(checkpoint)]
        prepared={}
        runtime_inputs=['runtime.tar']
        outputs=['results.tar'];remaps=[f'results.tar={GATE_ROOT}/results.tar']
        if args.reuse_prepared:
            runtime_inputs=['decoder-runtime.tar','decoder-assets.tar']
            for name in runtime_inputs:
                prepared[name]=(STAGING/name).stat().st_size
                if prepared[name]<=0:raise RuntimeError('Empty prepared artifact: '+name)
            with tarfile.open(STAGING/'decoder-assets.tar','r:') as archive:
                meta=json.load(archive.extractfile('assets/assets.json'))
                if not meta['complete'] or meta['checkpoint']!=cfg['checkpoint'] or meta['vgg']!=cfg['vgg']:
                    raise RuntimeError('Prepared asset metadata mismatch')
                for prefix,key in [('assets/','checkpoint'),('assets/torch/hub/checkpoints/','vgg')]:
                    member=archive.getmember(prefix+cfg[key]['filename'])
                    if not member.isfile() or member.size!=cfg[key]['bytes']:raise RuntimeError('Prepared asset size mismatch')
            with tarfile.open(STAGING/'decoder-runtime.tar','r:') as archive:
                for name in ['pyproject.toml','uv.lock']:
                    path='experiments/capsaicin/decoder_env/'+name
                    if archive.extractfile(path).read()!=(ROOT/path).read_bytes():
                        raise RuntimeError('Prepared decoder environment changed')
        else:
            outputs+=['decoder-assets.tar','decoder-runtime.tar']
            remaps+=[f'decoder-assets.tar={STAGING}/decoder-assets.tar',f'decoder-runtime.tar={STAGING}/decoder-runtime.tar']
        inputs+=['osdf:///chtc'+str(STAGING/name) for name in [*runtime_inputs,'capsaicin-data.tar','robojepa-weights.tar',*[f'features-{i}.tar' for i in range(4)]]]
        lines=['universe = vanilla',f'container_image = {IMAGE}','executable = worker.sh',
               'should_transfer_files = YES','when_to_transfer_output = ON_EXIT','stream_output = True','stream_error = True',
               'requirements = (HasCHTCStaging == true)','request_cpus = 4','request_memory = 16GB','request_disk = 100GB',
               'request_gpus = 1','gpus_minimum_memory = 76000M',
               'require_gpus = regexp("H100|H200",DeviceName) && !regexp("MIG",DeviceName) && (GlobalMemoryMb >= 76000)',
               'rank = 100*ifThenElse(regexp("H100|H200",TARGET.GPUs_DeviceName),1.0,0.0) + ifThenElse(regexp("^backfill",TARGET.Name),0.0,1.0)',
               '+WantGPULab = true','+GPUJobLength = "short"',
               '+JobBatchName = "robojepa-capsaicin-decoder-gate"','+RoboJepaKey = "capsaicin-decoder-gate"',
               'transfer_input_files = '+','.join(inputs),'transfer_output_files = '+','.join(outputs),
               'transfer_output_remaps = "'+';'.join(remaps)+'"','log = logs/job.log','output = logs/job.out','error = logs/job.err',
               'hold = true','queue 1']
        (GATE_ROOT/'job.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dryrun.ad','job.sub'],GATE_ROOT)
        state={'key':'decoder-gate','stage':'decoder-gate','source':source,'formal':False,'run_id':None,
               'prepared_artifacts':prepared,'submitted_at':time.time()}
        write_json(GATE_ROOT/'intent.json',state)
        job=command(['condor_submit','-terse','job.sub'],GATE_ROOT).split()[0]
        state.update(job=job,status='held');write_json(GATE_ROOT/'state.json',state)
        ad=json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestMemory']==16384 and ad['RequestDisk']==104857600
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and '.netrc' not in ad['TransferInput']
        assert set(ad['TransferOutput'].split(','))==set(outputs)
        write_json(GATE_ROOT/'verified.json',{k:ad.get(k) for k in ['ClusterId','ProcId','JobStatus','RequestCpus','RequestMemory','RequestDisk','RequestGPUs','RequireGPUs','TransferInput','TransferOutput','ContainerImage']})
        command(['condor_release',job]);state['status']='queued';write_json(GATE_ROOT/'state.json',state)
        print(json.dumps(state,indent=2))


if __name__=='__main__': main()
