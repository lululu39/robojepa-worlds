"""Submit exactly one reviewed formal decoder run, using existing prepared inputs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import uuid

from robojepa_worlds.common import ROOT, read_json, write_json
from .submit import JOBS, STAGING, IMAGE, command, pack_source
from .decoder_submit import CACHE_AND_PREDICTOR_CODE

TRAIN_ROOT = JOBS/'decoder/train'


def main():
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    with (TRAIN_ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOBS/'STOPPED').exists() or (TRAIN_ROOT/'STOPPED').exists():
            raise RuntimeError('Formal decoder training stopped')
        if (TRAIN_ROOT/'state.json').exists():
            print(json.dumps(read_json(TRAIN_ROOT/'state.json'))); return
        if (TRAIN_ROOT/'intent.json').exists():
            raise RuntimeError('Existing submission intent; inspect before proceeding')
        active = json.loads(command(['condor_q', 'yzhong239', '-constraint', 'RoboJepaKey == "capsaicin-decoder-train"', '-json']) or '[]')
        if active or command(['git', 'status', '--porcelain']):
            raise RuntimeError('Existing decoder run or uncommitted source')
        gate = read_json(ROOT/'experiments/capsaicin/reports/decoder-gate.json')
        cfg = read_json(ROOT/'experiments/capsaicin/decoder-train.json')
        if not gate['complete'] or not gate['recovery']['parameters']['exact'] or not cfg['formal']:
            raise RuntimeError('Passing decoder gate and explicit formal configuration required')
        for key in ['checkpoint','vgg','height','width','input_dim','decoder_input_dim','deterministic_algorithms','pixel_weight','lpips_weight']:
            if cfg[key] != gate['config'][key]:
                raise RuntimeError('Decoder numerical contract changed: '+key)
        core = CACHE_AND_PREDICTOR_CODE + ['robojepa_worlds/rgb_decoder.py', 'experiments/capsaicin/decoder_env/pyproject.toml', 'experiments/capsaicin/decoder_env/uv.lock']
        if command(['git','diff','--name-only',gate['source']['commit'],'HEAD','--',*core]):
            raise RuntimeError('Validated model/cache/runtime changed')
        # Completed preparation reports already validate archive layouts; check sizes only here.
        expected = {**read_json(ROOT/'experiments/capsaicin/reports/prepare.json')['artifacts']}
        expected['robojepa-weights.tar'] = read_json(ROOT/'experiments/capsaicin/reports/gate.json')['compact_weights_archive_bytes']
        prior = read_json(ROOT/'experiments/capsaicin/reports/decoder-attempt-1-failure.json')
        expected.update(prior['prepared_artifacts'])
        for i in range(4):
            report = read_json(ROOT/f'experiments/capsaicin/reports/encode-{i}.json')
            if not report['complete']:
                raise RuntimeError('Incomplete latent cache')
            name = f'features-{i}.tar'
            size = (STAGING/name).stat().st_size
            if not report['bytes'] <= size <= report['bytes'] + report['clips']*4096 + 10240:
                raise RuntimeError('Latent archive size inconsistent with validated tensors')
            expected[name] = size
        names = ['decoder-runtime.tar','decoder-assets.tar','capsaicin-data.tar','robojepa-weights.tar',*[f'features-{i}.tar' for i in range(4)]]
        for name in names:
            if (STAGING/name).stat().st_size != expected[name]:
                raise RuntimeError('Input size mismatch: '+name)
        credential = Path('/home/yzhong239/.netrc')
        if credential.stat().st_mode & 0o777 != 0o600:
            raise RuntimeError('Credential permissions must be 0600')
        (TRAIN_ROOT/'logs').mkdir(exist_ok=True)
        source = pack_source(TRAIN_ROOT/'source.tar')
        shutil.copy2(ROOT/'cluster/decoder_train_worker.sh', TRAIN_ROOT/'worker.sh')
        os.chmod(TRAIN_ROOT/'worker.sh', 0o755)
        # The selected predictor archive is read-only and distinct from decoder recovery state.
        predictor = TRAIN_ROOT/'predictor-checkpoints.tar'
        if not predictor.exists():
            predictor.symlink_to(JOBS/'train/checkpoints.tar')
        with tarfile.open(predictor, 'r:') as archive:
            item = archive.getmember('checkpoints/best.pt')
            if not item.isfile() or item.size <= 0:
                raise RuntimeError('Missing selected predictor')
        run_id = uuid.uuid4().hex[:8]
        checkpoint_archive = STAGING/f'decoder-train-{run_id}-checkpoints.tar'
        inputs = [str(TRAIN_ROOT/'source.tar'), str(predictor), str(credential)]
        inputs += ['osdf:///chtc'+str(STAGING/name) for name in names]
        lines = ['universe = vanilla', f'container_image = {IMAGE}', 'executable = worker.sh', f'arguments = {run_id}',
                 'should_transfer_files = YES', 'when_to_transfer_output = ON_EXIT', 'stream_output = True', 'stream_error = True',
                 'requirements = (HasCHTCStaging == true)', 'request_cpus = 4', 'request_memory = 16GB', 'request_disk = 100GB',
                 'request_gpus = 1', 'gpus_minimum_memory = 76000M',
                 'require_gpus = regexp("H100|H200",DeviceName) && !regexp("MIG",DeviceName) && (GlobalMemoryMb >= 76000)',
                 'rank = 100*ifThenElse(regexp("H100|H200",TARGET.GPUs_DeviceName),1.0,0.0) + ifThenElse(regexp("^backfill",TARGET.Name),0.0,1.0)',
                 '+WantGPULab = true', '+GPUJobLength = "short"', '+JobBatchName = "robojepa-capsaicin-decoder-train"',
                 '+RoboJepaKey = "capsaicin-decoder-train"', 'transfer_input_files = '+','.join(inputs),
                 'transfer_output_files = results.tar,checkpoints.tar',
                 f'transfer_output_remaps = "results.tar={TRAIN_ROOT}/results.tar;checkpoints.tar={checkpoint_archive}"',
                 'checkpoint_exit_code = 85', 'transfer_checkpoint_files = checkpoints.tar,results.tar',
                 'encrypt_input_files = /home/yzhong239/.netrc', '+is_resumable = true',
                 'log = logs/job.log', 'output = logs/job.out', 'error = logs/job.err', 'hold = true', 'queue 1']
        (TRAIN_ROOT/'job.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dryrun.ad','job.sub'], TRAIN_ROOT)
        state = dict(stage='decoder-train', source=source, run_id=run_id, formal=True, submitted_at=time.time(), config=cfg,
                     checkpoint_archive=str(checkpoint_archive))
        write_json(TRAIN_ROOT/'intent.json', state)
        job = command(['condor_submit','-terse','job.sub'], TRAIN_ROOT).split()[0]
        state.update(job=job, status='held'); write_json(TRAIN_ROOT/'state.json', state)
        ad = json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0 and ad['RequestGPUs']==1
        assert ad['RequestMemory']==16384 and ad['RequestDisk']==104857600 and ad['SuccessCheckpointExitCode']==85
        assert '.netrc' in ad['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        assert set(ad['TransferOutput'].split(','))=={'results.tar','checkpoints.tar'}
        write_json(TRAIN_ROOT/'verified.json', {k:ad.get(k) for k in ['ClusterId','ProcId','JobStatus','RequestCpus','RequestMemory','RequestDisk','RequestGPUs','RequireGPUs','TransferInput','TransferOutput','ContainerImage','SuccessCheckpointExitCode']})
        command(['condor_release',job]); state['status']='queued'; write_json(TRAIN_ROOT/'state.json',state)
        print(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()
