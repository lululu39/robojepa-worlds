"""Monitor the single authorized decoder run until its first finite update."""
import fcntl
import json
import math
import time

from robojepa_worlds.common import read_json, write_json
from .decoder_train_submit import TRAIN_ROOT
from .submit import JOBS, command


def first_update(text, run_id):
    started = False; resume_step = 0
    for line in text.splitlines():
        if line.startswith('FORMAL_DECODER_START '):
            parts = line.split()
            started = len(parts)==4 and parts[1]==run_id and parts[2]=='resume_step'
            if started:
                resume_step = int(parts[3])
        elif started and line.startswith('{'):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'train/loss' not in item:
                continue
            if not all(isinstance(item.get(k), (int,float)) and math.isfinite(item[k])
                       for k in ['train/loss','train/mse','train/lpips','train/grad_norm']):
                raise RuntimeError('Invalid formal decoder metrics')
            if item.get('step',0)>resume_step:
                return item
    return None


def main():
    with (TRAIN_ROOT/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        missing = 0
        while True:
            if (JOBS/'STOPPED').exists() or (TRAIN_ROOT/'STOPPED').exists():
                return
            state = read_json(TRAIN_ROOT/'state.json')
            try:
                ads = json.loads(command(['condor_q',state['job'],'-json']) or '[]')
                if not ads:
                    ads = json.loads(command(['condor_history',state['job'],'-limit','1','-json']) or '[]')
                if not ads:
                    missing += 1
                    if missing >= 3:
                        raise RuntimeError('Job missing from queue and history for three observations')
                    time.sleep(30); continue
                missing = 0; ad = ads[0]; status = ad['JobStatus']
                if status in [3,5] or (status==4 and (ad.get('ExitBySignal') or ad.get('ExitCode')!=0)):
                    raise RuntimeError(f"Decoder job failed/held/removed: status={status}, exit={ad.get('ExitCode')}")
                path = TRAIN_ROOT/'logs/job.out'
                first = first_update(path.read_text(errors='replace') if path.exists() else '', state['run_id'])
                observation = dict(time=time.time(), job=state['job'], run_id=state['run_id'], job_status=status,
                                   remote_host=ad.get('RemoteHost'), starts=ad.get('NumJobStarts',0))
                if first:
                    write_json(TRAIN_ROOT/'first-update.json', {**observation,'status':'formal_training_started','metrics':first})
                    print('FORMAL_DECODER_OBSERVED', json.dumps(first), flush=True); return
                if status==4:
                    raise RuntimeError('Job completed without a finite formal decoder update')
                if status==1 and time.time()-ad['EnteredCurrentStatus']>=3600 and not state.get('a100_fallback'):
                    requirement='regexp("H100|H200|A100",DeviceName) && !regexp("MIG",DeviceName) && (GlobalMemoryMb >= 76000)'
                    command(['condor_qedit',state['job'],'RequireGPUs',requirement])
                    state['a100_fallback']=True;write_json(TRAIN_ROOT/'state.json',state)
                write_json(TRAIN_ROOT/'watch-status.json',{**observation,'status':'waiting_for_first_update'})
            except Exception as error:
                write_json(TRAIN_ROOT/'watch-status.json',dict(time=time.time(),status='needs_attention',error=str(error)))
                (TRAIN_ROOT/'STOPPED').write_text('Failure: inspect in an interactive debug allocation before further changes.\n')
                raise
            time.sleep(60)


if __name__ == '__main__':
    main()
