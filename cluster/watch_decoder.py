"""Collect this decoder gate only; no submissions, retries or formal training."""
import fcntl
import json
import re
import tarfile
import time

from robojepa_worlds.common import ROOT,read_json,write_json
from .decoder_submit import GATE_ROOT
from .submit import JOBS,STAGING
from .workflow import query,publish


def main():
    with (GATE_ROOT/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (JOBS/'STOPPED').exists() or (GATE_ROOT/'STOPPED').exists(): return
            try:
                state=read_json(GATE_ROOT/'state.json');status,ad=query(state)
                write_json(GATE_ROOT/'watch-status.json',{'time':time.time(),'job':state['job'],'job_status':status,'remote_host':ad.get('RemoteHost')})
                if status==4:
                    with tarfile.open(GATE_ROOT/'results.tar','r:') as archive:
                        item=archive.getmember('results/decoder-gate.json')
                        if item.size>2_000_000 or not item.isfile():raise ValueError('Invalid report member')
                        report=json.load(archive.extractfile(item))
                        if report.get('complete') is not True:raise RuntimeError('Incomplete GPU gate')
                        env=archive.extractfile('results/environment.txt').read(100_000).decode().splitlines()
                        if not all(re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',line) for line in env):raise ValueError('Invalid dependency report')
                    report.update(job_id=state['job'],installed_dependencies=env,
                                  artifacts={n:(STAGING/n).stat().st_size for n in ['decoder-assets.tar','decoder-runtime.tar']})
                    target=ROOT/'experiments/capsaicin/reports/decoder-gate.json';write_json(target,report)
                    publish([target])
                    write_json(GATE_ROOT/'watch-status.json',{'time':time.time(),'job':state['job'],'status':'complete'})
                    print('DECODER_GATE_COLLECTED',state['job'],flush=True);return
                print(time.strftime('%FT%TZ',time.gmtime()),'waiting',status,flush=True)
            except Exception as error:
                write_json(GATE_ROOT/'watch-status.json',{'time':time.time(),'status':'needs_attention','error':str(error)})
                (GATE_ROOT/'STOPPED').write_text('Failed; diagnose before retrying.\n')
                raise
            time.sleep(60)


if __name__=='__main__': main()
