"""Publish this authorized public repository, keeping credentials out of Git/process arguments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request


def main():
    p=argparse.ArgumentParser(); p.add_argument('--create',action='store_true'); a=p.parse_args()
    credential=Path('/home/yzhong239/.config/gradformer/github.token')
    if credential.stat().st_mode & 0o777 != 0o600: raise RuntimeError('Credential permissions must be 0600')
    token=credential.read_text().strip()
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'robojepa-worlds'}
    if a.create:
        payload=json.dumps({'name':'robojepa-worlds','private':False,'description':'Single-GPU RoboJEPA adaptation to interactive environments; independent reproducible sub-experiments','auto_init':False}).encode()
        req=urllib.request.Request('https://api.github.com/orgs/lululu39/repos',data=payload,headers=headers,method='POST')
        try:
            r=json.loads(urllib.request.urlopen(req,timeout=30).read()); print(r['html_url'])
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'GitHub create failed: HTTP {e.code}') from None
    root=Path(__file__).resolve().parents[1]
    url='https://github.com/lululu39/robojepa-worlds.git'
    if subprocess.run(['git','remote','get-url','origin'],cwd=root,capture_output=True).returncode:
        subprocess.run(['git','remote','add','origin',url],cwd=root,check=True)
    with tempfile.TemporaryDirectory(prefix='robojepa-askpass-') as td:
        helper=Path(td)/'askpass.py'
        helper.write_text('#!/usr/bin/python\nimport os,sys\nfrom pathlib import Path\nprint("x-access-token" if "Username" in sys.argv[1] else Path(os.environ["ROBO_GITHUB_TOKEN_FILE"]).read_text().strip())\n')
        helper.chmod(0o700)
        env={**os.environ,'GIT_ASKPASS':str(helper),'GIT_TERMINAL_PROMPT':'0','ROBO_GITHUB_TOKEN_FILE':str(credential)}
        subprocess.run(['git','push','-u','origin','main'],cwd=root,env=env,check=True)


if __name__=='__main__': main()
