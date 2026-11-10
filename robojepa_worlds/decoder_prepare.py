"""Stream only the two public decoder assets; no checkpoint deserialization."""
import argparse
import os
from pathlib import Path
import urllib.request

from .common import ROOT, read_json, write_json


def download(url, destination, size):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError('Existing asset has an unexpected size: ' + destination.name)
        return
    tmp = destination.with_suffix(destination.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as source, tmp.open('wb') as output:
        while chunk := source.read(8 * 1024 * 1024):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if tmp.stat().st_size != size:
        raise ValueError('Downloaded asset size mismatch: ' + destination.name)
    tmp.replace(destination)
    print('DECODER_ASSET_READY', destination.name, size, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    root = Path(args.out); cfg = read_json(ROOT / 'experiments/capsaicin/decoder-gate.json')
    ckpt = cfg['checkpoint']; vgg = cfg['vgg']
    url = f"https://huggingface.co/{ckpt['repo_id']}/resolve/{ckpt['revision']}/{ckpt['filename']}"
    download(url, root / ckpt['filename'], ckpt['bytes'])
    download(vgg['url'], root / 'torch/hub/checkpoints' / vgg['filename'], vgg['bytes'])
    write_json(root / 'assets.json', {'complete': True, 'checkpoint': ckpt, 'vgg': vgg})


if __name__ == '__main__': main()
