"""Validate the reused runtime, including decord's documented wheel-tag error."""
import importlib.metadata
import io
import json
import platform
import subprocess
import sys


def known_decord_tag_error(returncode, output, wheel, version, machine):
    # https://github.com/dmlc/decord/issues/356
    # Accept only this one metadata defect; all other conflicts remain fatal.
    lines = [line.strip() for line in output.splitlines()]
    diagnostics = [line for line in lines if line and not line.startswith('Checked ')]
    return (returncode == 1 and version == '0.6.0' and machine == 'x86_64'
            and diagnostics == ['Found 1 incompatibility',
                                'The package `decord` was built for a different platform']
            and [line for line in wheel.splitlines() if line.startswith('Tag: ')]
            == ['Tag: cp36-cp36m-manylinux2010_x86_64'])


def main():
    result = subprocess.run(['uv', 'pip', 'check', '--python', sys.executable],
                            capture_output=True, text=True, check=False)
    package = importlib.metadata.distribution('decord')
    accepted = known_decord_tag_error(result.returncode, result.stdout + result.stderr,
                                     package.read_text('WHEEL') or '', package.version, platform.machine())
    if result.returncode and not accepted:
        raise RuntimeError('Runtime dependency check failed: ' + result.stdout + result.stderr)
    # Load the native decoder and decode actual video, rather than relying on its tag.
    import av
    import decord
    import numpy as np
    buffer = io.BytesIO()
    with av.open(buffer, 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for level in [32, 224]:
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), level, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reader = decord.VideoReader(io.BytesIO(buffer.getvalue()), num_threads=1)
    frames = reader.get_batch([0, 1]).asnumpy()
    if len(reader) != 2 or frames.shape != (2, 32, 32, 3) or frames[1].mean() - frames[0].mean() < 180:
        raise RuntimeError('Native decord video smoke test failed')
    print('DECODER_ENVIRONMENT_CHECK', json.dumps({'decord_version': package.version,
          'known_wheel_tag_defect': accepted, 'native_video_decode': True}), flush=True)


if __name__ == '__main__':
    main()
