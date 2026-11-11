"""Frozen RGB/action and long-rollout evaluation; no training or W&B."""
import argparse
from fractions import Fraction
import math
from pathlib import Path
import time

import av
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .common import ROOT, read_json, write_json
from .control_protocol import CONTROL_NAMES, controls, training_yaw
from .decoder_gate import panels
from .model import NavigationModel
from .rgb_data import RgbFeatureDataset
from .rgb_decoder import RgbDecoder, make_loss, rgb_loss


def video(path, frames, rate):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=rate)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = 'yuv420p'
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def control_panels(rgb, clip_id):
    frames = []
    for step in range(rgb.shape[1]):
        canvas = Image.new('RGB', (960, 416), 'black')
        draw = ImageDraw.Draw(canvas)
        for i, name in enumerate(CONTROL_NAMES):
            x, y = (i % 3)*320, (i // 3)*208
            pixels = (rgb[i, step].clamp(0, 1)*255).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(pixels), (x, y+32))
            draw.text((x+4, y+2), f'{name}; step {step}', fill='white')
            draw.text((x+4, y+17), clip_id[:48], fill='white')
        frames.append(canvas)
    return frames


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ['data', 'features', 'weights', 'assets', 'predictor-checkpoint', 'decoder-checkpoint', 'out']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    cfg = read_json(ROOT/'experiments/capsaicin/control-eval.json')
    decoder_cfg = read_json(ROOT/'experiments/capsaicin/decoder-train.json')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One GPU required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = read_json(Path(args.data)/'index.json')
    yaw = training_yaw(index)
    report = dict(complete=False, formal=False, config=cfg, source=read_json(ROOT/'source.json'),
                  dataset_revision=index['source']['revision'], yaw=yaw,
                  gpu=torch.cuda.get_device_name(), clips=[], limitations=[
                      'Four deterministic held-out anchors; not a full benchmark.',
                      'No simulator ground truth for alternative controls or beyond cached frame 49.',
                      'Counterfactual differences measure sensitivity, not direction correctness.',
                      '100 steps exceed the trained 20-step windows; input history remains 16 states.',
                      'Zero input is not guaranteed to stop physical momentum.'])
    predictor = NavigationModel(args.weights).cuda().eval().requires_grad_(False)
    decoder = RgbDecoder(Path(args.assets)/decoder_cfg['checkpoint']['filename']).cuda().eval().requires_grad_(False)
    for name, model, path, run_id, step in [
        ('predictor', predictor, args.predictor_checkpoint, 'b27dd2b1', 1600),
        ('decoder', decoder, args.decoder_checkpoint, '1e866f14', 5000),
    ]:
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if state['run_id'] != run_id or state['step'] != step:
            raise ValueError(f'Unexpected {name} checkpoint')
        model.load_state_dict(state['model'], strict=True)
        report[name] = dict(run_id=run_id, step=step)
        del state
    loss_fn = make_loss(decoder_cfg).cuda()

    for split in ['val', 'test']:
        dataset = RgbFeatureDataset(args.data, args.features, split, cache_size=1)
        for clip_index in [0, len(dataset.rows)//2]:
            row = dataset.rows[clip_index]
            cached, pixels = dataset.clip(clip_index)
            real = cached['features'][None].cuda()
            commands = controls(cached['actions'].tolist(), cfg['steps'], yaw['degrees_per_source_frame'])
            actions = torch.tensor([commands[n] for n in CONTROL_NAMES], device='cuda')
            history = real[:, :1].expand(len(CONTROL_NAMES), -1, -1, -1, -1, -1).clone()
            latents = [history[:, 0].float().cpu()]
            rgb = []
            torch.cuda.synchronize(); clip_started = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rgb.append(decoder(history)[:, 0].float().cpu())
                # Same 16-state sliding-window recurrence as model.rollout, but retain
                # only 16 GPU states and decode online; CPU outputs are evaluation artifacts.
                for step in range(cfg['steps']):
                    if time.monotonic()-started > cfg['max_compute_seconds']:
                        raise TimeoutError('Bounded evaluation exceeded compute limit')
                    left = max(0, step+1-16)
                    future = predictor(history, actions[:, left:step+1])[:, -1:]
                    if not torch.isfinite(future).all():
                        raise FloatingPointError('Non-finite latent rollout')
                    decoded = decoder(future)[:, 0]
                    if not torch.isfinite(decoded).all():
                        raise FloatingPointError('Non-finite RGB rollout')
                    latents.append(future[:, 0].float().cpu())
                    rgb.append(decoded.float().cpu())
                    history = torch.cat([history, future], dim=1)[:, -16:]
                    if (step+1) % 20 == 0:
                        print(f'CONTROL_PROGRESS {split} {clip_index} step={step+1}', flush=True)
                ceiling = torch.cat([decoder(real[:, i:i+4]).cpu() for i in range(0, 41, 4)], dim=1)[0, :41]
            torch.cuda.synchronize()
            clip_seconds = time.monotonic()-clip_started
            rgb = torch.stack(rgb, dim=1)
            latents = torch.stack(latents, dim=1)
            truth = pixels[:41].float()/255.
            metrics = []
            for horizon in cfg['gt_horizons']:
                target = truth[horizon:horizon+1, None].cuda()
                item = dict(horizon=horizon)
                for label, predicted in [('recorded', rgb[0, horizon:horizon+1]),
                                         ('real_latent', ceiling[horizon:horizon+1]),
                                         ('persistence_rgb', truth[:1])]:
                    prediction = predicted[:, None].cuda()
                    _, values = rgb_loss(loss_fn, prediction, target, 0)
                    mse = float(F.mse_loss(prediction.clamp(0, 1), target))
                    item[label] = dict(**values, clipped_mse=mse, psnr=-10*math.log10(max(mse, 1e-12)))
                target_latent = real[0, horizon].float().cpu()
                item['latent_mae'] = float((latents[0, horizon]-target_latent).abs().mean())
                item['persistence_latent_mae'] = float((latents[0, 0]-target_latent).abs().mean())
                metrics.append(item)
            response = {}
            for i, name in enumerate(CONTROL_NAMES):
                response[name] = {}
                for horizon in cfg['control_horizons']:
                    image = rgb[i, horizon].clamp(0, 1)
                    response[name][str(horizon)] = dict(
                        rgb_mae_vs_zero=float((image-rgb[1, horizon].clamp(0, 1)).abs().mean()),
                        latent_mae_vs_zero=float((latents[i, horizon]-latents[1, horizon]).abs().mean()),
                        adjacent_rgb_mae=float((image-rgb[i, horizon-1].clamp(0, 1)).abs().mean()),
                        rgb_mae_vs_initial=float((image-rgb[i, 0].clamp(0, 1)).abs().mean()),
                        latent_rms=float(latents[i, horizon].square().mean().sqrt()),
                        image_std=float(image.std()),
                        unclipped_out_of_range_fraction=float(((rgb[i, horizon]<0)|(rgb[i, horizon]>1)).float().mean()))
            stem = f'{split}-{clip_index}'
            with av.open(str(dataset.data/row['video'])) as source:
                rate = Fraction(source.streams.video[0].average_rate)/4
            grids = control_panels(rgb, row['id'])
            video(out/(stem+'-controls.mp4'), grids, rate)
            for step in [0, 20, 40, 100]:
                grids[step].save(out/f'{stem}-controls-{step}.png')
            recorded = panels(truth, ceiling, rgb[0, :41], 'Recorded actions; GT / real latent / predicted latent')
            video(out/(stem+'-recorded.mp4'), recorded, rate)
            recorded[40].save(out/(stem+'-recorded-40.png'))
            entry = dict(id=row['id'], scene=row['scene'], split=split, clip_index=clip_index,
                         generation_seconds=clip_seconds, video_fps=float(rate),
                         simulated_duration_seconds=cfg['steps']/float(rate),
                         recorded_action_steps=len(cached['actions']), metrics=metrics,
                         control_response=response, media_prefix=stem)
            report['clips'].append(entry)
            write_json(out/'control-eval.json', report)
            print(f'CONTROL_CLIP_COMPLETE {stem} seconds={clip_seconds:.2f}', flush=True)
            del real, history, future, rgb, latents, ceiling
    report.update(complete=True, elapsed_seconds=time.monotonic()-started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'control-eval.json', report)
    print('CONTROL_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
