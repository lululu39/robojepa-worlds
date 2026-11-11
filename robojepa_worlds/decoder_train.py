"""Full-data pretrained RGB decoder adaptation; one resumable formal run."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import time

import av
import numpy as np
import torch
from torch.nn import functional as F

from . import checkpoint
from .common import ROOT, read_json, write_json
from .decoder_gate import panels
from .rgb_data import RgbFeatureDataset
from .rgb_decoder import RgbDecoder, make_loss, make_optimizer, rgb_loss
from .rgb_schedule import FrameSchedule, lr_scale


@torch.no_grad()
def evaluate_rgb(model, loss_fn, dataset, batch_size, preview=None):
    model.eval()
    totals = dict(loss=0., mse=0., lpips=0., clipped_mse=0.)
    count = 0
    for clip, row in enumerate(dataset.rows):
        for start in range(0, len(row['indices']), batch_size):
            indices = list(range(start, min(start+batch_size, len(row['indices']))))
            x, y = dataset.batch(clip, indices)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(x)
            loss, metrics = rgb_loss(loss_fn, prediction, y, 0)
            values = dict(loss=float(loss), **metrics, clipped_mse=float(F.mse_loss(prediction.clamp(0, 1), y)))
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError('Non-finite reconstruction evaluation')
            for key, value in values.items():
                totals[key] += value * len(indices)
            count += len(indices)
            if preview is not None and clip == start == 0:
                panels(y[:, 0], prediction[:, 0], title='Held-out RGB reconstruction')[0].save(preview)
    if not count:
        raise ValueError('Empty RGB evaluation split')
    result = {k: v/count for k, v in totals.items()}
    result.update(frames=count, clips=len(dataset.rows), psnr_clipped=-10*math.log10(max(result['clipped_mse'], 1e-12)))
    return result


@torch.inference_mode()
def export_rollout(model, dataset, weights, predictor_checkpoint, out, selected_step):
    from .model import NavigationModel, rollout
    predictor = NavigationModel(weights).cuda().eval().requires_grad_(False)
    state = torch.load(predictor_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if state['run_id'] != 'b27dd2b1' or state['step'] != 1600:
        raise ValueError('Unexpected selected frozen predictor')
    predictor.load_state_dict(state['model'], strict=True)
    del state
    cached, pixels = dataset.clip(0)
    features = cached['features'][:21][None].cuda()
    actions = cached['actions'][:20][None].cuda()
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        future = rollout(predictor, features[:, :1], actions, 20)
        predicted = torch.cat([features[:, :1], future], dim=1)
        real_rgb = torch.cat([model(features[:, i:i+1]) for i in range(21)], dim=1)
        pred_rgb = torch.cat([model(predicted[:, i:i+1]) for i in range(21)], dim=1)
    frames = panels(pixels[:21].float()/255., real_rgb[0], pred_rgb[0],
                    f'Decoder step {selected_step}; frozen predictor step 1600; held-out scene')
    frames[10].save(out/'heldout-rollout.png')
    with av.open(str(dataset.data/dataset.rows[0]['video'])) as source:
        rate = source.streams.video[0].average_rate
    with av.open(str(out/'heldout-rollout.mp4'), 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(rate)/4)
        stream.width, stream.height = frames[0].size; stream.pix_fmt = 'yuv420p'
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    parser = argparse.ArgumentParser()
    for name in ['data', 'features', 'assets', 'weights', 'predictor-checkpoint', 'out', 'checkpoints', 'run-id']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    cfg = read_json(ROOT/'experiments/capsaicin/decoder-train.json')
    if not cfg['formal'] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One-GPU formal decoder configuration required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(cfg['deterministic_algorithms'])
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    source = read_json(ROOT/'source.json'); index = read_json(Path(args.data)/'index.json')
    cfg.update(source=source, dataset_revision=index['source']['revision'])
    out, ckpt = Path(args.out), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    train = RgbFeatureDataset(args.data, args.features, 'train', cache_size=cfg['clip_group_size'])
    val = RgbFeatureDataset(args.data, args.features, 'val')
    schedule = FrameSchedule([len(r['indices']) for r in train.rows], cfg['batch_size'], cfg['clip_group_size'], cfg['seed'])
    model = RgbDecoder(Path(args.assets)/cfg['checkpoint']['filename'], (cfg['height'], cfg['width']), cfg['seed']).cuda()
    loss_fn = make_loss(cfg).cuda()
    step = position = frames_seen = 0
    previous_seconds = 0.; best = None; joint = False
    if (ckpt/'latest.pt').exists():
        meta = torch.load(ckpt/'latest.pt', map_location='cpu', mmap=True, weights_only=False)
        joint = meta['joint']; del meta
    optimizer = make_optimizer(model, cfg, joint=joint)
    if (ckpt/'latest.pt').exists():
        restored = checkpoint.restore(ckpt/'latest.pt', model, optimizer, args.run_id, cfg)
        step, position = restored['step'], restored['data_position']
        frames_seen, previous_seconds, best = restored['frames_seen'], restored['train_seconds'], restored['best_val']
    import wandb
    options = dict(entity=cfg['wandb_entity'], project=cfg['wandb_project'], id=args.run_id,
                   name='pretrained-jepawms-rgb-adaptation', job_type='decoder-train', config={**cfg,
                   'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'load_report':model.load_report,
                   'training_clips':len(train.rows), 'training_frames':sum(schedule.counts)}, dir=str(out))
    try:
        run = wandb.init(**options, resume='allow', settings=wandb.Settings(init_timeout=90))
    except wandb.errors.CommError:
        print('WANDB_OFFLINE_FALLBACK', args.run_id, flush=True)
        run = wandb.init(**options, mode='offline')
        write_json(out/'wandb-offline.json', dict(run_id=args.run_id, needs_sync=True))
    begin = time.monotonic()

    def elapsed():
        return previous_seconds + time.monotonic()-begin

    def save(name='latest.pt'):
        checkpoint.save(ckpt/name, model, optimizer, run_id=args.run_id, config=cfg, step=step,
                        data_position=position, frames_seen=frames_seen, train_seconds=elapsed(), best_val=best, joint=joint)

    def log(values):
        with (out/'metrics.jsonl').open('a') as file:
            file.write(json.dumps({'step':step, **values}, allow_nan=False)+'\n')
        run.log(values, step=step)
        print(json.dumps({'step':step, **values}, allow_nan=False), flush=True)

    write_json(out/'run.json', dict(run_id=args.run_id, config=cfg, complete=False, gpu=torch.cuda.get_device_name()))
    if step == 0:
        baseline = evaluate_rgb(model, loss_fn, val, cfg['batch_size'], out/'baseline-reconstruction.png')
        log({'baseline/dev/'+k:v for k, v in baseline.items()})
    print('FORMAL_DECODER_START', args.run_id, 'resume_step', step, flush=True)
    while step < cfg['max_steps'] and elapsed() < cfg['max_train_seconds']:
        started = time.monotonic()
        if not joint and step >= cfg['adapter_warmup_steps']:
            joint = True
            optimizer = make_optimizer(model, cfg, joint=True)
        scale = lr_scale(step, cfg)
        for group_index, group in enumerate(optimizer.param_groups):
            group['lr'] = (cfg['adapter_lr'] if group_index == 0 else cfg['decoder_lr']) * scale
        plan = schedule.at(position); count = sum(len(indices) for _, indices in plan)
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals = dict(loss=0., mse=0., lpips=0.)
        for clip, indices in plan:
            x, y = train.batch(clip, indices)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(x)
            value, metrics = rgb_loss(loss_fn, prediction, y, step)
            if not torch.isfinite(value):
                raise FloatingPointError('Non-finite decoder training loss')
            weight = len(indices)/count
            (value*weight).backward()
            for key, number in dict(loss=float(value.detach()), **metrics).items():
                totals[key] += number*weight
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        if gradient <= 0:
            raise RuntimeError('Decoder update has no gradient')
        optimizer.step(); step += 1; position += 1; frames_seen += count
        torch.cuda.synchronize()
        log({**{'train/'+k:v for k, v in totals.items()}, 'train/grad_norm':float(gradient),
             'train/joint':int(joint), 'train/adapter_lr':optimizer.param_groups[0]['lr'],
             'train/decoder_lr':optimizer.param_groups[1]['lr'] if joint else 0.,
             'train/epoch':schedule.epoch, 'train/frames_seen':frames_seen, 'train/data_position':position,
             'train/seconds_per_step':time.monotonic()-started, 'train/elapsed_seconds':elapsed(),
             'gpu/peak_memory_bytes':torch.cuda.max_memory_allocated()})
        if step % cfg['eval_every'] == 0:
            metrics = evaluate_rgb(model, loss_fn, val, cfg['batch_size'], out/'dev-reconstruction.png')
            log({'dev/'+k:v for k, v in metrics.items()})
            run.log({'dev/reconstruction':wandb.Image(str(out/'dev-reconstruction.png'))}, step=step)
            if best is None or metrics['loss'] < best:
                best = metrics['loss']; save('best.pt')
        if step % cfg['save_every'] == 0:
            save()
        if time.monotonic()-begin >= cfg['segment_seconds'] and step < cfg['max_steps'] and elapsed() < cfg['max_train_seconds']:
            save(); write_json(out/'status.json', dict(status='checkpoint', step=step, run_id=args.run_id, train_seconds=elapsed()))
            run.finish(); raise SystemExit(85)
    # Include the final step in development selection if the time budget ends between evaluations.
    if step % cfg['eval_every'] != 0 or best is None:
        metrics = evaluate_rgb(model, loss_fn, val, cfg['batch_size'])
        log({'dev/'+k:v for k, v in metrics.items()})
        if best is None or metrics['loss'] < best:
            best = metrics['loss']; save('best.pt')
    save()
    selected = torch.load(ckpt/'best.pt', map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(selected['model'], strict=True); selected_step = selected['step']; del selected
    final = {}
    for split, dataset in [('dev', val), ('test', RgbFeatureDataset(args.data, args.features, 'test'))]:
        final.update({split+'/'+k:v for k, v in evaluate_rgb(model, loss_fn, dataset, cfg['batch_size']).items()})
    final['evaluation/decoder_step'] = selected_step
    log(final)
    export_rollout(model, val, args.weights, args.predictor_checkpoint, out, selected_step)
    run.log({'evaluation/rollout':wandb.Video(str(out/'heldout-rollout.mp4'), format='mp4')}, step=step)
    result = dict(complete=True, run_id=args.run_id, step=step, frames_seen=frames_seen,
                  selected_step=selected_step, train_seconds=elapsed(), metrics=final, config=cfg,
                  stop_reason='steps' if step >= cfg['max_steps'] else 'time_budget')
    write_json(out/'completed.json', result)
    run.finish(); print('FORMAL_DECODER_COMPLETE', args.run_id, step, flush=True)


if __name__ == '__main__':
    main()
