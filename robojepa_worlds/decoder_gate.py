"""Bounded decoder integration/learning checks; no W&B and no reusable trained weights."""
import argparse
from fractions import Fraction
import gc
import json
import random
import tempfile
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from . import checkpoint
from .common import ROOT, read_json, write_json
from .model import NavigationModel, rollout
from .rgb_data import RgbFeatureDataset
from .rgb_decoder import RgbDecoder, make_loss, make_optimizer, rgb_loss, unpatchify
from .decoder_recovery import state_difference


def panels(target, reconstructed, predicted=None, title='Diagnostic decoder short fit'):
    groups = [target, reconstructed] + ([] if predicted is None else [predicted])
    labels = ['Ground truth', 'Decoded real latent', 'Decoded predicted latent']
    images = []
    for frame in range(len(target)):
        canvas = Image.new('RGB', (320 * len(groups), 208), 'black')
        draw = ImageDraw.Draw(canvas); draw.text((4, 2), title, fill='white')
        for col, group in enumerate(groups):
            pixels = (group[frame].detach().float().cpu().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(pixels), (col * 320, 32))
            draw.text((col * 320 + 4, 17), labels[col], fill='white')
        images.append(canvas)
    return images


@torch.no_grad()
def measure(model, loss_fn, x, y):
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(x)
    value, metrics = rgb_loss(loss_fn, prediction, y, 0)
    mse = F.mse_loss(prediction.clamp(0, 1), y)
    metrics.update(loss=float(value), psnr_clipped=float(-10 * torch.log10(mse.clamp_min(1e-12))))
    return metrics, prediction


def main():
    parser = argparse.ArgumentParser()
    for name in ['data', 'features', 'weights', 'assets', 'predictor-checkpoint', 'out']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    cfg = read_json(ROOT / 'experiments/capsaicin/decoder-gate.json')
    if cfg['formal'] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This is a one-GPU diagnostic only')
    torch.set_num_threads(4)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'formal': False, 'config': cfg, 'source': read_json(ROOT / 'source.json')}
    write_json(out / 'decoder-gate.json', report)
    started = time.monotonic()
    # Load the selected predictor first, then test coexistence with JEPA-WMs imports.
    predictor = NavigationModel(args.weights).cuda().eval().requires_grad_(False)
    pred_state = torch.load(args.predictor_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if pred_state['run_id'] != 'b27dd2b1' or pred_state['step'] != cfg['predictor_selected_step']:
        raise ValueError('Unexpected selected predictor checkpoint')
    predictor.load_state_dict(pred_state['model'], strict=True)
    del pred_state
    train = RgbFeatureDataset(args.data, args.features, 'train')
    val = RgbFeatureDataset(args.data, args.features, 'val')
    indices = [0, 12, 24, 36]
    x, y = train.batch(0, indices); vx, vy = val.batch(0, indices)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        probe = predictor(x[:1], torch.zeros(1, 1, 64, device='cuda')).clone()
    model = RgbDecoder(Path(args.assets) / cfg['checkpoint']['filename'], (cfg['height'], cfg['width']), cfg['seed']).cuda()
    report['load'] = model.load_report
    report['adapter_parameters'] = sum(p.numel() for p in model.adapter.parameters())
    report['decoder_trainable_parameters'] = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    model.eval(); model.decoder.use_activation_checkpointing = False
    with torch.no_grad():
        aligned = torch.randn(1, 1, 1, 16, 16, 1408, device='cuda')
        official = unpatchify(model.decoder(aligned), 8)
        adapted = model.decode_features(aligned, (256, 256))
        torch.testing.assert_close(adapted, official, rtol=1e-4, atol=2e-5)
        report['square_upstream_max_error'] = float((adapted - official).abs().max())
        independent = model(x[:2].transpose(0, 1).float())
        changed = x[:2].transpose(0, 1).float().clone(); changed[:, 1] *= -1
        altered = model(changed)
        torch.testing.assert_close(independent[:, 0], altered[:, 0], rtol=0, atol=0)
        report['frame_independence_max_error'] = float((independent[:, 0]-altered[:, 0]).abs().max())
        assert independent.shape == (1, 2, 3, cfg['height'], cfg['width'])
    del aligned, official, adapted, independent, changed, altered
    model.decoder.use_activation_checkpointing = True
    loss_fn = make_loss(cfg).cuda()
    # Verify the actual upstream loss with corrected input range against its formula.
    with torch.no_grad():
        test_rgb = y * .9 + .03
        loss, parts = rgb_loss(loss_fn, test_rgb, y, 0)
        expected = cfg['pixel_weight'] * F.mse_loss(test_rgb, y) + cfg['lpips_weight'] * loss_fn.perceptual_loss(y.flatten(0, 1)*2-1, test_rgb.flatten(0, 1)*2-1).mean()
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
        identity, _ = rgb_loss(loss_fn, y, y, 0)
        if abs(float(identity)) > 1e-6: raise AssertionError('Identity RGB loss is not zero')
    report['loss_formula_max_error'] = float((loss-expected).abs())
    report['train_before'], _ = measure(model, loss_fn, x, y)
    report['val_before'], _ = measure(model, loss_fn, vx, vy)
    initial_weights = {n:p.detach().cpu().clone() for n,p in model.decoder.named_parameters()}
    optimizer = make_optimizer(model, cfg, joint=False)
    torch.cuda.reset_peak_memory_stats()
    history = []
    report['history'] = history
    write_json(out / 'decoder-gate.json', report)

    def update(step):
        if time.monotonic()-started > cfg['max_seconds']:
            raise RuntimeError('Decoder diagnostic time budget exceeded')
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): prediction = model(x)
        value, metrics = rgb_loss(loss_fn, prediction, y, step)
        if not torch.isfinite(value): raise FloatingPointError('Non-finite RGB loss')
        value.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        adapter_grad = model.adapter.weight.grad.abs().sum()
        if not torch.isfinite(adapter_grad) or adapter_grad <= 0:
            raise AssertionError('Adapter has no finite nonzero gradient')
        if step > cfg['adapter_warmup_steps']:
            decoder_grad = model.decoder.decoder_proj.weight.grad
            if decoder_grad is None or not torch.isfinite(decoder_grad).all() or decoder_grad.abs().sum() <= 0:
                raise AssertionError('Decoder has no finite nonzero gradient')
        optimizer.step(); torch.cuda.synchronize()
        return {'step':step, 'loss':float(value), 'grad_norm':float(grad), **metrics}

    total_steps = cfg['adapter_warmup_steps'] + cfg['joint_steps']
    for step in range(1, total_steps+1):
        if step == cfg['adapter_warmup_steps']+1:
            for name, param in model.decoder.named_parameters():
                if not torch.equal(param.detach().cpu(), initial_weights[name]):
                    raise AssertionError('Decoder changed during adapter-only warmup')
            report['warmup_decoder_unchanged'] = True
            del initial_weights
            optimizer = make_optimizer(model, cfg, joint=True)
        result = update(step)
        history.append(result)
        report['updates'] = step
        write_json(out / 'decoder-gate.json', report)
        print('DECODER_GATE_UPDATE', json.dumps(result), flush=True)
        if step == cfg['adapter_warmup_steps']+16:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'resume.pt'
                checkpoint.save(path,model,optimizer,run_id='decoder-gate',config=cfg,step=step,data_position=step)
                expected_rng=(random.random(),float(np.random.random()),float(torch.rand(1,device='cuda')))
                expected_step=update(step+1)
                expected_parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                expected_gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                restored=checkpoint.restore(path,model,optimizer,'decoder-gate',cfg)
                saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
                recovery={'restored_model':state_difference(saved['model'],model.state_dict()),
                          'restored_optimizer':state_difference(saved['optimizer'],optimizer.state_dict())}
                del saved
                actual_rng=(random.random(),float(np.random.random()),float(torch.rand(1,device='cuda')))
                actual_step=update(step+1)
                recovery.update(rng_equal=expected_rng==actual_rng, data_position=restored['data_position'],
                                expected_update=expected_step,actual_update=actual_step,
                                loss_error=abs(expected_step['loss']-actual_step['loss']),
                                parameters=state_difference(expected_parameters,dict(model.named_parameters())),
                                gradients=state_difference(expected_gradients,{n:p.grad for n,p in model.named_parameters() if p.grad is not None}))
                error=recovery['parameters']['max_error']
                report['recovery']=recovery
                write_json(out/'decoder-gate.json',report)
                print('DECODER_RECOVERY',json.dumps(recovery),flush=True)
                del expected_gradients,expected_parameters
                if (not recovery['restored_model']['exact'] or not recovery['restored_optimizer']['exact']
                    or expected_rng != actual_rng or restored['data_position'] != step or error > 2e-6 or recovery['loss_error'] > 1e-6):
                    # Diagnose a rejected replay; never convert this failure into a pass.
                    checkpoint.restore(path,model,optimizer,'decoder-gate',cfg)
                    torch.use_deterministic_algorithms(True)
                    torch.backends.cudnn.deterministic=True
                    try:
                        recovery['strict_determinism_probe']={'update':update(step+1)}
                    except RuntimeError as exc:
                        recovery['strict_determinism_probe']={'error':str(exc)}
                    write_json(out/'decoder-gate.json',report)
                    print('DECODER_DETERMINISM_PROBE',json.dumps(recovery['strict_determinism_probe']),flush=True)
                    raise AssertionError('Decoder optimizer/RNG recovery mismatch')
                report['resume_max_parameter_error']=error
                # Return to the saved state so the enclosing loop performs exactly 64 updates.
                checkpoint.restore(path,model,optimizer,'decoder-gate',cfg)
            gc.collect()
    report['train_after'], train_reconstruction = measure(model, loss_fn, x, y)
    report['val_after'], _ = measure(model, loss_fn, vx, vy)
    if report['train_after']['loss'] >= .99 * report['train_before']['loss']:
        raise AssertionError('Fixed-frame reconstruction did not improve by at least 1%')
    panels(y[:,0], train_reconstruction[:,0])[0].save(out/'train-reconstruction.png')
    # The short-fit weights are diagnostic only; export a small held-out rollout for plumbing review.
    cached, pixels = val.clip(0)
    features = cached['features'][:21][None].cuda()
    actions = cached['actions'][:20][None].cuda()
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        probe_after = predictor(x[:1], torch.zeros(1,1,64,device='cuda'))
        torch.testing.assert_close(probe_after, probe, rtol=0, atol=0)
        future = rollout(predictor, features[:,:1], actions, 20)
        predicted_features = torch.cat([features[:,:1], future],dim=1)
        real_rgb = torch.cat([model(features[:,i:i+1]) for i in range(21)],dim=1)
        pred_rgb = torch.cat([model(predicted_features[:,i:i+1]) for i in range(21)],dim=1)
    report['predictor_unchanged']=True
    frames = panels(pixels[:21].float()/255., real_rgb[0], pred_rgb[0], '64-step diagnostic only; not final visual quality')
    frames[10].save(out/'heldout-rollout.png')
    with av.open(str(train.data/val.rows[0]['video'])) as source:
        source_rate=source.streams.video[0].average_rate
    with av.open(str(out/'heldout-rollout.mp4'),'w') as container:
        stream=container.add_stream('libx264',rate=Fraction(source_rate)/4)
        stream.width=frames[0].width; stream.height=frames[0].height; stream.pix_fmt='yuv420p'
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_image(frame)): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    report.update(complete=True,updates=total_steps,history=history,gpu=torch.cuda.get_device_name(),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
                  train_clip=train.rows[0]['id'],validation_clip=val.rows[0]['id'],frame_indices=indices,
                  predictor_step=cfg['predictor_selected_step'],diagnostic_weights_retained=False,
                  interpretation='Engineering/short-fit check only; four training frames, not a trained RGB model.')
    write_json(out/'decoder-gate.json',report)
    print('DECODER_GATE_COMPLETE',flush=True)


if __name__ == '__main__': main()
