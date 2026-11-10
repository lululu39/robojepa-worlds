"""Pretrained JEPA-WMs RGB decoder with a learned RoboJEPA feature adapter.

Upstream attention blocks and decoder parameters are reused unchanged. Only
spatial resizing, fixed-position interpolation and pixel assembly are extended
to rectangles. The GPU gate compares the square path with upstream forward.
"""
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import yaml

from .common import ROOT

UPSTREAM = ROOT / 'vendor/jepa_wms'
UPSTREAM_CONFIG = UPSTREAM / 'configs/vjepa_wm/vm2m/open_source_decs/step2_lpips_vm2m_vjepa2vitgopen_vgxf_vitldec_dup_256_vjtransf_norm_bs4_4n.yaml'


def load_file(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def upstream_decoder_class():
    """Stable module aliases prevent collisions with RoboJEPA's src package."""
    name = 'robojepa_worlds._jepawms_decoder'
    if name in sys.modules:
        return sys.modules[name].VisionTransformerDecoder
    dependencies = {
        'src.models.utils.modules': ('blocks', 'src/models/utils/modules.py'),
        'src.models.utils.pos_embs': ('positions', 'src/models/utils/pos_embs.py'),
        'src.utils.tensors': ('tensors', 'src/utils/tensors.py'),
    }
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'src' or k.startswith('src.')}
    try:
        for parent in ['src', 'src.models', 'src.models.utils', 'src.utils']:
            package = types.ModuleType(parent)
            package.__path__ = [str(UPSTREAM.joinpath(*parent.split('.')))]
            sys.modules[parent] = package
        for target, (suffix, path) in dependencies.items():
            sys.modules[target] = load_file('robojepa_worlds._jepawms_' + suffix, UPSTREAM / path)
        return load_file(name, UPSTREAM / 'app/plan_common/models/decoder.py').VisionTransformerDecoder
    finally:
        for key in list(sys.modules):
            if key == 'src' or key.startswith('src.'):
                del sys.modules[key]
        sys.modules.update(saved)


def unpatchify(patches, patch_size=8):
    # Exact upstream order: b t v h w (p q c) -> b t c (h p) (w q).
    b, t, v, h, w, channels = patches.shape
    if v != 1 or channels != patch_size * patch_size * 3:
        raise ValueError('Expected one RGB view with p*q*c patch ordering')
    return (patches[:, :, 0].reshape(b, t, h, w, patch_size, patch_size, 3)
            .permute(0, 1, 6, 2, 4, 3, 5).reshape(b, t, 3, h * patch_size, w * patch_size))


class RgbDecoder(nn.Module):
    def __init__(self, checkpoint_path, output_size=(176, 320), seed=42):
        super().__init__()
        torch.manual_seed(seed)
        config = yaml.safe_load(UPSTREAM_CONFIG.read_text())['model']['heads_cfg']['architectures']['image_head']['config']
        self.decoder = upstream_decoder_class()(**config)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        weights = {k.removeprefix('module.'): v for k, v in state['model'].items()}
        # Keep the original 32x32 position tensor; resize its value at forward time.
        self.decoder.load_state_dict(weights, strict=True)
        self.load_report = {'strict': True, 'tensors': len(weights), 'epoch': state.get('epoch'),
                            'upstream_input_dim': config['embed_dim'], 'hidden_dim': config['decoder_embed_dim'],
                            'depth': config['depth'], 'position_shape': list(self.decoder.decoder_pos_embed.shape)}
        del state, weights
        self.decoder.use_activation_checkpointing = True
        self.adapter = nn.Linear(1664, config['embed_dim'], bias=True)
        self.output_size = tuple(output_size)
        if any(x % self.decoder.patch_size for x in self.output_size):
            raise ValueError('Output dimensions must divide by decoder patch size')
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1), persistent=False)

    def decode_features(self, features, output_size=None):
        """Decode already aligned 1408-d features; also used by upstream parity."""
        b, t, v, h, w, c = features.shape
        if v != 1 or c != 1408:
            raise ValueError('Expected B,T,1,H,W,1408 aligned features')
        gh, gw = [x // self.decoder.patch_size for x in (output_size or self.output_size)]
        x = features.reshape(b*t*v, h, w, c).permute(0, 3, 1, 2)
        if (h, w) != (gh, gw):
            x = F.interpolate(x, size=(gh, gw), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1).reshape(b, t, v, gh, gw, c)
        x = self.decoder.decoder_embed(x)
        dim = x.shape[-1]
        side = self.decoder.grid_size
        position = self.decoder.decoder_pos_embed
        if (gh, gw) != (side, side):
            position = F.interpolate(position.reshape(1, side, side, dim).permute(0, 3, 1, 2),
                                     size=(gh, gw), mode='bicubic', align_corners=False)
            position = position.permute(0, 2, 3, 1).reshape(1, gh*gw, dim)
        x = x + position.reshape(1, 1, 1, gh, gw, dim) + self.decoder.view_tokens.reshape(1, 1, 1, 1, 1, dim)
        x = x.reshape(b*t, gh*gw, dim)
        for block in self.decoder.decoder_blocks:
            if self.decoder.use_activation_checkpointing and torch.is_grad_enabled():
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.decoder.decoder_proj(self.decoder.decoder_norm(x))
        return unpatchify(x.reshape(b, t, 1, gh, gw, -1), self.decoder.patch_size)

    def forward(self, features):
        if features.shape[-1] != 1664:
            raise ValueError('Expected 1664-d RoboJEPA features')
        normalized_rgb = self.decode_features(self.adapter(features))
        # The checkpoint predicts ImageNet-normalized pixels. Do not clamp during training.
        return normalized_rgb.float() * self.rgb_std + self.rgb_mean

    def set_phase(self, joint):
        self.adapter.requires_grad_(True)
        self.decoder.requires_grad_(joint)
        self.decoder.decoder_pos_embed.requires_grad_(False)


def make_loss(cfg):
    module = load_file('robojepa_worlds._jepawms_lpips', UPSTREAM / 'app/plan_common/models/lpips_loss.py')
    # Upstream applies inverse ImageNet normalization but passes [0,1] to LPIPS.
    # Correct LPIPS to [-1,1]; divide the pixel coefficient by four to retain
    # exactly 10*MSE([0,1]) + LPIPS([-1,1]). No discriminator or learned logvar.
    loss = module.LPIPSWithDiscriminator(
        disc_start=10**12, kl_weight=0., pixelloss_weight=cfg['pixel_weight']/4,
        perceptual_weight=cfg['lpips_weight'], disc_factor=0., add_discriminator=False)
    return loss.eval().requires_grad_(False)


def rgb_loss(loss_fn, predicted, target, step):
    predicted = predicted.flatten(0, 1).float()
    target = target.flatten(0, 1).float()
    value, logs = loss_fn(target*2-1, predicted*2-1, optimizer_idx=0, global_step=step)
    return value, {'mse': float(logs['train/rec_loss']/4), 'lpips': float(logs['train/perception_loss'])}


def make_optimizer(model, cfg, joint):
    model.set_phase(joint)
    groups = [{'params': list(model.adapter.parameters()), 'lr': cfg['adapter_lr'], 'weight_decay': 0.}]
    if joint:
        for decay in [True, False]:
            params = [p for n, p in model.decoder.named_parameters()
                      if p.requires_grad and (p.ndim > 1 and 'bias' not in n) == decay]
            groups.append({'params': params, 'lr': cfg['decoder_lr'],
                           'weight_decay': cfg['weight_decay'] if decay else 0.})
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
