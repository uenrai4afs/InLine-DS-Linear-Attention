# =============================================================================
# Unified InLine builder for the four softmax-free InLine backbones:
#   inline_deit_*  inline_swin_*  inline_pvt_*  inline_cswin_*
# Every backbone is constructed DIRECTLY (no timm.create_model / repo dispatch)
# with attn_type forced to 'IIII' so ONLY InLine (linear) attention is used and
# no softmax attention class is ever instantiated.
# =============================================================================
import math
from functools import partial

import torch
import torch.nn as nn

# Standard size presets (faithful to the InLine paper backbones).
_SWIN_PRESETS = {
    'tiny':  dict(embed_dim=96,  depths=[2, 2, 6, 2],  num_heads=[3, 6, 12, 24]),
    'small': dict(embed_dim=96,  depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'base':  dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
}
_DEIT_PRESETS = {
    'tiny':  dict(embed_dim=192, depth=12, num_heads=6),
    'small': dict(embed_dim=320, depth=12, num_heads=10),
    'base':  dict(embed_dim=384, depth=12, num_heads=12),
}
_PVT_PRESETS = {
    'tiny':   dict(depths=[2, 2, 2, 2]),
    'small':  dict(depths=[3, 4, 6, 3]),
    'medium': dict(depths=[3, 4, 18, 3]),
    'large':  dict(depths=[3, 8, 27, 3]),
}
_CSWIN_PRESETS = {
    'tiny':  dict(embed_dim=64, depth=[2, 4, 18, 1], num_heads=[2, 4, 8, 16]),
    'small': dict(embed_dim=64, depth=[3, 6, 29, 2], num_heads=[2, 4, 8, 16]),
    'base':  dict(embed_dim=96, depth=[3, 6, 29, 2], num_heads=[4, 8, 16, 32]),
}


def family_of(model_type):
    mt = str(model_type).lower()
    if 'cswin' in mt:
        return 'cswin'
    if 'swin' in mt:
        return 'swin'
    if 'pvt' in mt:
        return 'pvt'
    return 'deit'


def size_of(model_type):
    mt = str(model_type).lower()
    for k in ('tiny', 'small', 'medium', 'large', 'base'):
        if k in mt:
            return k
    return 'tiny'


def _sanitize_attn_type(at, num_stages=4):
    """Force an InLine-only attn_type: every stage is 'I' (no 'S'/softmax)."""
    at = 'I' * num_stages if at is None else ''.join('I' for _ in str(at))
    if len(at) < num_stages:
        at = (at + 'I' * num_stages)[:num_stages]
    return at[:num_stages]


def _safe_window(grid, default=7):
    cands = [d for d in range(min(default, grid), 0, -1) if grid % d == 0]
    return cands[0] if cands else max(1, grid)


def _cfg_get(node, name, default=None):
    return getattr(node, name, default) if hasattr(node, name) else default


def _ensure_model_utils(model):
    """Attach no-op helpers expected by the training loop if a backbone lacks them."""
    if not hasattr(model, 'no_weight_decay'):
        model.no_weight_decay = lambda: set()
    if not hasattr(model, 'no_weight_decay_keywords'):
        model.no_weight_decay_keywords = lambda: set()
    if not hasattr(model, 'flops'):
        model.flops = lambda: 0
    return model


def build_model(config):
    model_type = str(config.MODEL.TYPE)
    fam = family_of(model_type)
    sz = size_of(model_type)

    num_classes = int(config.MODEL.NUM_CLASSES)
    img_size = int(config.DATA.IMG_SIZE)
    drop_path = float(_cfg_get(config.MODEL, 'DROP_PATH_RATE', 0.1))
    drop_rate = float(_cfg_get(config.MODEL, 'DROP_RATE', 0.0))
    attn_drop = float(_cfg_get(config.MODEL, 'ATTN_DROP_RATE', 0.0))

    swin_node = _cfg_get(config.MODEL, 'SWIN', None)
    patch = int(_cfg_get(swin_node, 'PATCH_SIZE', 4)) if swin_node is not None else 4

    inline_node = _cfg_get(config.MODEL, 'INLINE', None)
    attn_type = _sanitize_attn_type(str(_cfg_get(inline_node, 'ATTN_TYPE', 'IIII')))
    cswin_split = str(_cfg_get(inline_node, 'CSWIN_LA_SPLIT_SIZE', '56-28-14-7'))
    pvt_sr = str(_cfg_get(inline_node, 'PVT_LA_SR_RATIOS', '1111'))

    if fam == 'deit':
        from .inline_deit import VisionTransformer
        p = _DEIT_PRESETS[sz]
        model = VisionTransformer(
            img_size=img_size, patch_size=patch, in_chans=3, num_classes=num_classes,
            embed_dim=p['embed_dim'], depth=p['depth'], num_heads=p['num_heads'],
            mlp_ratio=4., qkv_bias=True, drop_rate=drop_rate,
            attn_drop_rate=attn_drop, drop_path_rate=drop_path)
        return _ensure_model_utils(model)

    if fam == 'swin':
        from .inline_swin import InLineSwin
        grid = max(1, img_size // patch)
        # Four stages -> three halvings; the first-stage grid must be >= 8.
        if grid < 8:
            patch = max(1, img_size // 8)
            grid = max(1, img_size // patch)
        window = int(_cfg_get(swin_node, 'WINDOW_SIZE', 7)) if swin_node is not None else 7
        window = _safe_window(grid, default=window if window > 0 else 7)
        p = _SWIN_PRESETS[sz]
        model = InLineSwin(
            img_size=img_size, patch_size=patch, in_chans=3, num_classes=num_classes,
            embed_dim=p['embed_dim'], depths=p['depths'], num_heads=p['num_heads'],
            window_size=window, mlp_ratio=4., qkv_bias=True, drop_rate=drop_rate,
            attn_drop_rate=attn_drop, drop_path_rate=drop_path, attn_type=attn_type)
        return _ensure_model_utils(model)

    if fam == 'pvt':
        from .inline_pvt import PyramidVisionTransformer
        p = _PVT_PRESETS[sz]
        model = PyramidVisionTransformer(
            img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,
            embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16],
            mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=p['depths'], sr_ratios=[8, 4, 2, 1], la_sr_ratios=pvt_sr,
            attn_type=attn_type, drop_rate=drop_rate, drop_path_rate=drop_path)
        return _ensure_model_utils(model)

    if fam == 'cswin':
        from .inline_cswin import CSWinTransformer
        p = _CSWIN_PRESETS[sz]
        model = CSWinTransformer(
            img_size=img_size, patch_size=4, in_chans=3, num_classes=num_classes,
            embed_dim=p['embed_dim'], depth=p['depth'], split_size=[1, 2, 7, 7],
            la_split_size=cswin_split, num_heads=p['num_heads'], mlp_ratio=4.,
            qkv_bias=True, drop_rate=drop_rate, attn_drop_rate=attn_drop,
            drop_path_rate=drop_path, attn_type=attn_type)
        return _ensure_model_utils(model)

    raise NotImplementedError("Unknown InLine model type: %s" % model_type)
