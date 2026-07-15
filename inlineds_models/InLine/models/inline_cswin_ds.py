# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Xiaoyi Dong
# ------------------------------------------
# Bridging the Divide: Reconsidering Softmax and Linear Attention
# Modified by Dongchen Han
# -----------------------------------------------------------------------
# Extended with the InLine^D-S attention model from inline_deit_ds.py
# -----------------------------------------------------------------------
# The attention *mathematics* are now taken verbatim from inline_deit_ds.py
# (the DeiT InLine^D-S kernel) and applied inside CSWin's cross-shaped
# windows.  Concretely, the following DeiT components replace CSWin's previous
# softmax-free quadratic mean-correction attention:
#
#   * Magnitude-Modulated Spectral Kernel (MMSK):  a strictly positive O(N)
#     feature map phi(x) = softplus(tau * rot + beta) * mag_gate, where the
#     direction is rotated by an orthogonal W = exp(skew(A)).
#   * Dual low/high frequency streams with content-adaptive logit-space gating.
#   * INJECTIVE linear attention:  the divisive normalisation is replaced by a
#     subtractive (mean) one, so a positive rescaling of phi(q) no longer
#     cancels -- this is what makes the attention injective.
#   * LOCAL attention residual:  an MLP-predicted per-channel 3x3 neighbourhood
#     aggregation of the values, which injects the local inductive bias that
#     linear attention lacks (the local-modelling property).
#
# Only CSWin's own scaffolding is preserved unchanged: the cross-shaped window
# partitioning (im2cswin / windows2img), the LePE depthwise positional
# encoding, the branch_num==2 horizontal/vertical split convention, and
# CSWinBlock.forward(x).
# -----------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(crop_pct=1.0),
}


# =============================================================================
# 1.  SpectralKernel  (Magnitude-Modulated Spectral Kernel, MMSK)
#     Ported verbatim from inline_deit_ds.py.
# =============================================================================
class SpectralKernel(nn.Module):
    """Magnitude-Modulated Spectral Kernel (MMSK): a strictly positive, O(N)
    feature map that can actually concentrate.

    Each token x is split into a unit direction d = x / rms(x) and a scalar
    magnitude rms(x).  The direction is rotated by the orthogonal matrix
    W = exp(skew(A)) (norm preserving), passed through a softplus with a learnable
    per-head temperature ``tau`` and floor ``beta`` (together they set how peaked
    the kernel is), then scaled by a monotone softplus gate of the magnitude (how
    strongly a token participates).
    Because the map is elementwise per token and non-negative, the linear
    attention sum stays O(N) and softmax-free, while content-dependent focus is
    restored.
    """

    _GRAD_HOOK_MAX_NORM: float = 0.5
    _A_NORM_MAX: float = math.pi
    _PHI_EPS: float = 1e-4
    _NORM_EPS: float = 1e-6

    def __init__(self, head_dim: int, num_heads: int, skip_connect: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        # Kept for reference/compatibility; intentionally NOT applied to the input
        # of the feature map any more (that scaling caused the entropy collapse).
        self.scale = 1.0 / math.sqrt(head_dim)
        self.A = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim))
        self.gamma = nn.Parameter(torch.zeros(num_heads)) if skip_connect else None

        # Learnable per-head sharpening temperature, tau = softplus(log_tau) > 0.
        # Initialised so that tau ~= 1.0 (softplus(0.5413) == 1.0).
        self.log_tau = nn.Parameter(torch.full((num_heads,), 0.5413))
        # Learnable per-head floor bias of the positive map. A slightly lower
        # initial beta reduces the constant positive floor that can make the
        # kernel too uniform on small images, while softplus still keeps the map
        # strictly positive.
        self.beta = nn.Parameter(torch.full((num_heads,), 0.0))
        # Monotone magnitude gate m = softplus(mag_w * rms + mag_b) >= 0.
        # A small positive initial slope lets token magnitude contribute from the
        # start without changing the positive/injective-friendly feature map.
        self.mag_w = nn.Parameter(torch.full((num_heads,), 0.05))
        self.mag_b = nn.Parameter(torch.full((num_heads,), 0.5413))

    def register_orthogonal_constraint(self):
        def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
            norms = grad.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            return grad * (self._GRAD_HOOK_MAX_NORM / norms).clamp(max=1.0)

        self.A.register_hook(_grad_hook)

        def _pre_forward_hook(module, _args):
            with torch.no_grad():
                norms = module.A.data.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
                module.A.data.mul_((module._A_NORM_MAX / norms).clamp(max=1.0))

        self.register_forward_pre_hook(_pre_forward_hook)

    @staticmethod
    def _orthogonal_from_skew(A: torch.Tensor) -> torch.Tensor:
        skew = A.float() - A.float().transpose(-2, -1)
        return torch.linalg.matrix_exp(skew)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._orthogonal_from_skew(self.A)
        x32 = x.float()

        # Scale / shape decomposition.  We divide by the RMS (not the L2 norm) so
        # that the normalised coordinates stay O(1) instead of collapsing to
        # ~1/sqrt(d); this is what lets the temperature actually sharpen the map.
        # The overall scale (rms) is kept as the magnitude signal for the gate.
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self._NORM_EPS).sqrt()  # [B,H,N,1]
        shape = x32 / rms                                                       # unit-rms

        # rotate the unit-rms features by the orthogonal W (L2/RMS preserving)
        rot = torch.einsum('bhnd,hde->bhne', shape, W)

        # learnable per-head temperature + floor bias: softplus(tau*rot + beta)
        # is a strictly positive map whose peakiness the head controls via beta.
        # Numerical guard: unconstrained tau/beta/magnitude scalars can grow under
        # a high LR and make phi(q)^T phi(k) overflow.  These clamps do not change
        # the D-S attention formulation; they only keep the same positive map in a
        # finite operating range during training from scratch.
        tau = F.softplus(self.log_tau).clamp(0.25, 4.0).view(1, self.num_heads, 1, 1).float()
        beta = self.beta.clamp(-6.0, 6.0).view(1, self.num_heads, 1, 1).float()
        feat_in = (tau * rot + beta).clamp(-12.0, 12.0)
        feat = F.softplus(feat_in) + self._PHI_EPS

        # monotone magnitude gate: high-scale tokens contribute more (restores
        # content-dependent concentration). For queries this scales numerator and
        # denominator equally and cancels; for keys it re-weights contributions.
        mag_w = self.mag_w.clamp(-2.0, 2.0).view(1, self.num_heads, 1, 1).float()
        mag_b = self.mag_b.clamp(-6.0, 6.0).view(1, self.num_heads, 1, 1).float()
        mag_arg = (mag_w * rms.clamp(max=10.0) + mag_b).clamp(-12.0, 12.0)
        mag_gate = F.softplus(mag_arg).clamp(1e-4, 8.0)
        feat = (feat * mag_gate).clamp(min=self._PHI_EPS, max=32.0)

        if self.gamma is not None:
            residual_feat = F.softplus(shape.clamp(-12.0, 12.0))
            gamma = self.gamma.clamp(-2.0, 2.0).view(1, self.num_heads, 1, 1).float()
            feat = feat + gamma * residual_feat
        feat = torch.nan_to_num(feat, nan=self._PHI_EPS, posinf=32.0, neginf=self._PHI_EPS)
        return feat.to(dtype=x.dtype)


# =============================================================================
# 1b. Token frequency modulator  (dual low/high streams)
#     Ported verbatim from inline_deit_ds.py.  In CSWin there are no CLS /
#     distillation prefix tokens, so each attention module is instantiated with
#     prefix_tokens=0 and every window token is treated as a spatial patch.
# =============================================================================
class TokenFrequencyModulator(nn.Module):
    """
    Low/high token-frequency decomposition using a cosine basis.

    Prefix tokens such as CLS/distillation tokens are not forced into the DCT
    basis; they are passed through the low stream and zeroed in the high stream.
    This avoids treating the CLS token as a spatial patch.
    """

    def __init__(self, max_tokens: int, num_heads: int, rank: int = 8, prefix_tokens: int = 1):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.prefix_tokens = prefix_tokens

        patch_tokens = max(1, max_tokens - prefix_tokens)
        basis = self._build_basis(patch_tokens, rank, device=None)
        self.register_buffer('basis', basis, persistent=False)

        self.w_mask = nn.Parameter(torch.full((num_heads, rank), 0.25))
        self.stream_gate = nn.Parameter(torch.full((num_heads,), 0.15))
        # Per-head sensitivity of the low/high blend to per-token frequency energy.
        # Initialised to 0 so the gate starts as the original per-head constant and
        # *learns* content adaptivity rather than being forced into it.
        self.gate_scale = nn.Parameter(torch.zeros(num_heads))

    @staticmethod
    def _build_basis(n_tokens: int, rank: int, device=None) -> torch.Tensor:
        rank_eff = min(rank, max(1, n_tokens))
        k = torch.arange(rank_eff, dtype=torch.float32, device=device).unsqueeze(1)
        n = torch.arange(n_tokens, dtype=torch.float32, device=device).unsqueeze(0)
        basis = torch.cos(math.pi * k * (n + 0.5) / float(n_tokens))
        basis = basis / basis.norm(dim=1, keepdim=True).clamp(min=1e-8)
        if rank_eff < rank:
            pad = torch.zeros(rank - rank_eff, n_tokens, dtype=basis.dtype, device=device)
            basis = torch.cat([basis, pad], dim=0)
        return basis

    def _basis_for(self, n_tokens: int, device, dtype) -> torch.Tensor:
        if n_tokens == self.basis.shape[1]:
            return self.basis.to(device=device, dtype=dtype)
        return self._build_basis(n_tokens, self.rank, device=device).to(dtype=dtype)

    def forward(self, x: torch.Tensor):
        B, H, N, D = x.shape
        prefix = min(self.prefix_tokens, N)
        x_prefix = x[:, :, :prefix, :]
        x_patch = x[:, :, prefix:, :]

        base = self.stream_gate.view(1, H, 1, 1).float()

        if x_patch.shape[2] == 0:
            gate = torch.sigmoid(base).expand(B, H, N, 1).to(dtype=x.dtype)
            return x, torch.zeros_like(x), gate

        basis = self._basis_for(x_patch.shape[2], x.device, x.dtype)
        mask = torch.sigmoid(self.w_mask).to(dtype=x.dtype)  # [H, r]

        coeff = torch.einsum('bhnd,rn->bhrd', x_patch, basis)
        coeff_low = coeff * mask.unsqueeze(0).unsqueeze(-1)
        patch_low = torch.einsum('bhrd,rn->bhnd', coeff_low, basis)
        patch_high = x_patch - patch_low

        # Content-adaptive low/high blend: route per token by the relative energy
        # of its low- vs high-frequency content. gate_scale=0 at init reproduces
        # the original per-head constant gate exactly.
        e_low = patch_low.float().pow(2).mean(dim=-1, keepdim=True)   # [B,H,Np,1]
        e_high = patch_high.float().pow(2).mean(dim=-1, keepdim=True)
        frac_low = e_low / (e_low + e_high + 1e-6)                    # in (0, 1)
        g_scale = self.gate_scale.view(1, H, 1, 1).float()
        g_patch = torch.sigmoid(base + g_scale * (2.0 * frac_low - 1.0))
        g_prefix = torch.sigmoid(base).expand(B, H, prefix, 1)
        gate = torch.cat([g_prefix, g_patch], dim=2).to(dtype=x.dtype)  # [B,H,N,1]

        # Prefix tokens (CLS/dist) now read content in BOTH frequency streams,
        # instead of being zeroed out of the high stream (which previously turned
        # the CLS high-stream output into a blurry average and diluted the readout).
        low = torch.cat([x_prefix, patch_low], dim=2)
        high = torch.cat([x_prefix, patch_high], dim=2)
        return low, high, gate


# =============================================================================
# Helper functions (unchanged from original CSWin)
# =============================================================================
def img2windows(img, H_sp, W_sp):
    """img: B C H W  →  B' N C"""
    B, C, H, W = img.shape
    img = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    return img.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """B' N C  →  B H W C"""
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    return img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _make_divisible_cswin_split(split_size, resolution):
    """Clamp CSWin strip size to a positive divisor of the current resolution.

    This is a geometry-only guard for CSWin window partitioning. It prevents
    invalid view() shapes on small inputs such as CIFAR/SVHN while leaving the
    InLine^D-S attention mathematics unchanged.
    """
    resolution = max(1, int(resolution))
    size = max(1, min(int(split_size), resolution))
    while size > 1 and resolution % size != 0:
        size -= 1
    return size


# =============================================================================
# 2.  Standard CSWin attention  (used when attn_type == 'S')
# =============================================================================
class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None,
                 num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim        = dim
        self.dim_out    = dim_out or dim
        self.resolution = resolution
        split_size = _make_divisible_cswin_split(split_size, resolution)
        self.split_size = split_size
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        if idx == -1:
            self.H_sp, self.W_sp = resolution, resolution
        elif idx == 0:
            self.H_sp, self.W_sp = resolution, split_size
        elif idx == 1:
            self.W_sp, self.H_sp = resolution, split_size
        else:
            raise ValueError(f"Unknown idx {idx}")

        self.get_v     = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W   = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads,
                       C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W   = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self.H_sp, self.W_sp
        x    = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x    = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)
        lepe = (lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp)
                    .permute(0, 1, 3, 2).contiguous())
        x    = (x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp)
                  .permute(0, 1, 3, 2).contiguous())
        return x, lepe

    def forward(self, qkv, x_mean):
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = W   = self.resolution
        B, L, C = q.shape
        assert L == H * W
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)
        q    = q * self.scale
        attn = F.softmax(q @ k.transpose(-2, -1), dim=-1,
                         dtype=torch.float32).to(q.dtype)
        attn = self.attn_drop(attn)
        x    = (attn @ v) + lepe
        x    = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        return windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)


# =============================================================================
# 3.  InLineDSAttention  (CSWin variant)
#
#     CSWin partitions tokens into cross-shaped horizontal/vertical strips.
#     After im2cswin partitioning, the per-window multi-head tensors have the
#     exact [b, H, n, d] layout that the DeiT InLine^D-S kernel expects, so the
#     DeiT attention mathematics are applied verbatim inside each window:
#       MMSK spectral kernel -> dual low/high streams -> INJECTIVE linear
#       attention -> LOCAL attention residual, plus the SVG gate, EMA key
#       centering and the optional softmax residual gate.
#     LePE (local positional encoding via depthwise conv) and the cross-shaped
#     window partitioning are preserved from CSWin.
# =============================================================================
class InLineDSAttention(nn.Module):
    """InLine^D-S attention for CSWin cross-shaped windows.

    The attention kernel is the DeiT InLine^D-S kernel (see inline_deit_ds.py);
    only CSWin's window partitioning and LePE positional encoding are kept.
    Maintains the two defining properties of the DeiT model:
      * Injective property      -> _linear_attn_single (subtractive, not
                                    divisive, normalisation).
      * Local-modelling property -> _local_attn_residual (MLP-predicted 3x3
                                    neighbourhood aggregation of the values).
    """

    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None,
                 num_heads=8, attn_drop=0., proj_drop=0.,
                 d_alpha: float = 0.3, skip_connect: bool = True,
                 norm_eps: float = 1e-6, use_residual_gate: bool = False,
                 freq_rank: int = 8, ema_momentum: float = 0.98,
                 use_svg: bool = True, lin_attn_eps: float = 1e-4,
                 local_ratio: int = 2, **kwargs):
        super().__init__()
        self.dim        = dim
        self.dim_out    = dim_out or dim
        self.resolution = resolution
        split_size = _make_divisible_cswin_split(split_size, resolution)
        self.split_size = split_size
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.use_residual_gate = use_residual_gate
        self.use_svg    = use_svg
        self.lin_attn_eps = lin_attn_eps
        self.ema_momentum = ema_momentum

        # CSWin spatial partitioning parameters
        if idx == -1:
            self.H_sp, self.W_sp = resolution, resolution
        elif idx == 0:
            self.H_sp, self.W_sp = resolution, split_size
        elif idx == 1:
            self.W_sp, self.H_sp = resolution, split_size
        else:
            raise ValueError(f"Unknown idx {idx}")

        # LePE: local positional encoding via depthwise conv (CSWin-native).
        # Kernel enlarged 3x3 -> 5x5 to widen the local receptive field on small
        # images (CIFAR/TinyImageNet); depthwise so the cost stays negligible.
        # Acts on V only, so injectivity of the query->key map is unaffected.
        self.get_v     = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.lepe_gamma = nn.Parameter(torch.tensor(0.10))
        self.attn_drop = nn.Dropout(attn_drop)

        # ── MMSK spectral kernels (separate q / k, exactly as DeiT) ──────────
        self.kernel_q = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        self.kernel_k = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        with torch.no_grad():
            self.kernel_k.A.add_(0.01 * torch.randn_like(self.kernel_k.A))
        self.kernel_q.register_orthogonal_constraint()
        self.kernel_k.register_orthogonal_constraint()

        # ── Dual low/high frequency streams (no CLS prefix in CSWin) ─────────
        num_tokens = self.H_sp * self.W_sp
        self.tfm_q = TokenFrequencyModulator(num_tokens, num_heads, rank=freq_rank, prefix_tokens=0)
        self.tfm_k = TokenFrequencyModulator(num_tokens, num_heads, rank=freq_rank, prefix_tokens=0)

        # ── EMA key centering ────────────────────────────────────────────────
        self.register_buffer('k_ema', torch.zeros(1, num_heads, 1, self.head_dim), persistent=True)

        # ── Local attention residual (local-modelling capability) ────────────
        # r = MLP(mean token) predicts a per-channel 3x3 kernel; the depthwise
        # 3x3 aggregation of the values over the strip neighbourhood is added to
        # the global linear-attention output.  Same construction as DeiT's
        # local_res, kept per-head via the grouped 1x1 convs.
        # local residual neighbourhood size. The kernel is kept at 5x5, but the
        # hidden generator is widened so the local branch gains capacity without
        # changing the D-S attention mathematics.
        self.local_k = 5
        self.local_ratio = int(local_ratio)
        local_hidden_dim = dim * self.local_ratio
        self.local_res = nn.Sequential(
            nn.Conv1d(dim, local_hidden_dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(local_hidden_dim, dim * self.local_k * self.local_k,
                      kernel_size=1, groups=num_heads),
        )
        # Gentle non-zero start: this lets the whole local branch receive
        # gradients immediately, while local_gamma prevents a large random
        # residual at epoch 0.
        trunc_normal_(self.local_res[-1].weight, std=1e-4)
        if self.local_res[-1].bias is not None:
            nn.init.zeros_(self.local_res[-1].bias)
        self.local_gamma = nn.Parameter(torch.tensor(0.10))

        # ── Saliency-vs-global (SVG) gate ────────────────────────────────────
        if use_svg:
            self.w_svg = nn.Parameter(torch.tensor(0.05))
            self.b_svg = nn.Parameter(torch.tensor(-1.5))
        else:
            self.w_svg = None
            self.b_svg = None

        # ── Optional per-head softmax residual gate (starts ~off, init -5.0) ──
        if use_residual_gate:
            self.residual_gate_raw = nn.Parameter(torch.full((1, num_heads, 1, 1), -5.0))
        else:
            self.residual_gate_raw = None

        self._register_scalar_hooks()

    def _register_scalar_hooks(self):
        cap = 0.5
        for p in [self.w_svg, self.b_svg, self.residual_gate_raw]:
            if isinstance(p, nn.Parameter):
                p.register_hook(lambda g, c=cap: g.clamp(-c, c))

    # ── CSWin helper methods (unchanged except lepe shape) ───────────────────
    def im2cswin(self, x):
        B, N, C = x.shape
        H = W   = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        return img2windows(x, self.H_sp, self.W_sp)   # B' N C (flat)

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W   = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        H_sp, W_sp = self.H_sp, self.W_sp
        x    = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x    = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)                              # B' C H_sp W_sp
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        x    = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    # ── DeiT InLine^D-S kernel mathematics (ported verbatim) ─────────────────
    @staticmethod
    def _stable_gate(gate_q: torch.Tensor, gate_k: torch.Tensor) -> torch.Tensor:
        gate_q = gate_q.clamp(1e-4, 1.0 - 1e-4)
        gate_k = gate_k.clamp(1e-4, 1.0 - 1e-4)
        return torch.sigmoid(0.5 * (torch.logit(gate_q) + torch.logit(gate_k)))

    @staticmethod
    def _linear_attn_single(phi_q, phi_k, v, eps: float):
        # Injective linear attention (InLine).
        #
        # Vanilla linear attention normalises the kernel similarities by their
        # sum (out = num / denom).  That division is exactly what makes the
        # attention function non-injective: any positive rescaling of phi(q)
        # cancels in the ratio, so collinear queries of different magnitude
        # collapse to identical attention rows ("semantic confusion").
        #
        # Following the InLine formulation, we keep the *same* O(N) kernel
        # compute -- the key-value summary KV = sum_j phi(k_j) v_j^T and the key
        # summary k_sum = sum_j phi(k_j) are computed and reused exactly as
        # before -- but we replace the divisive normalisation with a subtractive
        # one.  The attention weights become
        #     w_ij = phi(q_i)^T phi(k_j) - (1/N) sum_s phi(q_i)^T phi(k_s) + 1/N,
        # which still sum to 1 over j while making the map injective (a positive
        # rescaling of phi(q_i) no longer cancels).  The output is then
        #     o_i = phi(q_i)^T KV - (phi(q_i)^T k_sum - 1) * (1/N) sum_j v_j,
        # i.e. the divide is swapped for a mean-subtraction.  This changes only
        # the normalisation, not the underlying kernel mathematics.
        KV = torch.einsum('bhnd,bhnv->bhdv', phi_k, v)
        num = torch.einsum('bhnd,bhdv->bhnv', phi_q, KV)
        k_sum = phi_k.sum(dim=2)
        # qk_sum_i = phi(q_i)^T (sum_j phi(k_j))   -> [B, H, N]
        qk_sum = torch.einsum('bhnd,bhd->bhn', phi_q, k_sum)
        v_mean = v.mean(dim=2, keepdim=True)  # (1/N) sum_j v_j  -> [B, H, 1, Dv]
        out = num - (qk_sum.unsqueeze(-1) - 1.0) * v_mean
        # Constant non-zero scaling improves numerical stability for larger
        # strips without introducing softmax or query-dependent divisive
        # normalisation, so the injective D-S mechanism is preserved.
        out = out * (1.0 / math.sqrt(max(1, phi_q.shape[2])))
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
        qk_sum = torch.nan_to_num(qk_sum, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
        # qk_sum is returned in place of the old denominator so the dual-stream
        # blend (which mixes the per-stream normalising scalars) keeps working.
        return out, qk_sum

    def _dual_stream_attn(self, q, k, v):
        q_low, q_high, gate_q = self.tfm_q(q)
        k_low, k_high, gate_k = self.tfm_k(k)

        phi_q_low = self.kernel_q(q_low)
        phi_q_high = self.kernel_q(q_high)
        phi_k_low = self.kernel_k(k_low)
        phi_k_high = self.kernel_k(k_high)

        out_low, denom_low = self._linear_attn_single(phi_q_low, phi_k_low, v, self.lin_attn_eps)
        out_high, denom_high = self._linear_attn_single(phi_q_high, phi_k_high, v, self.lin_attn_eps)

        gate = self._stable_gate(gate_q, gate_k)  # [1, H, 1, 1]
        out = gate * out_low + (1.0 - gate) * out_high
        denom = gate.squeeze(-1) * denom_low + (1.0 - gate.squeeze(-1)) * denom_high
        return out, denom, q_high

    def _ema_centering(self, k: torch.Tensor) -> torch.Tensor:
        k_mean = k.mean(dim=(0, 2), keepdim=True)
        if self.training:
            with torch.no_grad():
                self.k_ema.mul_(self.ema_momentum).add_(k_mean.detach(), alpha=1.0 - self.ema_momentum)
        return k - self.k_ema.to(dtype=k.dtype)

    def _local_attn_residual(self, x_mean: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Local attention residual: sum_{j in N(i)} r_j * V_j over the 3x3
        neighbourhood of each query inside its cross-shaped strip, with
        r = MLP(mean token).  This injects the local inductive bias that linear
        attention lacks (the local-modelling property).

        Args:
            x_mean: per-(image, branch) mean token, [B, C], used to predict the
                    local kernel exactly as DeiT predicts it from x.mean(dim=1).
            v:      per-head values, [b, H, n, d] (b = number of windows).
        Returns:
            residual in [b, H, n, d] to be added to the attention output.
        """
        b, Hh, n, d = v.shape
        C = Hh * d
        h, w = self.H_sp, self.W_sp
        if h * w != n or n <= 0:
            return torch.zeros_like(v)

        # values back to [b, n, C]
        v_bnc = v.transpose(1, 2).reshape(b, n, C)

        # Predict the per-channel 3x3 kernel from the mean token, then share it
        # across the windows that belong to the same image.  img2windows lays the
        # windows of each image out contiguously, so a repeat_interleave by the
        # per-image window count reproduces that ordering.
        B = x_mean.shape[0]
        num_windows = max(1, b // B)
        K = self.local_k
        res_weight = self.local_res(x_mean.unsqueeze(-1))          # [B, C*K*K, 1]
        res_weight = res_weight.reshape(B, C, K, K)
        res_weight = res_weight.repeat_interleave(num_windows, dim=0)  # [b, C, K, K]
        res_weight = res_weight.reshape(b * C, 1, K, K)

        # depthwise KxK aggregation over the strip grid, grouped per (window, ch)
        v_patch = v_bnc.transpose(1, 2).reshape(1, b * C, h, w)
        local = F.conv2d(v_patch, res_weight, None, padding=(K // 2, K // 2), groups=b * C)
        local = local.reshape(b, C, n).transpose(1, 2)            # [b, n, C]
        local = local.reshape(b, n, Hh, d).permute(0, 2, 1, 3)    # [b, H, n, d]
        return self.local_gamma.float().clamp(0.0, 1.0).to(dtype=local.dtype) * local

    def forward(self, qkv, x_mean):
        """
        Args:
            qkv    : (3, B, L, C)   L = H*W
            x_mean : (B, C)         mean of full-resolution tokens (for res conv)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        in_dtype = q.dtype
        H = W   = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        num_heads = self.num_heads
        head_dim  = self.head_dim

        # ── CSWin partitioning ───────────────────────────────────────────────
        q_win = self.im2cswin(q)            # B' n C   n = H_sp*W_sp
        k_win = self.im2cswin(k)
        v_win, lepe = self.get_lepe(v, self.get_v)

        b, n, c = q_win.shape

        # Multi-head form: [b, H, n, d] -- exactly the DeiT kernel layout.
        q_mh = q_win.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).float()
        k_mh = k_win.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).float()
        v_mh = v_win.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).float()

        # ── EMA key centering, then injective dual-stream linear attention ───
        k_centred = self._ema_centering(k_mh)
        out, _denom, q_high = self._dual_stream_attn(q_mh, k_centred, v_mh)

        # ── Local attention residual (local-modelling capability) ────────────
        out = out + self._local_attn_residual(x_mean.float(), v_mh)

        # ── SVG gate (saliency vs. global) ───────────────────────────────────
        if self.use_svg:
            q_high_energy = q_high.pow(2).mean(dim=-1, keepdim=True).clamp(max=50.0)
            svg_arg = (self.w_svg.float().clamp(-2.0, 2.0) * q_high_energy +
                       self.b_svg.float().clamp(-8.0, 8.0)).clamp(-12.0, 12.0)
            svg_gate = torch.sigmoid(svg_arg)
            out = out * (1.0 + 0.25 * svg_gate)

        # ── Optional softmax residual gate (per head, starts ~off) ───────────
        if self.use_residual_gate and self.residual_gate_raw is not None:
            gate = torch.sigmoid(self.residual_gate_raw).float()
            logits = torch.matmul(q_mh, k_mh.transpose(-2, -1)) / math.sqrt(head_dim)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            v_soft = torch.matmul(torch.softmax(logits, dim=-1), v_mh)
            out = (1.0 - gate) * out + gate * v_soft

        # ── Window form -> [b, n, c], add LePE, then back to image ───────────
        out = out.transpose(1, 2).reshape(b, n, c)
        out = out + self.lepe_gamma.float().clamp(0.0, 1.0) * lepe.float()
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
        out = windows2img(out, self.H_sp, self.W_sp, H, W).view(B, -1, C)
        return out.to(dtype=in_dtype)

    def get_learned_params(self) -> dict:
        """Expose the learned scalar gates for inspection / visualisation."""
        with torch.no_grad():
            result = {}
            if self.use_svg and self.w_svg is not None:
                result['w_svg'] = float(self.w_svg.cpu())
                result['b_svg'] = float(self.b_svg.cpu())
            if self.use_residual_gate and self.residual_gate_raw is not None:
                g = torch.sigmoid(self.residual_gate_raw).squeeze().cpu()
                result['residual_gate'] = g.tolist() if g.dim() else float(g)
            return result


# =============================================================================
# 4.  Mlp  (unchanged)
# =============================================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# =============================================================================
# 5.  CSWinBlock  (updated to route 'D'/'I' → InLineDSAttention)
# =============================================================================
class CSWinBlock(nn.Module):
    def __init__(self, dim, reso, num_heads, split_size=7, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, attn_type='D'):
        super().__init__()
        split_size = _make_divisible_cswin_split(split_size, reso)
        self.dim               = dim
        self.num_heads         = num_heads
        self.patches_resolution = reso
        self.split_size        = split_size
        self.mlp_ratio         = mlp_ratio
        self.qkv     = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1   = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        self.branch_num = 1 if last_stage else 2
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        assert attn_type in ('D', 'I', 'S')
        Attn = InLineDSAttention if attn_type in ('D', 'I') else LePEAttention

        if last_stage:
            self.attns = nn.ModuleList([
                Attn(dim, resolution=self.patches_resolution, idx=-1,
                     split_size=split_size, num_heads=num_heads, dim_out=dim,
                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for _ in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                Attn(dim // 2, resolution=self.patches_resolution, idx=i,
                     split_size=split_size, num_heads=num_heads // 2,
                     dim_out=dim // 2,
                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2     = norm_layer(dim)

    def forward(self, x):
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W
        img  = self.norm1(x)
        qkv  = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2],
                                x.mean(dim=1)[:, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:],
                                x.mean(dim=1)[:, C // 2:])
            attended = torch.cat([x1, x2], dim=2)
        else:
            attended = self.attns[0](qkv, x.mean(dim=1))

        attended = self.proj(attended)
        x = x + self.drop_path(attended)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# 6.  Merge_Block  (unchanged)
# =============================================================================
class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        return self.norm(x)


# =============================================================================
# 7.  CSWinTransformerDS  — top-level model
# =============================================================================
class CSWinTransformerDS(nn.Module):
    """CSWin Transformer with InLine^D-S (Spectral Kernel) attention.

    All constructor arguments are identical to the original CSWinTransformer
    so build_ds.py can use the same config keys.
    attn_type: 4-char string, e.g. 'DDDD' (all D-S) or 'DDSS' (mixed).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=96, depth=(2, 2, 6, 2),
                 split_size=(1, 2, 7, 7), la_split_size='2-2-7-7',
                 num_heads=(2, 4, 8, 16), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 use_chk=False, attn_type='DDDD'):
        super().__init__()
        la_split_size = la_split_size.split('-')
        self.use_chk       = use_chk
        self.num_classes   = num_classes
        self.num_features  = self.embed_dim = embed_dim
        heads = num_heads

        # Convolutional stem: two overlapping 3x3 stride-2 convs replace the
        # single aggressive 7x7 stride-4 patch embed. Same overall /4 downsample
        # (so all downstream resolutions/split sizes are unchanged), but it
        # preserves far more spatial detail on small inputs (CIFAR/TinyImageNet).
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            Rearrange('b c h w -> b (h w) c',
                      h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim))

        curr_dim = embed_dim
        dpr      = [x.item() for x in
                    torch.linspace(0, drop_path_rate, int(np.sum(depth)))]

        attn_type = 'DDDD' if (attn_type is None) else attn_type

        def _atypes(stage_char, d):
            return [(stage_char if stage_char != 'M'
                     else ('D' if i < int(attn_type[4:]) else 'S'))
                    for i in range(d)]

        def _safe_split_size(raw_size, resolution):
            """Return a CSWin strip size that is valid for the current stage.

            CSWin partitions tensors with view(B, C, H//H_sp, H_sp, W//W_sp, W_sp),
            so the selected strip size must be positive, no larger than the stage
            resolution, and divide the stage resolution.  The ImageNet defaults
            [1, 2, 7, 7] are valid for a 224 input, but the same defaults are too
            large for CIFAR/SVHN where the stage resolutions are 8, 4, 2, 1.
            This guard only adapts the CSWin window geometry; it does not alter
            the InLine^D-S attention mathematics.
            """
            resolution = int(resolution)
            size = max(1, min(int(raw_size), resolution))
            if resolution % size == 0:
                return size
            for candidate in range(size, 0, -1):
                if resolution % candidate == 0:
                    return candidate
            return 1

        def _ssizes(atypes, la_sz, base_sz, d, resolution):
            return [
                _safe_split_size(int(la_sz) if atypes[i] == 'D' else base_sz,
                                 resolution)
                for i in range(d)
            ]

        # ── Stage 1 ──────────────────────────────────────────────────────
        at1 = _atypes(attn_type[0], depth[0])
        res1 = img_size // 4
        ss1 = _ssizes(at1, la_split_size[0], split_size[0], depth[0], res1)
        self.stage1 = nn.ModuleList([
            CSWinBlock(curr_dim, reso=res1, num_heads=heads[0],
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, split_size=ss1[i],
                       drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[i], norm_layer=norm_layer,
                       attn_type=at1[i])
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim   *= 2

        # ── Stage 2 ──────────────────────────────────────────────────────
        at2 = _atypes(attn_type[1], depth[1])
        res2 = img_size // 8
        ss2 = _ssizes(at2, la_split_size[1], split_size[1], depth[1], res2)
        self.stage2 = nn.ModuleList([
            CSWinBlock(curr_dim, reso=res2, num_heads=heads[1],
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, split_size=ss2[i],
                       drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[int(np.sum(depth[:1])) + i],
                       norm_layer=norm_layer, attn_type=at2[i])
            for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim   *= 2

        # ── Stage 3 ──────────────────────────────────────────────────────
        at3 = _atypes(attn_type[2], depth[2])
        res3 = img_size // 16
        ss3 = _ssizes(at3, la_split_size[2], split_size[2], depth[2], res3)
        self.stage3 = nn.ModuleList([
            CSWinBlock(curr_dim, reso=res3, num_heads=heads[2],
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, split_size=ss3[i],
                       drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[int(np.sum(depth[:2])) + i],
                       norm_layer=norm_layer, attn_type=at3[i])
            for i in range(depth[2])])

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim   *= 2

        # ── Stage 4 ──────────────────────────────────────────────────────
        at4 = _atypes(attn_type[3], depth[3])
        res4 = img_size // 32
        ss4 = _ssizes(at4, la_split_size[3], split_size[3], depth[3], res4)
        self.stage4 = nn.ModuleList([
            CSWinBlock(curr_dim, reso=res4, num_heads=heads[3],
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, split_size=ss4[i],
                       drop=drop_rate, attn_drop=attn_drop_rate,
                       drop_path=dpr[int(np.sum(depth[:-1])) + i],
                       norm_layer=norm_layer, last_stage=True,
                       attn_type=at4[i])
            for i in range(depth[3])])

        self.num_features = curr_dim
        self.norm = norm_layer(curr_dim)
        self.head = (nn.Linear(self.num_features, num_classes)
                     if num_classes > 0 else nn.Identity())
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.num_features, num_classes)
                     if num_classes > 0 else nn.Identity())
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            x = checkpoint.checkpoint(blk, x) if self.use_chk else blk(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blocks:
                x = checkpoint.checkpoint(blk, x) if self.use_chk else blk(x)
        return torch.mean(self.norm(x), dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


# =============================================================================
# 8.  Factory functions  (named *_ds_* for build_ds.py)
# =============================================================================
@register_model
def inline_cswin_ds_tiny(pretrained=False, **kwargs):
    model = CSWinTransformerDS(
        patch_size=4, embed_dim=64, depth=[2, 2, 9, 1],
        split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16],
        mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


@register_model
def inline_cswin_ds_small(pretrained=False, **kwargs):
    model = CSWinTransformerDS(
        patch_size=4, embed_dim=64, depth=[2, 2, 9, 1],
        split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16],
        mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


@register_model
def inline_cswin_ds_base(pretrained=False, **kwargs):
    model = CSWinTransformerDS(
        patch_size=4, embed_dim=96, depth=[2, 2, 9, 1],
        split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32],
        mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


@register_model
def inline_cswin_ds_base_384(pretrained=False, **kwargs):
    model = CSWinTransformerDS(
        patch_size=4, embed_dim=96, depth=[2, 2, 9, 1],
        split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32],
        mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model
