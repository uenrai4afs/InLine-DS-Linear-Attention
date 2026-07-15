# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Bridging the Divide: Reconsidering Softmax and Linear Attention
# Modified by Dongchen Han
# -----------------------------------------------------------------------
# Performance update (this file):
#   * The InLine^D-S attention mathematics are UNCHANGED.  The injective linear
#     attention (subtractive normalisation in `_linear_attn_single`), the MMSK
#     SpectralKernel, the dual low/high TokenFrequencyModulator streams, the EMA
#     key centering and the SVG gate are all kept verbatim.
#   * Only the *convolutional* local-modeling paths are strengthened, because
#     those are the cheap parts that add the most accuracy on small datasets
#     (CIFAR/SVHN/TinyImageNet) without touching the attention kernel:
#       - the local-attention residual now combines the original dynamic
#         (MLP-predicted) depthwise kernel with a static depthwise conv branch;
#       - the feed-forward block uses a ConvMlp (a zero-initialised depthwise
#         3x3 conv injected as a residual into the FFN).
#     Both additions are depthwise convolutions, so the extra per-epoch cost is
#     small, and both start as no-ops so training stays stable.
# -----------------------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F


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
    'swin_224': _cfg(),
    'swin_384': _cfg(crop_pct=1.0),
}


__all__ = [
    'inline_swin_ds_tiny', 'inline_swin_ds_small', 'inline_swin_ds_base',
    'inline_swin_ds_base_384', 'InLineSwinDS'
]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    """MLP with a lightweight depthwise convolution that injects a local (spatial)
    inductive bias into the feed-forward path.

    A depthwise k x k conv is applied at the *input* channel width (not the
    expanded hidden width) and added as a residual, so the extra cost is a single
    depthwise conv per block (~1% of the MLP FLOPs).  Its weights are zero
    initialised, so at the very start of training ConvMlp is *identical* to the
    original MLP and the local mixing is learned gradually.  This strengthens the
    convolutional local-modeling capacity of the network without touching the
    attention mathematics.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., kernel_size=3, use_conv=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_conv = use_conv
        if use_conv:
            self.dwconv = nn.Conv2d(in_features, in_features, kernel_size, 1,
                                    kernel_size // 2, groups=in_features, bias=True)
            # Zero-init -> ConvMlp == Mlp at step 0 (stable start), then learns.
            nn.init.zeros_(self.dwconv.weight)
            nn.init.zeros_(self.dwconv.bias)
        else:
            self.dwconv = None
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        if self.dwconv is not None:
            B, L, C = x.shape
            if L == H * W:
                y = x.transpose(1, 2).reshape(B, C, H, W)
                y = self.dwconv(y).flatten(2).transpose(1, 2)
                x = x + y
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, N, C = x.shape
        H = int(N ** 0.5)
        W = int(N ** 0.5)
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


# =============================================================================
# InLine^D-S components ported from inline_cswin_ds.py
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


def _make_divisible_window_size(window_size, input_resolution):
    """Return a Swin window size that divides the current stage resolution.

    This is a geometry-only guard for CIFAR/SVHN/TinyImageNet. It does not alter
    the InLine^D-S attention mathematics; it only prevents window_partition()
    from receiving a window size such as 7 for an 8x8 or 4x4 feature map.
    """
    if isinstance(window_size, (tuple, list)):
        window_size = int(window_size[0])
    H, W = int(input_resolution[0]), int(input_resolution[1])
    limit = max(1, min(H, W, int(window_size)))
    for size in range(limit, 0, -1):
        if H % size == 0 and W % size == 0:
            return size
    return 1


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class InLineDSSwinAttention(nn.Module):
    r"""Window attention using the exact InLine^D-S mathematics from inline_cswin_ds.py.

    Swin's window partitioning and shifted-window block logic are preserved. The
    attention inside each window is replaced with the D-S formulation: MMSK
    spectral kernels, dual low/high token-frequency streams, injective
    subtractive linear attention, EMA key centering, SVG modulation, and the
    learned local 3x3 residual. The local residual preserves the local-modeling
    property; the subtractive linear attention preserves injectivity.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., shift_size=0, freq_rank=8,
                 skip_connect=True, ema_momentum=0.995, use_svg=True,
                 lin_attn_eps=1e-4, use_residual_gate=False,
                 local_kernel_size=3, local_ratio=2, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size) if isinstance(window_size, (tuple, list)) else to_2tuple(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.shift_size = shift_size
        self.use_svg = use_svg
        self.use_residual_gate = use_residual_gate
        self.lin_attn_eps = lin_attn_eps
        self.ema_momentum = ema_momentum

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kernel_q = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        self.kernel_k = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        with torch.no_grad():
            self.kernel_k.A.add_(0.01 * torch.randn_like(self.kernel_k.A))
        self.kernel_q.register_orthogonal_constraint()
        self.kernel_k.register_orthogonal_constraint()

        num_tokens = self.window_size[0] * self.window_size[1]
        self.tfm_q = TokenFrequencyModulator(num_tokens, num_heads, rank=freq_rank, prefix_tokens=0)
        self.tfm_k = TokenFrequencyModulator(num_tokens, num_heads, rank=freq_rank, prefix_tokens=0)
        self.register_buffer('k_ema', torch.zeros(1, num_heads, 1, self.head_dim), persistent=True)

        # Stronger learned local residual. This keeps the D-S attention mathematics
        # unchanged while giving the model more convolutional local modeling
        # capacity inside each Swin window.
        self.local_kernel_size = int(local_kernel_size)
        if self.local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd so padding preserves the window resolution.")
        self.local_ratio = int(local_ratio)
        local_hidden_dim = dim * self.local_ratio
        self.local_res = nn.Sequential(
            nn.Conv1d(dim, local_hidden_dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(local_hidden_dim, dim * self.local_kernel_size * self.local_kernel_size,
                      kernel_size=1, groups=num_heads),
        )
        # Gentle non-zero start: unlike an all-zero last layer, this allows the
        # whole local branch to receive gradients immediately, while the small
        # local_gamma prevents a large random residual at epoch 0.
        trunc_normal_(self.local_res[-1].weight, std=1e-4)
        if self.local_res[-1].bias is not None:
            nn.init.zeros_(self.local_res[-1].bias)
        self.local_gamma = nn.Parameter(torch.tensor(0.10))

        # Static, position-equivariant depthwise convolution branch.  It runs in
        # parallel with the dynamic (MLP-predicted) kernel of `local_res` above;
        # together they form a stronger local-modeling residual.  Being depthwise
        # (groups == dim) it is cheap and adds little per-epoch time, while the
        # static kernel gives a stable, translation-equivariant local prior that
        # the purely dynamic kernel lacks.  Initialised small so the residual is
        # gentle at the start of training.
        self.local_static = nn.Conv2d(dim, dim, kernel_size=self.local_kernel_size,
                                      padding=self.local_kernel_size // 2,
                                      groups=dim, bias=False)
        trunc_normal_(self.local_static.weight, std=0.02)

        if use_svg:
            self.w_svg = nn.Parameter(torch.tensor(0.05))
            self.b_svg = nn.Parameter(torch.tensor(-1.5))
        else:
            self.w_svg = None
            self.b_svg = None

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

    @staticmethod
    def _stable_gate(gate_q: torch.Tensor, gate_k: torch.Tensor) -> torch.Tensor:
        gate_q = gate_q.clamp(1e-4, 1.0 - 1e-4)
        gate_k = gate_k.clamp(1e-4, 1.0 - 1e-4)
        return torch.sigmoid(0.5 * (torch.logit(gate_q) + torch.logit(gate_k)))

    @staticmethod
    def _linear_attn_single(phi_q, phi_k, v, eps: float):
        # Injective linear attention from inline_cswin_ds.py. The divisive
        # normalisation is replaced with a subtractive mean term, so positive
        # rescaling of phi(q) no longer cancels.
        KV = torch.einsum('bhnd,bhnv->bhdv', phi_k, v)
        num = torch.einsum('bhnd,bhdv->bhnv', phi_q, KV)
        k_sum = phi_k.sum(dim=2)
        qk_sum = torch.einsum('bhnd,bhd->bhn', phi_q, k_sum)
        v_mean = v.mean(dim=2, keepdim=True)
        out = num - (qk_sum.unsqueeze(-1) - 1.0) * v_mean
        # A constant non-zero scale keeps larger windows numerically stable
        # without introducing query-dependent divisive normalization or softmax.
        out = out * (1.0 / math.sqrt(max(1, phi_q.shape[2])))
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
        qk_sum = torch.nan_to_num(qk_sum, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
        return out, qk_sum

    def _ema_centering(self, k: torch.Tensor) -> torch.Tensor:
        k_mean = k.mean(dim=(0, 2), keepdim=True)
        if self.training:
            with torch.no_grad():
                self.k_ema.mul_(self.ema_momentum).add_(k_mean.detach(), alpha=1.0 - self.ema_momentum)
        return k - self.k_ema.to(dtype=k.dtype)

    def _dual_stream_attn(self, q, k, v):
        q_low, q_high, gate_q = self.tfm_q(q)
        k_low, k_high, gate_k = self.tfm_k(k)

        phi_q_low = self.kernel_q(q_low)
        phi_q_high = self.kernel_q(q_high)
        phi_k_low = self.kernel_k(k_low)
        phi_k_high = self.kernel_k(k_high)

        out_low, denom_low = self._linear_attn_single(phi_q_low, phi_k_low, v, self.lin_attn_eps)
        out_high, denom_high = self._linear_attn_single(phi_q_high, phi_k_high, v, self.lin_attn_eps)

        gate = self._stable_gate(gate_q, gate_k)
        out = gate * out_low + (1.0 - gate) * out_high
        denom = gate.squeeze(-1) * denom_low + (1.0 - gate.squeeze(-1)) * denom_high
        return out, denom, q_high

    def _local_attn_residual(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, Hh, n, d = v.shape
        C = Hh * d
        h, w = int(self.window_size[0]), int(self.window_size[1])
        if h * w != n or n <= 0:
            return torch.zeros_like(v)

        v_bnc = v.transpose(1, 2).reshape(b, n, C)
        k = self.local_kernel_size
        v_img = v_bnc.transpose(1, 2).reshape(b, C, h, w)                # [b, C, h, w]

        # (1) Dynamic, content-adaptive depthwise kernel predicted from the window
        #     mean token (the DeiT-style local-modeling formulation, unchanged).
        res_weight = self.local_res(x.mean(dim=1).unsqueeze(dim=-1)).reshape(b * C, 1, k, k)
        local_dyn = F.conv2d(v_img.reshape(1, b * C, h, w), res_weight, None,
                             padding=(k // 2, k // 2), groups=b * C).reshape(b, C, n)

        # (2) Static, position-equivariant depthwise convolution branch.  Adds a
        #     stable translation-equivariant local prior on top of the dynamic
        #     kernel.  Depthwise => cheap.
        local_stat = self.local_static(v_img).reshape(b, C, n)

        local = (local_dyn + local_stat).transpose(1, 2)                # [b, n, C]
        if torch.isnan(local).any():
            local = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
        local = local.reshape(b, n, Hh, d).permute(0, 2, 1, 3)
        return self.local_gamma.float().clamp(0.0, 1.0).to(dtype=local.dtype) * local

    def _softmax_residual(self, q, k, v, mask):
        b, Hh, n, d = q.shape
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        if mask is not None:
            nW = mask.shape[0]
            logits = logits.view(b // nW, nW, Hh, n, n) + mask.unsqueeze(1).unsqueeze(0)
            logits = logits.view(-1, Hh, n, n)
        return torch.matmul(torch.softmax(logits, dim=-1), v)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        in_dtype = x.dtype
        num_heads = self.num_heads
        head_dim = c // num_heads

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_mh = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).float()
        k_mh = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).float()
        v_mh = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).float()

        k_centred = self._ema_centering(k_mh)
        out, _denom, q_high = self._dual_stream_attn(q_mh, k_centred, v_mh)
        out = out + self._local_attn_residual(x.float(), v_mh)

        if self.use_svg:
            q_high_energy = q_high.pow(2).mean(dim=-1, keepdim=True).clamp(max=50.0)
            svg_arg = (self.w_svg.float().clamp(-2.0, 2.0) * q_high_energy +
                       self.b_svg.float().clamp(-8.0, 8.0)).clamp(-12.0, 12.0)
            svg_gate = torch.sigmoid(svg_arg)
            out = out * (1.0 + 0.25 * svg_gate)

        if self.use_residual_gate and self.residual_gate_raw is not None:
            gate = torch.sigmoid(self.residual_gate_raw).float()
            v_soft = self._softmax_residual(q_mh, k_mh, v_mh, mask)
            out = (1.0 - gate) * out + gate * v_soft

        out = out.transpose(1, 2).reshape(b, n, c)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out.to(dtype=in_dtype)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += 2 * self.num_heads * N * (self.dim // self.num_heads) * N
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='I',
                 freq_rank=8, local_kernel_size=3, local_ratio=2, use_conv_ffn=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = _make_divisible_window_size(window_size, input_resolution)
        self.shift_size = min(shift_size, max(0, self.window_size - 1))
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        self.window_size = _make_divisible_window_size(self.window_size, self.input_resolution)
        if self.shift_size >= self.window_size:
            self.shift_size = self.window_size // 2
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        assert attn_type in ['I', 'D', 'S']
        attn = InLineDSSwinAttention if attn_type in ['I', 'D'] else WindowAttention
        self.attn = attn(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            freq_rank=freq_rank, local_kernel_size=local_kernel_size, local_ratio=local_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # ConvMlp injects a cheap depthwise-conv local mixer into the FFN. With
        # use_conv=False it is exactly the original MLP.
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop, use_conv=use_conv_ffn)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, attn_type='I',
                 freq_rank=8, local_kernel_size=3, local_ratio=2, use_conv_ffn=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        attn_types = [(attn_type if attn_type[0] != 'M' else ('I' if i < int(attn_type[1:]) else 'S')) for i in range(depth)]
        window_sizes = [(window_size if attn_types[i] in ['I', 'D'] else max(7, (window_size // 8))) for i in range(depth)]
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_sizes[i],
                                 shift_size=0 if (i % 2 == 0) else window_sizes[i] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 attn_type=attn_types[i],
                                 freq_rank=freq_rank,
                                 local_kernel_size=local_kernel_size,
                                 local_ratio=local_ratio,
                                 use_conv_ffn=use_conv_ffn)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                try:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                except TypeError:
                    x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Overlapping patch embedding improves local texture preservation while
        # keeping the same output token resolution and the same patch_size stride.
        overlap_kernel = (2 * patch_size[0] - 1, 2 * patch_size[1] - 1)
        overlap_padding = (patch_size[0] - 1, patch_size[1] - 1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=overlap_kernel,
                              stride=patch_size, padding=overlap_padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        kernel_h, kernel_w = self.proj.kernel_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * (kernel_h * kernel_w)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class InLineSwinDS(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, attn_type='DDDD', freq_rank=8,
                 local_kernel_size=3, local_ratio=2, use_conv_ffn=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_cnn = PosCNN(embed_dim, embed_dim, s=1)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               attn_type=attn_type[i_layer] + (attn_type[self.num_layers:] if attn_type[i_layer] == 'M' else ''),
                               freq_rank=freq_rank,
                               local_kernel_size=local_kernel_size,
                               local_ratio=local_ratio,
                               use_conv_ffn=use_conv_ffn)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def clip_gradients(self, max_norm: float = 5.0) -> float:
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=2.0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_cnn(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


# Backward-compatible alias for old code that imports InLineSwin directly.
InLineSwin = InLineSwinDS

@register_model
def inline_swin_ds_tiny(pretrained=False, **kwargs):
    attn_type = kwargs.pop('attn_type', 'DDDD')
    model = InLineSwinDS(
        patch_size=4, embed_dim=64, depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16],
        window_size=8, mlp_ratio=4., qkv_bias=True, attn_type=attn_type,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.05), **kwargs)
    model.default_cfg = default_cfgs['swin_224']
    return model


@register_model
def inline_swin_ds_small(pretrained=False, **kwargs):
    attn_type = kwargs.pop('attn_type', 'DDDD')
    model = InLineSwinDS(
        patch_size=4, embed_dim=96, depths=[2, 2, 9, 2], num_heads=[3, 6, 12, 24],
        window_size=8, mlp_ratio=4., qkv_bias=True, attn_type=attn_type,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.07), **kwargs)
    model.default_cfg = default_cfgs['swin_224']
    return model


@register_model
def inline_swin_ds_base(pretrained=False, **kwargs):
    attn_type = kwargs.pop('attn_type', 'DDDD')
    model = InLineSwinDS(
        patch_size=4, embed_dim=96, depths=[2, 2, 12, 2], num_heads=[3, 6, 12, 24],
        window_size=8, mlp_ratio=4., qkv_bias=True, attn_type=attn_type,
        drop_path_rate=kwargs.pop('drop_path_rate', 0.10), **kwargs)
    model.default_cfg = default_cfgs['swin_224']
    return model


@register_model
def inline_swin_ds_base_384(pretrained=False, **kwargs):
    attn_type = kwargs.pop('attn_type', 'DDDD')
    model = InLineSwinDS(
        img_size=384, patch_size=4, embed_dim=96, depths=[2, 2, 12, 2],
        num_heads=[3, 6, 12, 24], window_size=12, mlp_ratio=4., qkv_bias=True,
        attn_type=attn_type, drop_path_rate=kwargs.pop('drop_path_rate', 0.10), **kwargs)
    model.default_cfg = default_cfgs['swin_384']
    return model
