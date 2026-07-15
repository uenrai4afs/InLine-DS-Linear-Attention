# -----------------------------------------------------------------------------
# DeiT / Vision Transformer — InLine^D-S Optimized HFALA
# -----------------------------------------------------------------------------
# Clean rewrite focused on CIFAR-100 stability and accuracy.
#
# Why accuracy used to stagnate (diagnosis)
# -----------------------------------------
# The previous feature map computed phi(x) = ELU(x / sqrt(d)) + 1.  The 1/sqrt(d)
# factor is the dot-product scaling that belongs to *softmax* attention; applied
# here to the *input of an elementwise map* it crushes every coordinate into the
# flat region of ELU, so phi(x) ~= 1 for all tokens.  The implied linear-attention
# weights phi(q).phi(k) are then dominated by a constant offset and become
# essentially uniform (measured normalized entropy = 1.000 at every layer).  The
# attention degenerates into a plain token average, the network over-smooths with
# depth, and accuracy saturates early no matter how long it trains.
#
# Fix (Magnitude-Modulated Spectral Kernel, MMSK)
# -----------------------------------------------
# The kernel now factors each token into a unit *direction* and a scalar
# *magnitude*, rotates the direction by the orthogonal W = exp(skew(A)), maps it
# through a softplus with a learnable per-head *temperature* and *floor bias*
# (which together control how peaked the kernel is), and multiplies by a monotone
# softplus gate of the token magnitude so that salient (high-norm) keys
# contribute more.  This restores content-dependent
# concentration while remaining strictly positive, softmax-free, elementwise per
# token, and therefore O(N).  The low/high streams now also blend with a
# content-adaptive gate, and prefix tokens (CLS/dist) participate in both streams.
#
# Incorporated improvements:
#   1. Magnitude-modulated, temperature-controlled positive kernel (fixes the
#      entropy collapse above) for linear attention.
#   2. Dual low/high frequency streams with content-adaptive logit-space gating.
#   3. Mean-correction branch removed; it was suppressing token contrast.
#   4. Relative positional bias is injected into patch keys before attention.
#   5. Local depthwise token mixer restores CNN-like locality for small images.
#   6. RMSNorm is used by default (LLaMA-style RMS normalization, no mean
#      subtraction / no bias) for ViT-style optimization stability and speed.
#   7. LayerScale stabilizes residual updates.
#   8. Patch size is resolved adaptively for CIFAR/TinyImageNet/ImageNet.
# -----------------------------------------------------------------------------

import math
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = {
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ),
}


# =============================================================================
# Normalization
# =============================================================================
class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        # eps=1e-6 matches the timm LayerNorm convention this norm replaces.
        # The previous 1e-8 is unnecessarily small for the residual stream and
        # can let normalized activations blow up early in training (especially
        # with LayerScale shrinking the residual), so 1e-6 is the safer default.
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = (x_float / rms) * self.weight.float()
        return y.to(dtype=x.dtype)


# =============================================================================
# Relative position bias
# =============================================================================
class RelativePositionBias(nn.Module):
    """Factorized O(N) patch bias: row bias + column bias."""

    def __init__(self, grid_size: int, num_heads: int):
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads
        num_offsets = 2 * grid_size - 1

        self.row_bias = nn.Parameter(torch.zeros(num_heads, num_offsets))
        self.col_bias = nn.Parameter(torch.zeros(num_heads, num_offsets))
        nn.init.trunc_normal_(self.row_bias, std=0.02)
        nn.init.trunc_normal_(self.col_bias, std=0.02)

        coords = torch.arange(grid_size)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('token_row', gy.flatten().long(), persistent=False)
        self.register_buffer('token_col', gx.flatten().long(), persistent=False)

    def forward(self) -> torch.Tensor:
        row = self.row_bias[:, self.token_row]
        col = self.col_bias[:, self.token_col]
        return (row + col).unsqueeze(0)  # [1, H, N_patch]


# =============================================================================
# Token frequency modulator
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
# Positive spectral kernel
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
    restored.  See the module header for the failure mode this replaces.
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
        # Learnable per-head floor bias of the positive map. The map is
        # softplus(tau * rot + beta); lowering beta shrinks the constant floor
        # that otherwise dominates the kernel dot-product and forces uniform
        # attention, so each head can dial in how peaked it is. Init beta=0.5413
        # so the map starts ~= the original ELU+1 floor, then learns to sharpen.
        self.beta = nn.Parameter(torch.full((num_heads,), 0.5413))
        # Monotone magnitude gate m = softplus(mag_w * rms + mag_b) >= 0.
        # Initialised at mag_w = 0, softplus(mag_b) == 1.0, i.e. a no-op gate that
        # the optimiser can open up per head where magnitude is informative.
        self.mag_w = nn.Parameter(torch.zeros(num_heads))
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
        tau = F.softplus(self.log_tau).view(1, self.num_heads, 1, 1).float()
        beta = self.beta.view(1, self.num_heads, 1, 1).float()
        feat = F.softplus(tau * rot + beta) + self._PHI_EPS

        # monotone magnitude gate: high-scale tokens contribute more (restores
        # content-dependent concentration). For queries this scales numerator and
        # denominator equally and cancels; for keys it re-weights contributions.
        mag_gate = F.softplus(
            self.mag_w.view(1, self.num_heads, 1, 1).float() * rms
            + self.mag_b.view(1, self.num_heads, 1, 1).float()
        )
        feat = feat * mag_gate

        if self.gamma is not None:
            residual_feat = F.softplus(shape)
            feat = feat + self.gamma.view(1, self.num_heads, 1, 1).float() * residual_feat
        return feat.to(dtype=x.dtype)


# =============================================================================
# HFALA attention
# =============================================================================
class InLineDSAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        num_tokens: int = 197,
        prefix_tokens: int = 1,
        skip_connect: bool = True,
        use_residual_gate: bool = False,
        lin_attn_eps: float = 1e-4,
        freq_rank: int = 8,
        ema_momentum: float = 0.995,
        use_svg: bool = True,
        local_mixer_scale: float = 0.30,
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.prefix_tokens = prefix_tokens
        self.lin_attn_eps = lin_attn_eps
        self.use_residual_gate = use_residual_gate
        self.use_svg = use_svg
        self.local_mixer_scale = local_mixer_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kernel_q = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        self.kernel_k = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        with torch.no_grad():
            self.kernel_k.A.add_(0.01 * torch.randn_like(self.kernel_k.A))
        self.kernel_q.register_orthogonal_constraint()
        self.kernel_k.register_orthogonal_constraint()

        self.tfm_q = TokenFrequencyModulator(num_tokens, num_heads, rank=freq_rank, prefix_tokens=prefix_tokens)
        self.tfm_k = TokenFrequencyModulator(num_tokens, num_heads, rank=freq_rank, prefix_tokens=prefix_tokens)

        self.register_buffer('k_ema', torch.zeros(1, num_heads, 1, self.head_dim), persistent=True)
        self.ema_momentum = ema_momentum

        patch_tokens = max(0, num_tokens - prefix_tokens)
        grid = int(math.isqrt(patch_tokens))
        if grid * grid == patch_tokens and grid > 1:
            self.rpb = RelativePositionBias(grid, num_heads)
            self._rpb_grid = grid
            self._rpb_gate = nn.Parameter(torch.tensor(-2.0))
        else:
            self.rpb = None
            self._rpb_grid = 0
            self._rpb_gate = None

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_norm = RMSNorm(dim)

        # Local attention residual (InLine local modeling capability).
        #
        # In addition to the global linear attention, we predict a per-channel
        # 3x3 local attention kernel r = MLP(mean_token) and add the local
        # aggregation  sum_{j in N(i)} r_j * V_j  to the attention output, where
        # N(i) is the 3x3 spatial neighbourhood of query i.  This injects the
        # strong local inductive bias that linear attention lacks, while costing
        # only O(N) extra work (a depthwise 3x3 conv over the values).  The
        # 1x1 grouped convs keep the prediction per-head.
        self.local_res = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(dim, dim * 9, kernel_size=1, groups=num_heads),
        )

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
        for p in [self._rpb_gate, self.w_svg, self.b_svg, self.residual_gate_raw]:
            if isinstance(p, nn.Parameter):
                p.register_hook(lambda g, c=cap: g.clamp(-c, c))

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
        # qk_sum is returned in place of the old denominator so the dual-stream
        # blend (which mixes the per-stream normalising scalars) keeps working.
        return out, qk_sum

    def _apply_rpb_to_keys(self, k: torch.Tensor, N: int) -> torch.Tensor:
        if self.rpb is None:
            return k
        patch_tokens = self._rpb_grid * self._rpb_grid
        prefix = N - patch_tokens
        if prefix < 0 or patch_tokens <= 0:
            return k
        rpb = self.rpb().to(device=k.device, dtype=k.dtype)
        rpb = torch.sigmoid(self._rpb_gate).to(dtype=k.dtype) * rpb.unsqueeze(-1)
        k_prefix = k[:, :, :prefix, :]
        k_patch = k[:, :, prefix:, :] + rpb
        return torch.cat([k_prefix, k_patch], dim=2)

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

    def _local_token_mix(self, out: torch.Tensor) -> torch.Tensor:
        B, N, C = out.shape
        prefix = min(self.prefix_tokens, N)
        patch_tokens = N - prefix
        grid = int(math.isqrt(patch_tokens))
        if grid * grid != patch_tokens or patch_tokens <= 0:
            return self.local_norm(out)

        out_prefix = out[:, :prefix]
        patches = out[:, prefix:]
        patches_2d = patches.transpose(1, 2).reshape(B, C, grid, grid)
        local_feat = self.local_conv(patches_2d)
        local_feat = local_feat.reshape(B, C, patch_tokens).transpose(1, 2)
        patches = patches + self.local_mixer_scale * local_feat
        return self.local_norm(torch.cat([out_prefix, patches], dim=1))

    def _local_attn_residual(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Local attention residual: sum_{j in N(i)} r_j * V_j over the 3x3
        neighbourhood of each patch query, with r = MLP(mean token).

        Args:
            x: layer input, [B, N, C], used to predict the local kernel.
            v: per-head values, [B, H, N, D] (same layout as the attention output).
        Returns:
            residual in [B, H, N, D] to be added to the attention output.
        """
        B, Hh, N, D = v.shape
        C = Hh * D
        prefix = min(self.prefix_tokens, N)
        patch_tokens = N - prefix
        grid = int(math.isqrt(patch_tokens))
        # values back to [B, N, C]
        v_bnc = v.transpose(1, 2).reshape(B, N, C)
        res = torch.zeros_like(v_bnc)
        if grid * grid != patch_tokens or patch_tokens <= 0:
            return res.reshape(B, N, Hh, D).permute(0, 2, 1, 3)

        # predict a per-channel 3x3 kernel from the mean token: [B*C, 1, 3, 3]
        res_weight = self.local_res(x.mean(dim=1).unsqueeze(dim=-1)).reshape(B * C, 1, 3, 3)
        # patch values on the 2D grid, grouped per (batch, channel)
        v_patch = v_bnc[:, prefix:, :].transpose(1, 2).reshape(1, B * C, grid, grid)
        local = F.conv2d(v_patch, res_weight, None, padding=(1, 1), groups=B * C)
        local = local.reshape(B, C, patch_tokens).transpose(1, 2)  # [B, patch, C]
        res[:, prefix:, :] = local
        return res.reshape(B, N, Hh, D).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        x32 = x.float()

        qkv = self.qkv(x32).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = self._apply_rpb_to_keys(k, N)
        k_centred = self._ema_centering(k)
        out32, _denom, q_high = self._dual_stream_attn(q, k_centred, v)

        # Local attention residual (local modeling capability): add the
        # MLP-predicted 3x3 neighbourhood aggregation of the values to the
        # global linear-attention output (paper eq. 6).
        out32 = out32 + self._local_attn_residual(x32, v)

        if self.use_svg:
            q_high_energy = q_high.pow(2).mean(dim=-1, keepdim=True)
            svg_gate = torch.sigmoid(self.w_svg.float() * q_high_energy + self.b_svg.float())
            out32 = out32 * (1.0 + 0.25 * svg_gate)

        if self.use_residual_gate and self.residual_gate_raw is not None:
            gate = torch.sigmoid(self.residual_gate_raw).float()
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(D)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            v_soft = torch.matmul(torch.softmax(logits, dim=-1), v.float())
            out32 = (1.0 - gate) * out32 + gate * v_soft

        out = out32.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self._local_token_mix(out)
        out = self.proj_drop(out)
        return out.to(dtype=x.dtype)

    def get_learned_params(self) -> dict:
        with torch.no_grad():
            result = {}
            if self.rpb is not None and self._rpb_gate is not None:
                result['rpb_gate'] = float(torch.sigmoid(self._rpb_gate).cpu())
            if self.use_svg:
                result['w_svg'] = float(self.w_svg.cpu())
                result['b_svg'] = float(self.b_svg.cpu())
            if self.use_residual_gate and self.residual_gate_raw is not None:
                g = torch.sigmoid(self.residual_gate_raw).squeeze().cpu()
                result['residual_gate'] = g.tolist() if g.dim() else float(g)
            return result


# =============================================================================
# Transformer block
# =============================================================================
class InLineDSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer=nn.GELU,
        norm_layer=RMSNorm,
        num_tokens: int = 197,
        prefix_tokens: int = 1,
        skip_connect: bool = True,
        use_residual_gate: bool = False,
        freq_rank: int = 8,
        ema_momentum: float = 0.995,
        use_svg: bool = True,
        layer_scale_init: float = 1e-1,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = InLineDSAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_tokens=num_tokens,
            prefix_tokens=prefix_tokens,
            skip_connect=skip_connect,
            use_residual_gate=use_residual_gate,
            freq_rank=freq_rank,
            ema_momentum=ema_momentum,
            use_svg=use_svg,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1.to(dtype=x.dtype) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ls2.to(dtype=x.dtype) * self.mlp(self.norm2(x)))
        return x


# =============================================================================
# Vision Transformer
# =============================================================================
class VisionTransformerDS(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        representation_size=None,
        distilled: bool = False,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init: str = '',
        skip_connect: bool = True,
        use_residual_gate: bool = False,
        freq_rank: int = 8,
        ema_momentum: float = 0.995,
        use_svg: bool = True,
        layer_scale_init: float = 1e-1,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or RMSNorm
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        num_tokens_total = num_patches + self.num_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens_total, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            InLineDSBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                num_tokens=num_tokens_total,
                prefix_tokens=self.num_tokens,
                skip_connect=skip_connect,
                use_residual_gate=use_residual_gate,
                freq_rank=freq_rank,
                ema_momentum=ema_momentum,
                use_svg=use_svg,
                layer_scale_init=layer_scale_init,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode: str = ''):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        self.apply(init_vit_weights)

    def clip_gradients(self, max_norm: float = 5.0) -> float:
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=2.0))

    def forward_features(self, x: torch.Tensor):
        x = x.float()
        x = self.patch_embed(x)
        cls = self.cls_token.float().expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls, x), dim=1)
        else:
            dist = self.dist_token.float().expand(x.shape[0], -1, -1)
            x = torch.cat((cls, dist, x), dim=1)

        x = self.pos_drop(x + self.pos_embed.float())
        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            cls_out = x[:, 0]
            gap_out = x[:, 1:].mean(dim=1)
            return self.pre_logits(0.5 * (cls_out + gap_out))
        return x[:, 0], x[:, 1]

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x_cls, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training:
                return x_cls, x_dist
            return 0.5 * (x_cls + x_dist)
        return self.head(x)


# =============================================================================
# Weight init
# =============================================================================
def init_vit_weights(module, name='', head_bias=0., jax_impl=False):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# Factory helpers
# =============================================================================
def _to_int_img_size(img_size):
    if isinstance(img_size, (tuple, list)):
        return int(img_size[-1])
    return int(img_size)


def _resolve_patch_size(kwargs, default_large: int = 16):
    """
    Keep ImageNet behavior while making CIFAR/TinyImageNet practical.
    The original uploaded file hardcoded patch_size=4, so external configs such
    as MODEL.SWIN.PATCH_SIZE=2 could not actually change DeiT patching.

    An explicit caller-supplied patch_size always wins. Otherwise the resolution
    is chosen so the token grid is fine enough for a ViT to learn structure:
        img <=  32 (CIFAR)        -> patch 4  ->  8x8  =  64 tokens
        img <=  64 (TinyImageNet) -> patch 8  ->  8x8  =  64 tokens
        img <= 224 (Food101, etc) -> patch 16 -> 14x14 = 196 tokens
    The previous img<=64 branch used patch 16, giving only a 4x4 = 16-token grid
    for 64px TinyImageNet -- far too coarse, which capped accuracy regardless of
    the normalization. Patch 8 (64 tokens) matches the CIFAR token budget while
    keeping each patch a sensible 8x8 receptive field. For more accuracy at
    higher cost, pass patch_size=4 (16x16 = 256 tokens) explicitly.
    """
    if 'patch_size' in kwargs and kwargs['patch_size'] is not None:
        return kwargs.pop('patch_size')
    img = _to_int_img_size(kwargs.get('img_size', 224))
    if img <= 32:
        return 4
    if img <= 64:
        return 8
    if img <= 224:
        return 16
    return default_large




def _optimize_small_image_kwargs(kwargs):
    """Model-side defaults for CIFAR-style training when the caller is conservative."""
    img = _to_int_img_size(kwargs.get('img_size', 224))
    if img <= 32:
        # Your config used 0.1; 0.2 is usually better for CIFAR-100 ViT training.
        kwargs['drop_path_rate'] = max(float(kwargs.get('drop_path_rate', 0.0)), 0.20)
        kwargs.setdefault('layer_scale_init', 1e-1)
    elif img <= 64:
        kwargs['drop_path_rate'] = max(float(kwargs.get('drop_path_rate', 0.0)), 0.15)
        kwargs.setdefault('layer_scale_init', 1e-1)
    return kwargs

def _create_vision_transformer_ds(variant: str, pretrained: bool = False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    return build_model_with_cfg(
        VisionTransformerDS,
        variant,
        pretrained,
        default_cfg=default_cfg,
        pretrained_custom_load=False,
        **kwargs,
    )


# =============================================================================
# Registered models
# =============================================================================
@register_model
def inline_deit_ds_tiny(pretrained: bool = False, **kwargs):
    kwargs = _optimize_small_image_kwargs(kwargs)
    patch_size = _resolve_patch_size(kwargs)
    model_kwargs = dict(patch_size=patch_size, embed_dim=192, depth=12, num_heads=6, **kwargs)
    return _create_vision_transformer_ds('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)


@register_model
def inline_deit_ds_small(pretrained: bool = False, **kwargs):
    kwargs = _optimize_small_image_kwargs(kwargs)
    patch_size = _resolve_patch_size(kwargs)
    model_kwargs = dict(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kwargs)
    return _create_vision_transformer_ds('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)


@register_model
def inline_deit_ds_base(pretrained: bool = False, **kwargs):
    kwargs = _optimize_small_image_kwargs(kwargs)
    patch_size = _resolve_patch_size(kwargs)
    model_kwargs = dict(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer_ds('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
