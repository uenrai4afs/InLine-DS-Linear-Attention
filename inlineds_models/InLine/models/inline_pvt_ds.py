# -----------------------------------------------------------------------
# Pyramid Vision Transformer with InLine^D-S attention
# Ported from inline_deit_ds.py and adapted to inline_pvt.py stages.
# -----------------------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'inline_pvt_ds_tiny', 'inline_pvt_ds_small', 'inline_pvt_ds_medium', 'inline_pvt_ds_large'
]


# -----------------------------------------------------------------------
# MLP and normalization
# -----------------------------------------------------------------------
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


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = (x_float / rms) * self.weight.float()
        return y.to(dtype=x.dtype)


# -----------------------------------------------------------------------
# InLine^D-S components ported from DeiT and adapted for PVT
# -----------------------------------------------------------------------
class TokenFrequencyModulator(nn.Module):
    """
    Low/high token-frequency decomposition using a cosine basis.

    PVT stages have no CLS token, so prefix_tokens defaults to 0. The basis is
    rebuilt dynamically when spatial-reduction attention changes the number of
    key/value tokens.
    """

    def __init__(self, max_tokens: int, num_heads: int, rank: int = 8, prefix_tokens: int = 0):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.prefix_tokens = prefix_tokens
        patch_tokens = max(1, max_tokens - prefix_tokens)
        self.register_buffer('basis', self._build_basis(patch_tokens, rank, device=None), persistent=False)

        self.w_mask = nn.Parameter(torch.full((num_heads, rank), 0.25))
        self.stream_gate = nn.Parameter(torch.full((num_heads,), 0.15))
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
        mask = torch.sigmoid(self.w_mask).to(dtype=x.dtype)
        coeff = torch.einsum('bhnd,rn->bhrd', x_patch, basis)
        coeff_low = coeff * mask.unsqueeze(0).unsqueeze(-1)
        patch_low = torch.einsum('bhrd,rn->bhnd', coeff_low, basis)
        patch_high = x_patch - patch_low

        e_low = patch_low.float().pow(2).mean(dim=-1, keepdim=True)
        e_high = patch_high.float().pow(2).mean(dim=-1, keepdim=True)
        frac_low = e_low / (e_low + e_high + 1e-6)
        g_scale = self.gate_scale.view(1, H, 1, 1).float()
        g_patch = torch.sigmoid(base + g_scale * (2.0 * frac_low - 1.0))

        if prefix > 0:
            g_prefix = torch.sigmoid(base).expand(B, H, prefix, 1)
            gate = torch.cat([g_prefix, g_patch], dim=2).to(dtype=x.dtype)
            low = torch.cat([x_prefix, patch_low], dim=2)
            high = torch.cat([x_prefix, patch_high], dim=2)
        else:
            gate = g_patch.to(dtype=x.dtype)
            low, high = patch_low, patch_high
        return low, high, gate


class SpectralKernel(nn.Module):
    """
    Magnitude-Modulated Spectral Kernel (MMSK): a strictly positive feature map
    for O(N) softmax-free linear attention.
    """

    _GRAD_HOOK_MAX_NORM: float = 0.5
    _A_NORM_MAX: float = math.pi
    _PHI_EPS: float = 1e-4
    _NORM_EPS: float = 1e-6

    def __init__(self, head_dim: int, num_heads: int, skip_connect: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(head_dim)  # kept for compatibility, not applied to phi input
        self.A = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim))
        self.gamma = nn.Parameter(torch.zeros(num_heads)) if skip_connect else None
        self.log_tau = nn.Parameter(torch.full((num_heads,), 0.5413))
        self.beta = nn.Parameter(torch.full((num_heads,), 0.5413))
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
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self._NORM_EPS).sqrt()
        shape = x32 / rms
        rot = torch.einsum('bhnd,hde->bhne', shape, W)

        tau = F.softplus(self.log_tau).view(1, self.num_heads, 1, 1).float()
        beta = self.beta.view(1, self.num_heads, 1, 1).float()
        feat = F.softplus(tau * rot + beta) + self._PHI_EPS

        mag_gate = F.softplus(
            self.mag_w.view(1, self.num_heads, 1, 1).float() * rms
            + self.mag_b.view(1, self.num_heads, 1, 1).float()
        )
        feat = feat * mag_gate

        if self.gamma is not None:
            residual_feat = F.softplus(shape)
            feat = feat + self.gamma.view(1, self.num_heads, 1, 1).float() * residual_feat
        return feat.to(dtype=x.dtype)


class PVTDSAttention(nn.Module):
    """
    PVT attention using the InLine^D-S mathematics from inline_deit_ds.py.

    It keeps PVT's q projection, kv projection, and spatial reduction path, but
    replaces softmax/InLine attention with MMSK dual-stream linear attention.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, freq_rank=8, skip_connect=True, lin_attn_eps=1e-4,
                 ema_momentum=0.995, use_svg=True, local_mixer_scale=0.30, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.lin_attn_eps = lin_attn_eps
        self.ema_momentum = ema_momentum
        self.use_svg = use_svg
        self.local_mixer_scale = local_mixer_scale

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = RMSNorm(dim)

        self.kernel_q = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        self.kernel_k = SpectralKernel(self.head_dim, num_heads, skip_connect=skip_connect)
        with torch.no_grad():
            self.kernel_k.A.add_(0.01 * torch.randn_like(self.kernel_k.A))
        self.kernel_q.register_orthogonal_constraint()
        self.kernel_k.register_orthogonal_constraint()

        # Dynamic basis rebuilding lets this module work at every PVT stage and
        # after spatial reduction, where Nq and Nk can differ.
        self.tfm_q = TokenFrequencyModulator(max_tokens=1, num_heads=num_heads, rank=freq_rank, prefix_tokens=0)
        self.tfm_k = TokenFrequencyModulator(max_tokens=1, num_heads=num_heads, rank=freq_rank, prefix_tokens=0)

        self.register_buffer('k_ema', torch.zeros(1, num_heads, 1, self.head_dim), persistent=True)

        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_norm = RMSNorm(dim)

        # PVT stages have no CLS/prefix tokens; queries lie on a full H x W grid.
        self.prefix_tokens = 0

        # Local attention residual (InLine local modeling capability), ported
        # exactly from inline_deit_ds.py.  We predict a per-channel 3x3 local
        # attention kernel r = MLP(mean_token) and add the local aggregation
        # sum_{j in N(i)} r_j * V_j to the attention output, where N(i) is the
        # 3x3 spatial neighbourhood of query i on the H x W grid.  This injects
        # the strong local inductive bias that linear attention lacks at O(N)
        # cost.  Because PVT reduces the key/value tokens with sr_ratio, the
        # local branch uses a value taken at *query* resolution (see forward),
        # so every query can gather its own spatial neighbourhood.
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
        self._register_scalar_hooks()

    def _register_scalar_hooks(self):
        cap = 0.5
        for p in [self.w_svg, self.b_svg]:
            if isinstance(p, nn.Parameter):
                p.register_hook(lambda g, c=cap: g.clamp(-c, c))

    @staticmethod
    def _linear_attn_single(phi_q, phi_k, v, eps: float):
        # Injective linear attention (InLine), identical mathematics to
        # inline_deit_ds.py.
        #
        # Vanilla linear attention normalises the kernel similarities by their
        # sum (out = num / denom).  That division is exactly what makes the
        # attention function non-injective: any positive rescaling of phi(q)
        # cancels in the ratio, so collinear queries of different magnitude
        # collapse to identical attention rows ("semantic confusion").
        #
        # We keep the *same* O(N) kernel compute -- the key-value summary
        # KV = sum_j phi(k_j) v_j^T and the key summary k_sum = sum_j phi(k_j)
        # are computed and reused exactly as before -- but replace the divisive
        # normalisation with a subtractive one.  The attention weights become
        #     w_ij = phi(q_i)^T phi(k_j) - (1/N) sum_s phi(q_i)^T phi(k_s) + 1/N,
        # which still sum to 1 over j while making the map injective (a positive
        # rescaling of phi(q_i) no longer cancels).  The output is then
        #     o_i = phi(q_i)^T KV - (phi(q_i)^T k_sum - 1) * (1/N) sum_j v_j,
        # i.e. the divide is swapped for a mean-subtraction.  Here the sum over j
        # runs over the key/value tokens (N = Nk), so after PVT spatial reduction
        # the normalisation is taken over the reduced key set, exactly matching
        # the kernel similarities it normalises.
        KV = torch.einsum('bhnd,bhnv->bhdv', phi_k, v)
        num = torch.einsum('bhnd,bhdv->bhnv', phi_q, KV)
        k_sum = phi_k.sum(dim=2)
        # qk_sum_i = phi(q_i)^T (sum_j phi(k_j))   -> [B, H, Nq]
        qk_sum = torch.einsum('bhnd,bhd->bhn', phi_q, k_sum)
        v_mean = v.mean(dim=2, keepdim=True)  # (1/Nk) sum_j v_j  -> [B, H, 1, Dv]
        out = num - (qk_sum.unsqueeze(-1) - 1.0) * v_mean
        # qk_sum is returned in place of the old denominator so the dual-stream
        # blend (which mixes the per-stream normalising scalars) keeps working.
        return out, qk_sum

    @staticmethod
    def _combine_stream_gates(gate_q: torch.Tensor, gate_k: torch.Tensor) -> torch.Tensor:
        """Combine query gates with the mean key gate when Nq != Nk due to sr_ratio."""
        gate_q = gate_q.clamp(1e-4, 1.0 - 1e-4)
        gate_k = gate_k.mean(dim=2, keepdim=True).clamp(1e-4, 1.0 - 1e-4)
        return torch.sigmoid(0.5 * (torch.logit(gate_q) + torch.logit(gate_k)))

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

        gate = self._combine_stream_gates(gate_q, gate_k)
        out = gate * out_low + (1.0 - gate) * out_high
        denom = gate.squeeze(-1) * denom_low + (1.0 - gate.squeeze(-1)) * denom_high
        return out, denom, q_high

    def _local_token_mix(self, out: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = out.shape
        if H * W != N:
            return self.local_norm(out)
        patches_2d = out.transpose(1, 2).reshape(B, C, H, W)
        local_feat = self.local_conv(patches_2d).reshape(B, C, N).transpose(1, 2)
        out = out + self.local_mixer_scale * local_feat
        return self.local_norm(out)

    def _local_attn_residual(self, x: torch.Tensor, v_local: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Local attention residual: sum_{j in N(i)} r_j * V_j over the 3x3
        neighbourhood of each query on the H x W grid, with r = MLP(mean token).
        Identical mathematics to inline_deit_ds.py, using the explicit PVT grid.

        Args:
            x: layer input, [B, N, C], used to predict the local kernel.
            v_local: per-head values at *query* resolution, [B, Hh, N, D]
                     (N = H*W), so each query can gather its own neighbourhood.
            H, W: spatial dimensions of the query grid.
        Returns:
            residual in [B, Hh, N, D] to be added to the attention output.
        """
        B, Hh, N, D = v_local.shape
        C = Hh * D
        # values back to [B, N, C]
        v_bnc = v_local.transpose(1, 2).reshape(B, N, C)
        res = torch.zeros_like(v_bnc)
        if H * W != N or N <= 0:
            return res.reshape(B, N, Hh, D).permute(0, 2, 1, 3)

        # predict a per-channel 3x3 kernel from the mean token: [B*C, 1, 3, 3]
        res_weight = self.local_res(x.mean(dim=1).unsqueeze(dim=-1)).reshape(B * C, 1, 3, 3)
        # values on the 2D grid, grouped per (batch, channel)
        v_grid = v_bnc.transpose(1, 2).reshape(1, B * C, H, W)
        local = F.conv2d(v_grid, res_weight, None, padding=(1, 1), groups=B * C)
        local = local.reshape(B, C, N).transpose(1, 2)  # [B, N, C]
        res = local
        return res.reshape(B, N, Hh, D).permute(0, 2, 1, 3)

    def forward(self, x, H, W):
        B, N, C = x.shape
        xf = x.float()
        q = self.q(xf).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = xf.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = xf

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        k = self._ema_centering(k)

        out32, _denom, q_high = self._dual_stream_attn(q, k, v)

        # Local attention residual (local modeling capability): add the
        # MLP-predicted 3x3 neighbourhood aggregation of the values to the
        # global linear-attention output (paper eq. 6).  Because PVT reduces the
        # key/value tokens, we take the value at *query* resolution here by
        # reusing the kv value projection on the full-resolution input, so each
        # query gathers its own H x W neighbourhood.
        v_local = self.kv(xf)[..., C:].reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out32 = out32 + self._local_attn_residual(xf, v_local, H, W)

        if self.use_svg:
            q_high_energy = q_high.pow(2).mean(dim=-1, keepdim=True)
            svg_gate = torch.sigmoid(self.w_svg.float() * q_high_energy + self.b_svg.float())
            out32 = out32 * (1.0 + 0.25 * svg_gate)

        out = out32.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self._local_token_mix(out, H, W)
        out = self.proj_drop(out)
        return out.to(dtype=x.dtype)

    def get_learned_params(self) -> dict:
        with torch.no_grad():
            result = {}
            if self.use_svg:
                result['w_svg'] = float(self.w_svg.cpu())
                result['b_svg'] = float(self.b_svg.cpu())
            return result


# -----------------------------------------------------------------------
# PVT block and patch embedding
# -----------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, sr_ratio=1, attn_type='D',
                 freq_rank=8, skip_connect=True, layer_scale_init=1e-1, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # In inline_pvt_ds.py, 'D'/'I' both use the D-S attention path. 'S' is not
        # used because this file is intended to be softmax-free.
        self.attn = PVTDSAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
            freq_rank=freq_rank, skip_connect=skip_connect)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.ls1.to(dtype=x.dtype) * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ls2.to(dtype=x.dtype) * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=RMSNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


# -----------------------------------------------------------------------
# Pyramid Vision Transformer with D-S blocks
# -----------------------------------------------------------------------
class PyramidVisionTransformerDS(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=RMSNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], la_sr_ratios='8421', num_stages=4,
                 attn_type='DDDD', freq_rank=8, skip_connect=True, layer_scale_init=1e-1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dim = embed_dims[-1]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        attn_type = 'DDDD' if attn_type is None else attn_type
        if isinstance(attn_type, str) and len(attn_type) < num_stages:
            attn_type = (attn_type + attn_type[-1] * num_stages)[:num_stages]

        for i in range(num_stages):
            stage_img_size = img_size if i == 0 else img_size // (2 ** (i - 1) * patch_size)
            patch_embed = PatchEmbed(img_size=stage_img_size,
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i], norm_layer=norm_layer)
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            # Use la_sr_ratios for D-S/InLine stages, matching inline_pvt.py's logic.
            stage_sr = sr_ratios[i] if attn_type[i] == 'S' else int(la_sr_ratios[i])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=stage_sr, attn_type=attn_type[i],
                freq_rank=freq_rank, skip_connect=skip_connect, layer_scale_init=layer_scale_init)
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == patch_embed.num_patches:
            return pos_embed
        return F.interpolate(
            pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear", align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)

    def clip_gradients(self, max_norm: float = 5.0) -> float:
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=2.0))

    def forward_features(self, x):
        B = x.shape[0]
        x = x.float()

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            x = pos_drop(x + pos_embed.float())
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# -----------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------
def _conv_filter(state_dict, patch_size=16):
    """Convert patch embedding weight from manual patchify + linear projection to conv."""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def _ds_norm_layer():
    return RMSNorm


@register_model
def inline_pvt_ds_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformerDS(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=RMSNorm,
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], attn_type='DDDD', **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def inline_pvt_ds_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformerDS(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=RMSNorm,
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], attn_type='DDDD', **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def inline_pvt_ds_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformerDS(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=RMSNorm,
        depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], attn_type='DDDD', **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def inline_pvt_ds_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformerDS(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16],
        mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=RMSNorm,
        depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], attn_type='DDDD', **kwargs)
    model.default_cfg = _cfg()
    return model
