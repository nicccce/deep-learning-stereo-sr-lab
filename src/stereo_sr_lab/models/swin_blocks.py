"""Swin Transformer building blocks for image restoration.

Hand-written based on the SwinIR architecture (Liang et al., ICCVW 2021),
cross-referenced with the NTIRE22 SwiniPASSR implementation (Subury).

Self-contained: no dependency on ``timm`` or other external libraries beyond
PyTorch.  ``DropPath``, ``trunc_normal_`` and ``to_2tuple`` are implemented
inline so the module works on any standard PyTorch >= 1.10 environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# ---------------------------------------------------------------------------
# Utility helpers (replacing timm imports)
# ---------------------------------------------------------------------------

def to_2tuple(x):
    """Convert scalar to 2-tuple; pass-through if already a sequence."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Fill *tensor* in-place with values drawn from a truncated normal."""
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """Stochastic Depth – drops an entire residual branch during training.

    Reference: "Deep Networks with Stochastic Depth" (Huang et al., ECCV 2016).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: (B, 1, 1, ...) – broadcast over all spatial / token dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Two-layer MLP with GELU, used inside every Swin Transformer layer."""

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer=nn.GELU,
                 drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Window partition / reverse
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition a feature map into non-overlapping square windows.

    Args:
        x: ``(B, H, W, C)``
        window_size: side length of the square window.

    Returns:
        ``(num_windows * B, window_size, window_size, C)``
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = (x.permute(0, 1, 3, 2, 4, 5)
               .contiguous()
               .view(-1, window_size, window_size, C))
    return windows


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """Reverse of :func:`window_partition`.

    Args:
        windows: ``(num_windows * B, window_size, window_size, C)``
        window_size: side length used when partitioning.
        H, W: original spatial dimensions.

    Returns:
        ``(B, H, W, C)``
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Window Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    r"""Window-based Multi-Head Self-Attention (W-MSA) with learned relative
    position bias.

    Supports both regular (W-MSA) and shifted (SW-MSA) configurations through
    the ``mask`` argument in :meth:`forward`.

    Args:
        dim:       Number of input channels.
        window_size: ``(Wh, Ww)`` – height and width of the attention window.
        num_heads: Number of attention heads.
        qkv_bias:  Add learnable bias to Q, K, V projections.
        qk_scale:  Override default scale ``head_dim ** -0.5``.
        attn_drop: Dropout on attention weights.
        proj_drop: Dropout on output projection.
    """

    def __init__(self, dim: int, window_size: tuple[int, int],
                 num_heads: int, qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size          # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Learnable relative position bias table ---------------------------
        # Table size: (2*Wh - 1) * (2*Ww - 1)  entries,  nH  heads.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))

        # Pre-compute pair-wise relative position index --------------------
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"))     # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                  # 2, N
        relative_coords = (coords_flatten[:, :, None]
                           - coords_flatten[:, None, :])           # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1       # shift → ≥ 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1   # row stride
        relative_position_index = relative_coords.sum(-1)          # N, N
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    ``(nW*B, N, C)`` where ``N = Wh * Ww``.
            mask: ``(nW, N, N)`` or *None*.  Non-zero entries are filled with
                  ``-100`` so that softmax suppresses them.
        """
        B_, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)                        # each (B_, nH, N, Cd)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                 # (B_, nH, N, N)

        # ---- relative position bias ------------------------------------
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous() # nH, N, N
        attn = attn + rpb.unsqueeze(0)

        # ---- shifted-window mask ----------------------------------------
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, window_size={self.window_size}, "
                f"num_heads={self.num_heads}")


# ---------------------------------------------------------------------------
# Swin Transformer Layer  (one W-MSA or SW-MSA + FFN)
# ---------------------------------------------------------------------------

class SwinTransformerLayer(nn.Module):
    r"""A single Swin Transformer layer.

    Consists of a (shifted) window multi-head self-attention block followed by
    a two-layer MLP, both wrapped with LayerNorm and residual connections.

    When ``shift_size > 0`` the layer uses **SW-MSA** (shifted windows);
    otherwise it uses plain **W-MSA**.

    Args:
        dim:              Channel count.
        input_resolution: Nominal ``(H, W)`` for pre-computing the SW-MSA mask.
        num_heads:        Attention heads.
        window_size:      Side length of the (square) attention window.
        shift_size:       Cyclic-shift offset for SW-MSA (0 → W-MSA).
        mlp_ratio:        Hidden-dim expansion factor in the MLP.
        drop_path:        Stochastic depth rate.
    """

    def __init__(self, dim: int, input_resolution: tuple[int, int],
                 num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # If the feature map is smaller than the window, skip shifting.
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # Pre-compute the attention mask for the nominal resolution.
        if self.shift_size > 0:
            self.register_buffer("attn_mask",
                                 self._calculate_mask(self.input_resolution))
        else:
            self.register_buffer("attn_mask", None)

    # --------------------------------------------------------------------- #

    def _calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        """Build the ``-100 / 0`` additive mask for SW-MSA."""
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
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
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor,
                x_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x:      ``(B, H*W, C)`` – flattened token sequence.
            x_size: ``(H, W)`` – current spatial dimensions (may differ from
                    ``input_resolution`` at eval time).
        """
        H, W = x_size
        B, _L, C = x.shape

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # ---- cyclic shift -----------------------------------------------
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # ---- partition windows ------------------------------------------
        x_windows = window_partition(shifted_x, self.window_size)      # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # ---- W-MSA / SW-MSA --------------------------------------------
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(
                x_windows,
                mask=self._calculate_mask(x_size).to(x.device))

        # ---- merge windows ----------------------------------------------
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # ---- reverse cyclic shift ---------------------------------------
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # ---- FFN --------------------------------------------------------
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")


# ---------------------------------------------------------------------------
# BasicLayer — a stack of alternating W-MSA / SW-MSA layers
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    """Stack of :class:`SwinTransformerLayer` blocks for one RSTB.

    Layers alternate between ``shift_size=0`` (W-MSA) and
    ``shift_size=window_size//2`` (SW-MSA).
    """

    def __init__(self, dim: int, input_resolution: tuple[int, int],
                 depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float | list[float] = 0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerLayer(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list)
                           else drop_path),
                norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor,
                x_size: tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        return x


# ---------------------------------------------------------------------------
# Patch Embed / Un-embed  (spatial ↔ token reshape, no real "patching")
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Reshape ``(B, C, H, W) → (B, H*W, C)`` with optional LayerNorm.

    In SwinIR for SR tasks ``patch_size = 1``, so this is a pure reshape
    (no actual patch convolution).
    """

    def __init__(self, embed_dim: int = 96,
                 norm_layer: type | None = None) -> None:
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)           # B, C, H, W → B, HW, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Reshape ``(B, H*W, C) → (B, C, H, W)``."""

    def __init__(self, embed_dim: int = 96) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor,
                x_size: tuple[int, int]) -> torch.Tensor:
        return x.transpose(1, 2).view(-1, self.embed_dim,
                                      x_size[0], x_size[1])


# ---------------------------------------------------------------------------
# RSTB — Residual Swin Transformer Block
# ---------------------------------------------------------------------------

class RSTB(nn.Module):
    """Residual Swin Transformer Block.

    Structure::

        input (tokens)
          │
          ├──► BasicLayer (N × STL) ──► PatchUnEmbed ──► Conv ──► PatchEmbed ──┐
          │                                                                     │
          └─────────────────────────── (+) ◄────────────────────────────────────┘
          │
        output (tokens)

    The Conv layer bridges the gap between the self-attention token space and
    the spatial CNN domain, enabling cross-window information flow at each
    RSTB boundary.

    Args:
        resi_connection: ``"1conv"`` (single 3×3 conv) or ``"3conv"``
            (bottleneck: 3×3 → 1×1 → 3×3 with LeakyReLU).
    """

    def __init__(self, dim: int, input_resolution: tuple[int, int],
                 depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float | list[float] = 0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint: bool = False,
                 resi_connection: str = "1conv") -> None:
        super().__init__()

        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection!r}")

        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x: torch.Tensor,
                x_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x:      ``(B, H*W, C)`` token sequence.
            x_size: ``(H, W)`` spatial dims.

        Returns:
            ``(B, H*W, C)`` with residual connection.
        """
        out = self.residual_group(x, x_size)        # tokens → tokens
        out = self.patch_unembed(out, x_size)       # → (B, C, H, W)
        out = self.conv(out)                        # spatial conv
        out = self.patch_embed(out)                 # → (B, HW, C)
        return out + x                              # residual
