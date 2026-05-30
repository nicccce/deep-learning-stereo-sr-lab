"""SwinStereoSRNet — Swin Transformer based stereo super-resolution.

Implements the SwiniPASSR architecture (Jin et al., CVPRW 2022):

    conv_first → [RSTB × N/2] → conv_after_body₁  → biPAM → fusion
              → [RSTB × N/2] → conv_after_body₂ + shallow → Upsampler + bicubic

Key design points
-----------------
* **Two-stage RSTB stacks** sandwiching the Parallax Attention Module so that
  cross-view information from biPAM can be further refined by Stage 2.
* **Two conversion layers** (``conv_after_body1`` / ``conv_after_body2``): 3×3
  convolutions that align the Swin Transformer feature distribution to the CNN
  domain expected by biPAM and the upsampler.
* **Global residual** from shallow features (``conv_first`` output) to the
  final reconstruction, stabilising gradients.
* **Window-size padding**: inputs are reflect-padded to be divisible by
  ``window_size`` and cropped back before upsampling.

The forward signature is identical to :class:`StereoSRNet` so the existing
training / evaluation scripts work without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Upsampler
from .parallax_attention import ParallaxAttention
from .swin_blocks import RSTB, PatchEmbed, PatchUnEmbed, trunc_normal_


class SwinStereoSRNet(nn.Module):
    """Swin Transformer based Parallax Attention Network for Stereo Image SR.

    Args:
        scale:           Up-sampling factor (2 or 4).
        embed_dim:       Embedding / channel dimension throughout the network.
        depths:          List of ints — number of STL layers in each RSTB.
                         ``len(depths)`` must be **even**; the first half goes to
                         Stage 1 (before PAM), the second half to Stage 2 (after).
        num_heads:       Attention heads per RSTB (same length as *depths*).
        window_size:     Side length of the Swin attention window.
        mlp_ratio:       Hidden-dim expansion factor inside each STL's MLP.
        max_disp:        Maximum disparity for :class:`ParallaxAttention`
                         (0 = unlimited).
        drop_path_rate:  Peak stochastic-depth drop probability (linearly decayed
                         across all STL layers).
        resi_connection: ``"1conv"`` (single 3×3) or ``"3conv"`` (bottleneck)
                         inside each RSTB.
        use_checkpoint:  Trade compute for memory via gradient checkpointing.
        img_size:        Nominal LR spatial size for pre-computing attention masks
                         (does **not** restrict actual input size).
    """

    def __init__(
        self,
        scale: int = 2,
        embed_dim: int = 60,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        max_disp: int = 0,
        drop_path_rate: float = 0.1,
        resi_connection: str = "1conv",
        use_checkpoint: bool = False,
        img_size: int = 48,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [6, 6, 6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6, 6, 6]

        num_rstb = len(depths)
        assert num_rstb >= 2 and num_rstb % 2 == 0, (
            f"len(depths) must be ≥ 2 and even, got {num_rstb}")
        assert len(num_heads) == num_rstb
        half = num_rstb // 2

        self.scale = scale
        self.window_size = window_size
        input_resolution = (img_size, img_size)

        # Stochastic depth: linearly decaying drop rates across ALL STL layers
        total_stl = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_stl)]

        # =====================================================================
        # 1. Shallow feature extraction — single 3×3 conv, 3 → embed_dim
        # =====================================================================
        self.conv_first = nn.Conv2d(3, embed_dim, 3, 1, 1)

        # =====================================================================
        # 2. Stage 1: first half of RSTB stack
        # =====================================================================
        self.stage1_layers = nn.ModuleList()
        for i in range(half):
            dp_start = sum(depths[:i])
            dp_end = sum(depths[:i + 1])
            self.stage1_layers.append(RSTB(
                dim=embed_dim,
                input_resolution=input_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_path=dpr[dp_start:dp_end],
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
            ))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.patch_embed1 = PatchEmbed(embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.patch_unembed1 = PatchUnEmbed(embed_dim=embed_dim)

        # Conversion layer 1:  Swin → CNN domain (before PAM)
        self.conv_after_body1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # =====================================================================
        # 3. Parallax Attention + fusion
        # =====================================================================
        self.parallax = ParallaxAttention(embed_dim, max_disp=max_disp)
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # =====================================================================
        # 4. Stage 2: second half of RSTB stack
        # =====================================================================
        self.stage2_layers = nn.ModuleList()
        for i in range(half, num_rstb):
            dp_start = sum(depths[:i])
            dp_end = sum(depths[:i + 1])
            self.stage2_layers.append(RSTB(
                dim=embed_dim,
                input_resolution=input_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop_path=dpr[dp_start:dp_end],
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
            ))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.patch_embed2 = PatchEmbed(embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.patch_unembed2 = PatchUnEmbed(embed_dim=embed_dim)

        # Conversion layer 2:  Swin → CNN domain (final output)
        self.conv_after_body2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # =====================================================================
        # 5. Upsampler (PixelShuffle + final 3×3 conv → 3 channels)
        # =====================================================================
        self.upsampler = Upsampler(scale, embed_dim, out_channels=3)

        # =====================================================================
        # Weight initialisation
        # =====================================================================
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ init

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -------------------------------------------------------------- helpers

    def _check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Reflect-pad *x* so that H and W are multiples of ``window_size``."""
        _, _, h, w = x.size()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def _forward_swin_stage(
        self,
        x: torch.Tensor,
        layers: nn.ModuleList,
        norm: nn.LayerNorm,
        patch_embed: PatchEmbed,
        patch_unembed: PatchUnEmbed,
    ) -> torch.Tensor:
        """Drive one RSTB stack:  spatial → tokens → RSTBs → norm → spatial.

        Args:
            x: ``(B, C, H, W)`` spatial feature map.

        Returns:
            ``(B, C, H, W)`` after deep feature extraction.
        """
        x_size = (x.shape[2], x.shape[3])
        tokens = patch_embed(x)                     # (B, HW, C)
        for layer in layers:
            tokens = layer(tokens, x_size)
        tokens = norm(tokens)
        return patch_unembed(tokens, x_size)         # (B, C, H, W)

    # ------------------------------------------------------- per-view stages

    def _extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shallow feature + Stage 1 + conversion for **one** view.

        Returns:
            shallow:   ``(B, C, Hp, Wp)`` — stored for the global residual.
            conv_feat: ``(B, C, Hp, Wp)`` — CNN-domain features for PAM.
        """
        shallow = self.conv_first(x)
        deep1 = self._forward_swin_stage(
            shallow, self.stage1_layers, self.norm1,
            self.patch_embed1, self.patch_unembed1)
        conv_feat = self.conv_after_body1(deep1)
        return shallow, conv_feat

    def _reconstruct(self, fused: torch.Tensor,
                     shallow: torch.Tensor) -> torch.Tensor:
        """Stage 2 + conversion + global residual for **one** view.

        Returns:
            ``(B, C, Hp, Wp)`` — ready for upsampling.
        """
        deep2 = self._forward_swin_stage(
            fused, self.stage2_layers, self.norm2,
            self.patch_embed2, self.patch_unembed2)
        return self.conv_after_body2(deep2) + shallow   # global residual

    # -------------------------------------------------------------- forward

    def forward(self, lr_left: torch.Tensor, lr_right: torch.Tensor,
                return_attention: bool = True) -> dict:
        """
        Args:
            lr_left:  ``(B, 3, H, W)`` low-resolution left view.
            lr_right: ``(B, 3, H, W)`` low-resolution right view.
            return_attention: store full attention maps in the output dict.

        Returns:
            ``{"sr_left", "sr_right", "attention"}`` — same schema as
            :class:`StereoSRNet`.
        """
        _, _, H, W = lr_left.shape

        # Bicubic baseline (on original un-padded input)
        base_left = F.interpolate(
            lr_left, scale_factor=self.scale, mode="bicubic",
            align_corners=False)
        base_right = F.interpolate(
            lr_right, scale_factor=self.scale, mode="bicubic",
            align_corners=False)

        # Pad inputs to multiples of window_size
        lr_left_pad = self._check_image_size(lr_left)
        lr_right_pad = self._check_image_size(lr_right)

        # ---- Stage 1: shallow feature + Swin extraction + conversion -------
        shallow_left, conv_left = self._extract(lr_left_pad)
        shallow_right, conv_right = self._extract(lr_right_pad)

        # ---- Parallax Attention (operates in CNN domain) -------------------
        attention = self.parallax(conv_left, conv_right,
                                  return_attention=return_attention)

        # ---- Fusion --------------------------------------------------------
        fused_left = self.fusion(
            torch.cat([conv_left, attention["left_context"]], dim=1))
        fused_right = self.fusion(
            torch.cat([conv_right, attention["right_context"]], dim=1))

        # ---- Stage 2: post-fusion Swin + conversion + global residual ------
        out_left = self._reconstruct(fused_left, shallow_left)
        out_right = self._reconstruct(fused_right, shallow_right)

        # ---- Crop padding back to original LR size -------------------------
        out_left = out_left[:, :, :H, :W]
        out_right = out_right[:, :, :H, :W]

        # ---- Upsample + bicubic residual -----------------------------------
        sr_left = self.upsampler(out_left) + base_left
        sr_right = self.upsampler(out_right) + base_right

        return {
            "sr_left": sr_left,
            "sr_right": sr_right,
            "attention": attention,
        }
