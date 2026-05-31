from __future__ import annotations

import torch
import torch.nn.functional as F

from .swin_stereo_sr import SwinStereoSRNet


class SwinMonoSRNet(SwinStereoSRNet):
    """Single-image ablation aligned with SwinStereoSRNet.

    The network keeps the same shallow conv, two-stage RSTB backbone, fusion
    layer, upsampler, padding, and bicubic residual path as SwinStereoSRNet.
    The only structural difference is that stereo parallax attention is removed
    and the fusion context is synthesized from the current view only.
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
        upsampler: str = "pixelshuffle",
        pam_downsample: int = 1,
        use_checkpoint: bool = False,
        img_size: int = 48,
        fusion_context: str = "zero",
    ) -> None:
        super().__init__(
            scale=scale,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            max_disp=max_disp,
            drop_path_rate=drop_path_rate,
            resi_connection=resi_connection,
            upsampler=upsampler,
            pam_downsample=pam_downsample,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
        )
        if fusion_context not in {"zero", "self"}:
            raise ValueError("fusion_context must be 'zero' or 'self'")
        self.parallax = None
        self.fusion_context = fusion_context

    def _mono_context(self, conv_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion_context == "self":
            return conv_feat
        return torch.zeros_like(conv_feat)

    def _forward_one(self, lr_image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = lr_image.shape
        base = F.interpolate(
            lr_image,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        lr_pad = self._check_image_size(lr_image)
        shallow, conv_feat = self._extract(lr_pad)
        context = self._mono_context(conv_feat)
        fused = self.fusion(torch.cat([conv_feat, context], dim=1))
        out = self._reconstruct(fused, shallow)
        out = out[:, :, :height, :width]
        return self.upsampler(out) + base

    def forward(
        self,
        lr_left: torch.Tensor,
        lr_right: torch.Tensor,
        return_attention: bool = False,
    ) -> dict:
        return {
            "sr_left": self._forward_one(lr_left),
            "sr_right": self._forward_one(lr_right),
            "attention": {},
        }
