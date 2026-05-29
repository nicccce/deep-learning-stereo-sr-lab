import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualStack, Upsampler
from .parallax_attention import ParallaxAttention


class StereoSRNet(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        channels: int = 48,
        num_feature_blocks: int = 6,
        num_reconstruct_blocks: int = 4,
        max_disp: int = 0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.feature = ResidualStack(channels, num_feature_blocks, res_scale=0.2)
        self.parallax = ParallaxAttention(channels, max_disp=max_disp)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(channels, num_reconstruct_blocks, res_scale=0.2),
        )
        self.upsampler = Upsampler(scale, channels, out_channels=3)

    def _extract(self, x):
        return self.feature(self.head(x))

    def forward(self, lr_left, lr_right, return_attention: bool = True) -> dict:
        base_left = F.interpolate(lr_left, scale_factor=self.scale, mode="bicubic", align_corners=False)
        base_right = F.interpolate(lr_right, scale_factor=self.scale, mode="bicubic", align_corners=False)

        feat_left = self._extract(lr_left)
        feat_right = self._extract(lr_right)
        attention = self.parallax(feat_left, feat_right, return_attention=return_attention)

        fused_left = self.fusion(torch.cat([feat_left, attention["left_context"]], dim=1))
        fused_right = self.fusion(torch.cat([feat_right, attention["right_context"]], dim=1))

        return {
            "sr_left": self.upsampler(fused_left) + base_left,
            "sr_right": self.upsampler(fused_right) + base_right,
            "attention": attention,
        }

