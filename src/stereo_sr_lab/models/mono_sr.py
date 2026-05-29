import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualStack, Upsampler


class MonoSRNet(nn.Module):
    """Single-image SR baseline used for ablation against stereo fusion."""

    def __init__(
        self,
        scale: int = 2,
        channels: int = 48,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualStack(channels, num_blocks, res_scale=0.2),
            Upsampler(scale, channels, out_channels=3),
        )

    def _forward_one(self, image):
        base = F.interpolate(image, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return self.body(image) + base

    def forward(self, lr_left, lr_right, return_attention: bool = False) -> dict:
        return {
            "sr_left": self._forward_one(lr_left),
            "sr_right": self._forward_one(lr_right),
            "attention": {},
        }

