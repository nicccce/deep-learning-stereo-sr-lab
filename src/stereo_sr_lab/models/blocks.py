import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ResidualStack(nn.Module):
    def __init__(self, channels: int, num_blocks: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, res_scale=res_scale) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class Upsampler(nn.Module):
    def __init__(self, scale: int, channels: int, out_channels: int = 3) -> None:
        super().__init__()
        layers = []
        if scale in {2, 4, 8}:
            steps = scale.bit_length() - 1
            for _ in range(steps):
                layers += [
                    nn.Conv2d(channels, channels * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
        elif scale == 3:
            layers += [
                nn.Conv2d(channels, channels * 9, 3, padding=1),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.1, inplace=True),
            ]
        else:
            raise ValueError("scale must be one of 2, 3, 4, or 8")
        layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class PixelShuffleDirectUpsampler(nn.Module):
    def __init__(self, scale: int, channels: int, out_channels: int = 3) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be >= 2")
        self.body = nn.Sequential(
            nn.Conv2d(channels, out_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        return self.body(x)

