import math

import torch
import torch.nn.functional as F


def _crop(tensor: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return tensor
    if tensor.shape[-1] <= border * 2 or tensor.shape[-2] <= border * 2:
        return tensor
    return tensor[..., border:-border, border:-border]


def psnr(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0) -> float:
    pred = _crop(pred.detach().float().clamp(0, 1), crop_border)
    target = _crop(target.detach().float().clamp(0, 1), crop_border)
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_window(channels: int, size: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    sigma = 1.5
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0) -> float:
    pred = _crop(pred.detach().float().clamp(0, 1), crop_border)
    target = _crop(target.detach().float().clamp(0, 1), crop_border)
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    _, channels, height, width = pred.shape
    size = min(11, height, width)
    if size % 2 == 0:
        size -= 1
    if size < 3:
        return 0.0

    window = _gaussian_window(channels, size, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, groups=channels)
    mu2 = F.conv2d(target, window, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, groups=channels) - mu12

    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return value.mean().item()


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

