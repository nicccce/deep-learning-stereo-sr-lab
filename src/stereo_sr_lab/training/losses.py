import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        diff = torch.view_as_real(pred_fft - target_fft)
        distance = diff.pow(2).sum(dim=-1).add(1e-12).sqrt()
        with torch.no_grad():
            weight = distance.pow(self.alpha)
            dims = tuple(range(1, weight.ndim))
            denom = weight.amax(dim=dims, keepdim=True).clamp_min(1e-8)
            weight = weight / denom
        return (weight * distance).mean()


def attention_smoothness(attention: dict) -> torch.Tensor | None:
    maps = [attention.get("right_to_left"), attention.get("left_to_right")]
    valid_maps = [item for item in maps if item is not None]
    if not valid_maps:
        return None
    losses = []
    for item in valid_maps:
        if item.shape[1] > 1:
            losses.append((item[:, 1:] - item[:, :-1]).abs().mean())
        if item.shape[2] > 1:
            losses.append((item[:, :, 1:] - item[:, :, :-1]).abs().mean())
    return sum(losses) / max(len(losses), 1)


class StereoSRLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        ffl_weight: float = 0.0,
        ffl_alpha: float = 1.0,
        attn_smooth_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ffl_weight = ffl_weight
        self.attn_smooth_weight = attn_smooth_weight
        self.ffl = FocalFrequencyLoss(alpha=ffl_alpha)

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sr_left = outputs["sr_left"]
        sr_right = outputs["sr_right"]
        hr_left = batch["hr_left"]
        hr_right = batch["hr_right"]

        l1 = F.l1_loss(sr_left, hr_left) + F.l1_loss(sr_right, hr_right)
        total = l1 * self.l1_weight
        parts = {"l1": l1.detach()}

        if self.ffl_weight > 0:
            ffl = self.ffl(sr_left, hr_left) + self.ffl(sr_right, hr_right)
            total = total + ffl * self.ffl_weight
            parts["ffl"] = ffl.detach()

        if self.attn_smooth_weight > 0:
            smooth = attention_smoothness(outputs.get("attention", {}))
            if smooth is not None:
                total = total + smooth * self.attn_smooth_weight
                parts["attn_smooth"] = smooth.detach()

        parts["total"] = total.detach()
        return total, parts

