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


def _warp_by_attention(features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    attention = attention.to(features.dtype).contiguous().view(b * h, w, w)
    tokens = features.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
    warped = torch.bmm(attention, tokens)
    return warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)


def _crop_stereo_terms(attention: dict, lr_left: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
    m_right_to_left = attention.get("right_to_left")
    m_left_to_right = attention.get("left_to_right")
    v_left = attention.get("valid_left")
    v_right = attention.get("valid_right")
    if m_right_to_left is None or m_left_to_right is None or v_left is None or v_right is None:
        return None

    h = min(lr_left.shape[-2], m_right_to_left.shape[1], m_left_to_right.shape[1], v_left.shape[-2], v_right.shape[-2])
    w = min(
        lr_left.shape[-1],
        m_right_to_left.shape[2],
        m_right_to_left.shape[3],
        m_left_to_right.shape[2],
        m_left_to_right.shape[3],
        v_left.shape[-1],
        v_right.shape[-1],
    )
    if h <= 0 or w <= 0:
        return None

    return (
        m_right_to_left[:, :h, :w, :w],
        m_left_to_right[:, :h, :w, :w],
        v_left[:, :, :h, :w],
        v_right[:, :, :h, :w],
    )


class StereoSRLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        ffl_weight: float = 0.0,
        ffl_alpha: float = 1.0,
        attn_smooth_weight: float = 0.0,
        stereo_weight: float = 0.1,
        consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ffl_weight = ffl_weight
        self.attn_smooth_weight = attn_smooth_weight
        self.stereo_weight = stereo_weight
        self.consistency_weight = consistency_weight
        self.ffl = FocalFrequencyLoss(alpha=ffl_alpha)

    def _sr_loss(
        self,
        sr_left: torch.Tensor,
        sr_right: torch.Tensor,
        hr_left: torch.Tensor,
        hr_right: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1 = F.l1_loss(sr_left, hr_left) + F.l1_loss(sr_right, hr_right)
        loss_sr = l1 * self.l1_weight
        parts = {"sr": loss_sr.detach(), "l1": l1.detach()}

        if self.ffl_weight > 0:
            ffl = self.ffl(sr_left, hr_left) + self.ffl(sr_right, hr_right)
            loss_sr = loss_sr + ffl * self.ffl_weight
            parts["ffl"] = ffl.detach()
            parts["sr"] = loss_sr.detach()
        return loss_sr, parts

    def _stereo_losses(
        self,
        outputs: dict,
        batch: dict,
    ) -> dict[str, torch.Tensor] | None:
        attention = outputs.get("attention", {})
        cropped = _crop_stereo_terms(attention, batch["lr_left"])
        if cropped is None:
            return None
        m_right_to_left, m_left_to_right, v_left, v_right = cropped

        lr_left = batch["lr_left"][:, :, :v_left.shape[-2], :v_left.shape[-1]]
        lr_right = batch["lr_right"][:, :, :v_right.shape[-2], :v_right.shape[-1]]
        hr_left = batch["hr_left"]
        hr_right = batch["hr_right"]
        sr_left = outputs["sr_left"]
        sr_right = outputs["sr_right"]
        h, w = lr_left.shape[-2:]

        base_left = F.interpolate(lr_left, size=hr_left.shape[-2:], mode="bicubic", align_corners=False)
        base_right = F.interpolate(lr_right, size=hr_right.shape[-2:], mode="bicubic", align_corners=False)
        res_left = F.interpolate(torch.abs(hr_left - base_left), size=(h, w), mode="bicubic", align_corners=False)
        res_right = F.interpolate(torch.abs(hr_right - base_right), size=(h, w), mode="bicubic", align_corners=False)

        res_left_t = _warp_by_attention(res_right, m_right_to_left)
        res_right_t = _warp_by_attention(res_left, m_left_to_right)
        v_left_rgb = v_left.to(res_left.dtype).expand_as(res_left)
        v_right_rgb = v_right.to(res_right.dtype).expand_as(res_right)

        loss_photo = F.l1_loss(res_left * v_left_rgb, res_left_t * v_left_rgb)
        loss_photo = loss_photo + F.l1_loss(res_right * v_right_rgb, res_right_t * v_right_rgb)

        loss_h = F.l1_loss(m_right_to_left[:, :-1, :, :], m_right_to_left[:, 1:, :, :])
        loss_h = loss_h + F.l1_loss(m_left_to_right[:, :-1, :, :], m_left_to_right[:, 1:, :, :])
        loss_w = F.l1_loss(m_right_to_left[:, :, :-1, :-1], m_right_to_left[:, :, 1:, 1:])
        loss_w = loss_w + F.l1_loss(m_left_to_right[:, :, :-1, :-1], m_left_to_right[:, :, 1:, 1:])
        loss_smooth = loss_w + loss_h

        res_left_cycle = _warp_by_attention(res_right_t, m_right_to_left)
        res_right_cycle = _warp_by_attention(res_left_t, m_left_to_right)
        loss_cycle = F.l1_loss(res_left * v_left_rgb, res_left_cycle * v_left_rgb)
        loss_cycle = loss_cycle + F.l1_loss(res_right * v_right_rgb, res_right_cycle * v_right_rgb)

        sr_left_res = F.interpolate(torch.abs(hr_left - sr_left), size=(h, w), mode="bicubic", align_corners=False)
        sr_right_res = F.interpolate(torch.abs(hr_right - sr_right), size=(h, w), mode="bicubic", align_corners=False)
        sr_left_res_t = _warp_by_attention(sr_right_res, m_right_to_left.detach())
        sr_right_res_t = _warp_by_attention(sr_left_res, m_left_to_right.detach())
        loss_cons = F.l1_loss(sr_left_res * v_left_rgb, sr_left_res_t * v_left_rgb)
        loss_cons = loss_cons + F.l1_loss(sr_right_res * v_right_rgb, sr_right_res_t * v_right_rgb)

        return {
            "photo": loss_photo,
            "smooth": loss_smooth,
            "cycle": loss_cycle,
            "cons": loss_cons,
        }

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sr_left = outputs["sr_left"]
        sr_right = outputs["sr_right"]
        hr_left = batch["hr_left"]
        hr_right = batch["hr_right"]

        total, parts = self._sr_loss(sr_left, sr_right, hr_left, hr_right)

        stereo_losses = self._stereo_losses(outputs, batch)
        if stereo_losses is not None:
            total = total + self.consistency_weight * stereo_losses["cons"]
            total = total + self.stereo_weight * (
                stereo_losses["photo"] + stereo_losses["smooth"] + stereo_losses["cycle"]
            )
            for key, value in stereo_losses.items():
                parts[key] = value.detach()

        if self.attn_smooth_weight > 0:
            smooth = attention_smoothness(outputs.get("attention", {}))
            if smooth is not None:
                total = total + smooth * self.attn_smooth_weight
                parts["attn_smooth"] = smooth.detach()

        parts["total"] = total.detach()
        return total, parts
