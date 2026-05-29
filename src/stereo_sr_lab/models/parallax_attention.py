import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallaxAttention(nn.Module):
    """Row-wise bidirectional attention for rectified stereo pairs."""

    def __init__(self, channels: int, max_disp: int = 0) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def _row_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        b, c, h, w = tensor.shape
        return tensor.permute(0, 2, 3, 1).reshape(b * h, w, c)

    def _restore(self, tokens: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        c = tokens.shape[-1]
        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _mask_by_disparity(self, scores: torch.Tensor) -> torch.Tensor:
        if self.max_disp <= 0 or self.max_disp >= scores.shape[-1]:
            return scores
        width = scores.shape[-1]
        idx = torch.arange(width, device=scores.device)
        mask = (idx[None, :] - idx[:, None]).abs() > self.max_disp
        fill = -torch.finfo(scores.dtype).max / 4
        return scores.masked_fill(mask.unsqueeze(0), fill)

    def forward(self, left: torch.Tensor, right: torch.Tensor, return_attention: bool = True) -> dict:
        b, _, h, w = left.shape
        q_left = F.normalize(self._row_tokens(self.query(left)).float(), dim=-1)
        q_right = F.normalize(self._row_tokens(self.query(right)).float(), dim=-1)
        k_left = F.normalize(self._row_tokens(self.key(left)).float(), dim=-1)
        k_right = F.normalize(self._row_tokens(self.key(right)).float(), dim=-1)
        v_left = self._row_tokens(self.value(left))
        v_right = self._row_tokens(self.value(right))

        scale = self.logit_scale.clamp(1.0, 50.0)
        scores = torch.bmm(q_left, k_right.transpose(1, 2)) * scale
        scores = self._mask_by_disparity(scores)
        right_to_left = torch.softmax(scores, dim=-1).to(v_right.dtype)

        scores_t = torch.bmm(q_right, k_left.transpose(1, 2)) * scale
        scores_t = self._mask_by_disparity(scores_t)
        left_to_right = torch.softmax(scores_t, dim=-1).to(v_left.dtype)

        left_context = self._restore(torch.bmm(right_to_left, v_right), b, h, w)
        right_context = self._restore(torch.bmm(left_to_right, v_left), b, h, w)
        output = {
            "left_context": self.proj(left_context),
            "right_context": self.proj(right_context),
            "valid_left": right_to_left.max(dim=-1).values.reshape(b, 1, h, w),
            "valid_right": left_to_right.max(dim=-1).values.reshape(b, 1, h, w),
        }
        if return_attention:
            output["right_to_left"] = right_to_left.reshape(b, h, w, w)
            output["left_to_right"] = left_to_right.reshape(b, h, w, w)
        return output

