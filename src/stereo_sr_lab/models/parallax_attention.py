import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMatchingBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 4 if channels % 4 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


def M_Relax(attention: torch.Tensor, num_pixels: int = 2) -> torch.Tensor:
    """Relax a row-wise parallax map along the query-pixel dimension."""
    if num_pixels <= 0:
        return attention
    relaxed = [attention]
    for offset in range(1, num_pixels + 1):
        relaxed.append(F.pad(attention[:, :-offset, :], (0, 0, offset, 0)))
        relaxed.append(F.pad(attention[:, offset:, :], (0, 0, 0, offset)))
    return torch.stack(relaxed, dim=0).sum(dim=0)


class ParallaxAttention(nn.Module):
    """iPASSR-style bidirectional Parallax Attention Module.

    The module builds both right-to-left and left-to-right row-wise parallax
    maps from a shared stereo matching score, estimates mutual visibility with
    the M_Relax occlusion handling rule, and returns mask-gated transported
    features for stereo fusion.
    """

    def __init__(self, channels: int, max_disp: int = 0, relax_pixels: int = 2) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.relax_pixels = relax_pixels
        self.match_left = nn.Sequential(
            ResidualMatchingBlock(channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.match_right = nn.Sequential(
            ResidualMatchingBlock(channels),
            nn.Conv2d(channels, channels, 1),
        )

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

    def set_max_disp(self, max_disp: int) -> None:
        self.max_disp = int(max_disp)

    def _visibility(
        self,
        right_to_left: torch.Tensor,
        left_to_right: torch.Tensor,
        b: int,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        right_to_left_relaxed = M_Relax(right_to_left, self.relax_pixels)
        left_to_right_relaxed = M_Relax(left_to_right, self.relax_pixels)

        valid_left = torch.bmm(
            right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
            left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2),
        ).detach().contiguous().view(b, 1, h, w)
        valid_right = torch.bmm(
            left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),
            right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2),
        ).detach().contiguous().view(b, 1, h, w)

        return torch.tanh(5 * valid_left), torch.tanh(5 * valid_right)

    def forward(self, left: torch.Tensor, right: torch.Tensor, return_attention: bool = True) -> dict:
        b, _, h, w = left.shape
        query = self.match_left(left)
        support = self.match_right(right)
        query = query - query.mean(dim=3, keepdim=True)
        support = support - support.mean(dim=3, keepdim=True)

        q_left = self._row_tokens(query).float()
        k_right = support.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).float()
        scores = torch.bmm(q_left, k_right)
        scores = self._mask_by_disparity(scores)
        right_to_left = torch.softmax(scores, dim=-1).to(left.dtype)
        left_to_right = torch.softmax(scores.transpose(1, 2), dim=-1).to(right.dtype)
        valid_left, valid_right = self._visibility(
            right_to_left.float(), left_to_right.float(), b, h, w)
        valid_left = valid_left.to(left.dtype)
        valid_right = valid_right.to(right.dtype)

        v_left = self._row_tokens(left)
        v_right = self._row_tokens(right)
        left_transferred = self._restore(torch.bmm(right_to_left, v_right), b, h, w)
        right_transferred = self._restore(torch.bmm(left_to_right, v_left), b, h, w)
        left_context = left * (1 - valid_left) + left_transferred * valid_left
        right_context = right * (1 - valid_right) + right_transferred * valid_right
        output = {
            "left_context": left_context,
            "right_context": right_context,
            "left_transferred": left_transferred,
            "right_transferred": right_transferred,
            "valid_left": valid_left,
            "valid_right": valid_right,
            "occlusion_left": 1 - valid_left,
            "occlusion_right": 1 - valid_right,
        }
        if return_attention:
            output["right_to_left"] = right_to_left.reshape(b, h, w, w)
            output["left_to_right"] = left_to_right.reshape(b, h, w, w)
        return output
