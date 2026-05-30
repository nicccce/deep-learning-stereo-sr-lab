#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_sr_lab.models import create_model
from stereo_sr_lab.training.losses import StereoSRLoss
from stereo_sr_lab.training.metrics import psnr, ssim


def main() -> None:
    # ---- CNN baseline (StereoSRNet) ----
    config_cnn = {
        "data": {"scale": 2},
        "model": {"name": "stereo_sr", "scale": 2, "channels": 16, "num_feature_blocks": 2},
    }
    model = create_model(config_cnn)
    batch = {
        "lr_left": torch.rand(1, 3, 16, 24),
        "lr_right": torch.rand(1, 3, 16, 24),
        "hr_left": torch.rand(1, 3, 32, 48),
        "hr_right": torch.rand(1, 3, 32, 48),
    }
    outputs = model(batch["lr_left"], batch["lr_right"])
    assert outputs["sr_left"].shape == batch["hr_left"].shape
    loss, parts = StereoSRLoss(ffl_weight=0.01, attn_smooth_weight=0.001)(outputs, batch)
    assert torch.isfinite(loss)
    print("[CNN] OK", {"loss": float(loss), "parts": {key: float(value) for key, value in parts.items()}})
    print("[CNN]", {"psnr": psnr(outputs["sr_left"], batch["hr_left"]), "ssim": ssim(outputs["sr_left"], batch["hr_left"])})

    # ---- Swin Transformer (SwinStereoSRNet) ----
    config_swin = {
        "data": {"scale": 2},
        "model": {
            "name": "swin_stereo_sr",
            "scale": 2,
            "embed_dim": 12,
            "depths": [2, 2],
            "num_heads": [3, 3],
            "window_size": 4,
            "mlp_ratio": 2.0,
            "img_size": 16,
        },
    }
    swin_model = create_model(config_swin)
    swin_out = swin_model(batch["lr_left"], batch["lr_right"])
    assert swin_out["sr_left"].shape == batch["hr_left"].shape, (
        f"Shape mismatch: {swin_out['sr_left'].shape} vs {batch['hr_left'].shape}")
    swin_loss, swin_parts = StereoSRLoss(ffl_weight=0.01, attn_smooth_weight=0.001)(swin_out, batch)
    assert torch.isfinite(swin_loss)
    swin_loss.backward()
    print("[Swin] OK", {"loss": float(swin_loss), "parts": {key: float(value) for key, value in swin_parts.items()}})
    print("[Swin]", {"psnr": psnr(swin_out["sr_left"], batch["hr_left"]), "ssim": ssim(swin_out["sr_left"], batch["hr_left"])})

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()

