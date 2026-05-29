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
    config = {
        "data": {"scale": 2},
        "model": {"name": "stereo_sr", "scale": 2, "channels": 16, "num_feature_blocks": 2},
    }
    model = create_model(config)
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
    print({"loss": float(loss), "parts": {key: float(value) for key, value in parts.items()}})
    print({"psnr": psnr(outputs["sr_left"], batch["hr_left"]), "ssim": ssim(outputs["sr_left"], batch["hr_left"])})


if __name__ == "__main__":
    main()

