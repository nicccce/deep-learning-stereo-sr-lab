#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_sr_lab.data.image_io import pil_to_tensor, read_rgb
from stereo_sr_lab.models import create_model
from stereo_sr_lab.training.utils import get_device, load_checkpoint, load_config


def save_map(array: np.ndarray, path: Path) -> None:
    array = array - array.min()
    array = array / max(array.max(), 1e-8)
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((512, 512), Image.Resampling.NEAREST).save(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export row-wise parallax attention map.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stereo_sr_x2.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--row", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="attention_row.png")
    args = parser.parse_args()

    config = load_config(args.config)
    config["runtime"]["device"] = args.device
    device = get_device(args.device)
    model = create_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    left = pil_to_tensor(read_rgb(args.left)).unsqueeze(0).to(device)
    right = pil_to_tensor(read_rgb(args.right)).unsqueeze(0).to(device)
    outputs = model(left, right, return_attention=True)
    maps = outputs["attention"]["right_to_left"][0].detach().float().cpu()
    row = args.row if args.row >= 0 else maps.shape[0] // 2
    save_map(maps[row].numpy(), Path(args.output))


if __name__ == "__main__":
    main()

