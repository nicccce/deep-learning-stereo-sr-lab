#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_sr_lab.data import StereoSRDataset, build_pairs
from stereo_sr_lab.data.image_io import pil_to_tensor, read_rgb, save_tensor_image
from stereo_sr_lab.models import create_model
from stereo_sr_lab.training.utils import get_device, load_checkpoint, load_config


@torch.no_grad()
def run_pair(model, left_tensor, right_tensor, device):
    outputs = model(
        left_tensor.unsqueeze(0).to(device),
        right_tensor.unsqueeze(0).to(device),
        return_attention=False,
    )
    return outputs["sr_left"][0], outputs["sr_right"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stereo SR inference.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stereo_sr_x2.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--data-root")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
    device = get_device(config["runtime"].get("device", "cuda"))
    model = create_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    out_dir = Path(args.out_dir)

    if args.left:
        right_path = args.right or args.left
        left = pil_to_tensor(read_rgb(args.left))
        right = pil_to_tensor(read_rgb(right_path))
        sr_left, sr_right = run_pair(model, left, right, device)
        save_tensor_image(sr_left, out_dir / "sr_left.png")
        save_tensor_image(sr_right, out_dir / "sr_right.png")
        return

    data_root = args.data_root or config["data"]["root"]
    dataset_name = args.dataset or config["data"]["dataset"]
    split = args.split or config["data"].get("val_split", "Validation")
    pairs = build_pairs(data_root, dataset_name, split)
    if args.limit:
        pairs = pairs[: args.limit]
    dataset = StereoSRDataset(pairs, scale=config["data"]["scale"], split="eval")

    for item in dataset:
        sr_left, sr_right = run_pair(model, item["lr_left"], item["lr_right"], device)
        name = item["name"]
        save_tensor_image(sr_left, out_dir / f"{name}_L_sr.png")
        save_tensor_image(sr_right, out_dir / f"{name}_R_sr.png")


if __name__ == "__main__":
    main()

