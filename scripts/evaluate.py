#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_sr_lab.data import StereoSRDataset, build_pairs_from_source
from stereo_sr_lab.models import count_parameters, create_model
from stereo_sr_lab.training.engine import evaluate_model
from stereo_sr_lab.training.utils import get_device, load_checkpoint, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSNR, SSIM, and inference time.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stereo_sr_x2.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-crop-size", type=int)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    source = dict(
        data_cfg.get("test_source")
        or {
            "dataset": data_cfg["dataset"],
            "root": data_cfg["root"],
            "split": data_cfg.get("val_split", "Validation"),
        }
    )
    if args.data_root:
        source["root"] = args.data_root
    if args.dataset:
        source["dataset"] = args.dataset
    if args.split:
        source["split"] = args.split
    if args.eval_crop_size is not None:
        data_cfg["eval_crop_size"] = args.eval_crop_size
    if args.device:
        config["runtime"]["device"] = args.device

    device = get_device(config["runtime"].get("device", "cuda"))
    pairs = build_pairs_from_source(source)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No evaluation pairs found.")

    dataset = StereoSRDataset(
        pairs,
        scale=config["data"]["scale"],
        split="eval",
        eval_crop_size=config["data"].get("eval_crop_size", 0),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["data"].get("num_workers", 4))
    model = create_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate_model(model, loader, device, config["data"]["scale"], amp=False)
    metrics["parameters"] = count_parameters(model)
    metrics["num_pairs"] = len(dataset)
    save_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()

