#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_sr_lab.data import StereoSRDataset, build_pairs, build_pairs_from_sources, split_pairs
from stereo_sr_lab.models import count_parameters, create_model
from stereo_sr_lab.training.engine import evaluate_model, train_one_epoch
from stereo_sr_lab.training.losses import StereoSRLoss
from stereo_sr_lab.training.utils import get_device, load_config, load_checkpoint, save_checkpoint, save_json, set_seed


def override(config: dict, args) -> None:
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["runtime"]["output_dir"] = args.output_dir
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.device:
        config["runtime"]["device"] = args.device
    if args.mode:
        config["train"]["run_mode"] = args.mode
    if args.limit_train is not None:
        config["data"]["limit_train"] = args.limit_train
    if args.limit_val is not None:
        config["data"]["limit_val"] = args.limit_val


def make_train_val_pairs(config: dict) -> tuple[list, list]:
    data_cfg = config["data"]
    if "train_sources" in data_cfg:
        pairs = build_pairs_from_sources(data_cfg["train_sources"])
        if not pairs:
            raise RuntimeError("No training pairs found in configured train_sources.")
        return split_pairs(
            pairs,
            val_ratio=data_cfg.get("val_ratio", 0.1),
            seed=data_cfg.get("split_seed", config.get("seed", 42)),
            val_count=data_cfg.get("val_count"),
        )

    train_pairs = build_pairs(data_cfg["root"], data_cfg["dataset"], data_cfg["train_split"])
    val_pairs = build_pairs(data_cfg["root"], data_cfg["dataset"], data_cfg["val_split"])
    return train_pairs, val_pairs


def make_loader(
    config: dict,
    split_name: str,
    shuffle: bool,
    device: torch.device,
    *,
    eval_mode: bool = False,
    limit_override: int | None = None,
    pairs: list | None = None,
) -> DataLoader:
    data_cfg = config["data"]
    if pairs is None:
        split = data_cfg[f"{split_name}_split"]
        pairs = build_pairs(data_cfg["root"], data_cfg["dataset"], split)
    else:
        pairs = list(pairs)
    limit = data_cfg.get(f"limit_{split_name}", 0) if limit_override is None else limit_override
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"No {split_name} pairs found.")

    dataset = StereoSRDataset(
        pairs=pairs,
        scale=data_cfg["scale"],
        split="eval" if eval_mode else ("train" if split_name == "train" else "eval"),
        hr_patch_size=0 if eval_mode else data_cfg.get("hr_patch_size", 0),
        eval_crop_size=data_cfg.get("eval_crop_size", 0),
        augment=False if eval_mode else data_cfg.get("augment", False) and split_name == "train",
        fixed_crop=False if eval_mode else config["train"].get("run_mode") == "overfit",
    )
    return DataLoader(
        dataset,
        batch_size=1 if eval_mode or split_name != "train" else config["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stereo super-resolution model.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stereo_sr_x2.json"))
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--mode", choices=["train", "overfit"])
    parser.add_argument("--resume")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--eval-train-every", type=int)
    parser.add_argument("--eval-train-limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    override(config, args)
    if args.eval_train_every is not None:
        config["train"]["eval_train_every"] = args.eval_train_every
    if args.eval_train_limit is not None:
        config["train"]["eval_train_limit"] = args.eval_train_limit
    set_seed(config.get("seed", 42))
    device = get_device(config["runtime"].get("device", "cuda"))
    out_dir = Path(config["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["train"]
    run_mode = train_cfg.get("run_mode", "train")
    if run_mode == "overfit":
        config["data"]["limit_train"] = train_cfg["batch_size"]
        config["data"]["augment"] = False
        print("run mode: overfit one fixed training batch")
    else:
        print("run mode: train")
    save_json(config, out_dir / "config.json")

    train_pairs, val_pairs = make_train_val_pairs(config)
    print(f"data pairs: train={len(train_pairs)} val={len(val_pairs)}")
    train_loader = make_loader(config, "train", shuffle=run_mode != "overfit", device=device, pairs=train_pairs)
    val_loader = train_loader if run_mode == "overfit" else make_loader(
        config,
        "val",
        shuffle=False,
        device=device,
        pairs=val_pairs,
    )
    train_eval_loader = None
    train_eval_every = train_cfg.get("eval_train_every", 0)
    if run_mode != "overfit" and train_eval_every:
        train_eval_loader = make_loader(
            config,
            "train",
            shuffle=False,
            device=device,
            eval_mode=True,
            limit_override=train_cfg.get("eval_train_limit", 0),
            pairs=train_pairs,
        )
    model = create_model(config).to(device)
    print(f"model parameters: {count_parameters(model) / 1e6:.3f} M")

    loss_cfg = config.get("loss", {})
    criterion = StereoSRLoss(**loss_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=train_cfg.get("eta_min", 1e-6),
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("amp", False) and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", False) and device.type == "cuda")

    start_epoch = 1
    best_psnr = 0.0
    if args.resume:
        last_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = last_epoch + 1

    history = []
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
        scheduler.step()
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train": train_stats}

        if train_eval_loader is not None and epoch % train_eval_every == 0:
            train_eval_stats = evaluate_model(
                model,
                train_eval_loader,
                device,
                scale=config["data"]["scale"],
                amp=train_cfg.get("amp", False),
            )
            record["train_eval"] = train_eval_stats
            print(
                f"epoch {epoch:03d} train_eval | "
                f"PSNR={train_eval_stats['psnr']:.3f} SSIM={train_eval_stats['ssim']:.4f}"
            )

        if epoch % train_cfg.get("validate_every", 1) == 0:
            val_stats = evaluate_model(
                model,
                val_loader,
                device,
                scale=config["data"]["scale"],
                amp=train_cfg.get("amp", False),
            )
            record["val"] = val_stats
            print(f"epoch {epoch:03d} val | PSNR={val_stats['psnr']:.3f} SSIM={val_stats['ssim']:.4f}")
            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, best_psnr, config)

        if epoch % train_cfg.get("save_every", 1) == 0:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, scheduler, epoch, best_psnr, config)
        history.append(record)
        save_json(history, out_dir / "history.json")


if __name__ == "__main__":
    main()

