import time
from contextlib import nullcontext

import torch

from .metrics import AverageMeter, psnr, ssim


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int, config: dict) -> dict:
    model.train()
    meters: dict[str, AverageMeter] = {}
    amp = config["train"].get("amp", False)
    log_every = config["train"].get("log_every", 20)
    if config["train"].get("run_mode") == "overfit":
        log_every = 1
    clip_norm = config["train"].get("clip_grad_norm", 0.0)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            outputs = model(batch["lr_left"], batch["lr_right"], return_attention=True)
            loss, parts = criterion(outputs, batch)

        scaler.scale(loss).backward()
        if clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["lr_left"].shape[0]
        for key, value in parts.items():
            meters.setdefault(key, AverageMeter()).update(value.item(), batch_size)

        if step % log_every == 0:
            message = ", ".join(f"{key}={meter.avg:.4f}" for key, meter in meters.items())
            print(f"epoch {epoch:03d} step {step:04d}/{len(loader)} | {message}", flush=True)

    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate_model(model, loader, device, scale: int, amp: bool = False, max_batches: int = 0) -> dict:
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    time_meter = AverageMeter()

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast_context(device, amp):
            outputs = model(batch["lr_left"], batch["lr_right"], return_attention=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        batch_size = batch["lr_left"].shape[0]
        for side in ("left", "right"):
            pred = outputs[f"sr_{side}"]
            target = batch[f"hr_{side}"]
            psnr_meter.update(psnr(pred, target, crop_border=scale), batch_size)
            ssim_meter.update(ssim(pred, target, crop_border=scale), batch_size)
        time_meter.update(elapsed / batch_size, batch_size)

    return {
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
        "seconds_per_pair": time_meter.avg,
    }

