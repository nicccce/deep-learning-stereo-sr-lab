import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def save_checkpoint(path: str | Path, model, optimizer, scheduler, epoch: int, best_psnr: float, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_psnr": best_psnr,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_psnr", 0.0))

