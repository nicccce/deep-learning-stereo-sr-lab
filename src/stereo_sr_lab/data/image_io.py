from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def read_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_bicubic(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return image.resize(size, BICUBIC)


def mod_crop(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    width -= width % scale
    height -= height % scale
    return image.crop((0, 0, width, height))


def crop_to_common(left: Image.Image, right: Image.Image) -> tuple[Image.Image, Image.Image]:
    width = min(left.size[0], right.size[0])
    height = min(left.size[1], right.size[1])
    return left.crop((0, 0, width, height)), right.crop((0, 0, width, height))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().clamp(0, 1).cpu()
    if image.ndim == 4:
        image = image[0]
    array = image.permute(1, 2, 0).numpy()
    array = (array * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)

