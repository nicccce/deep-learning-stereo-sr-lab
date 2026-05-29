import random
from pathlib import Path

from PIL import ImageOps
from torch.utils.data import Dataset

from .image_io import crop_to_common, mod_crop, pil_to_tensor, read_rgb, resize_bicubic
from .pairs import StereoPair


class StereoSRDataset(Dataset):
    def __init__(
        self,
        pairs: list[StereoPair],
        scale: int,
        split: str,
        hr_patch_size: int = 0,
        eval_crop_size: int = 0,
        augment: bool = False,
        fixed_crop: bool = False,
    ) -> None:
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.pairs = pairs
        self.scale = scale
        self.split = split
        self.hr_patch_size = hr_patch_size
        self.eval_crop_size = eval_crop_size
        self.augment = augment
        self.fixed_crop = fixed_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop_train_pair(self, left, right):
        width, height = left.size
        size = min(self.hr_patch_size, width, height)
        size -= size % self.scale
        if size <= 0:
            raise ValueError(f"Patch size is too small for scale {self.scale}")
        if self.fixed_crop:
            x = (width - size) // 2
            y = (height - size) // 2
        else:
            x = random.randint(0, width - size)
            y = random.randint(0, height - size)
        return left.crop((x, y, x + size, y + size)), right.crop((x, y, x + size, y + size))

    def _crop_eval_pair(self, left, right):
        width, height = left.size
        size = min(self.eval_crop_size, width, height)
        size -= size % self.scale
        if size <= 0:
            return left, right
        x = (width - size) // 2
        y = (height - size) // 2
        return left.crop((x, y, x + size, y + size)), right.crop((x, y, x + size, y + size))

    def _augment_pair(self, left, right):
        if not self.augment:
            return left, right
        if random.random() < 0.5:
            left = ImageOps.mirror(left)
            right = ImageOps.mirror(right)
            left, right = right, left
        return left, right

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        left = read_rgb(pair.left)
        right = read_rgb(pair.right)
        left, right = crop_to_common(left, right)
        left = mod_crop(left, self.scale)
        right = mod_crop(right, self.scale)

        if self.split == "train" and self.hr_patch_size > 0:
            left, right = self._crop_train_pair(left, right)
            left, right = self._augment_pair(left, right)
        elif self.eval_crop_size > 0:
            left, right = self._crop_eval_pair(left, right)

        lr_size = (left.size[0] // self.scale, left.size[1] // self.scale)
        lr_left = resize_bicubic(left, lr_size)
        lr_right = resize_bicubic(right, lr_size)

        return {
            "lr_left": pil_to_tensor(lr_left),
            "lr_right": pil_to_tensor(lr_right),
            "hr_left": pil_to_tensor(left),
            "hr_right": pil_to_tensor(right),
            "name": pair.name,
            "source": pair.source,
            "left_path": str(Path(pair.left)),
            "right_path": str(Path(pair.right)),
        }

