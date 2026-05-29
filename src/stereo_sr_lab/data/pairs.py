from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class StereoPair:
    left: Path
    right: Path
    name: str
    source: str


def _images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def _split_root(root: Path, split: str | None) -> Path:
    if split and (root / split).is_dir():
        return root / split
    return root


def _pair_lr_suffix(files: Iterable[Path], source: str) -> list[StereoPair]:
    lefts: dict[str, Path] = {}
    rights: dict[str, Path] = {}

    for path in files:
        stem = path.stem
        if stem.endswith("_L"):
            lefts[stem[:-2]] = path
        elif stem.endswith("_R"):
            rights[stem[:-2]] = path
        elif stem.endswith("_left"):
            lefts[stem[:-5]] = path
        elif stem.endswith("_right"):
            rights[stem[:-6]] = path

    pairs = []
    for key in sorted(lefts.keys() & rights.keys()):
        pairs.append(StereoPair(lefts[key], rights[key], key, source))
    return pairs


def scan_flickr1024(root: str | Path, split: str = "Train") -> list[StereoPair]:
    folder = _split_root(Path(root), split)
    return _pair_lr_suffix(_images(folder), f"flickr1024/{split}")


def scan_middlebury(root: str | Path) -> list[StereoPair]:
    root = Path(root)
    pairs = []
    for left in sorted(root.rglob("im0.png")):
        scene_dir = left.parent
        right = None
        for name in ("im1.png", "im1E.png", "im1L.png"):
            candidate = scene_dir / name
            if candidate.is_file():
                right = candidate
                break
        if right is not None:
            pairs.append(StereoPair(left, right, scene_dir.name, "middlebury2014"))
    return pairs


def scan_kitti_depth_selection(root: str | Path, split: str = "val_selection_cropped") -> list[StereoPair]:
    root = Path(root)
    candidates = [
        root / "depth_selection" / split / "image",
        root / split / "image",
        root / "image",
        root,
    ]
    image_dir = next((path for path in candidates if path.is_dir()), None)
    if image_dir is None:
        return []

    lefts: dict[str, Path] = {}
    rights: dict[str, Path] = {}
    for path in _images(image_dir):
        stem = path.stem
        if stem.endswith("_image_02"):
            lefts[stem[:-9]] = path
        elif stem.endswith("_image_03"):
            rights[stem[:-9]] = path

    pairs = [
        StereoPair(lefts[key], rights[key], key, f"kitti_depth/{split}")
        for key in sorted(lefts.keys() & rights.keys())
    ]
    if pairs:
        return pairs

    # Some KITTI depth-completion subsets contain only a single rectified image.
    # Mirroring left/right keeps inference and timing scripts usable, but it is
    # not a valid stereo-quality benchmark.
    return [StereoPair(path, path, path.stem, f"kitti_depth/{split}/single") for path in _images(image_dir)]


def scan_folder_pairs(root: str | Path) -> list[StereoPair]:
    root = Path(root)
    for left_name, right_name in (("left", "right"), ("L", "R"), ("hr_left", "hr_right")):
        left_dir = root / left_name
        right_dir = root / right_name
        if left_dir.is_dir() and right_dir.is_dir():
            rights = {path.stem: path for path in _images(right_dir)}
            pairs = []
            for left in _images(left_dir):
                if left.stem in rights:
                    pairs.append(StereoPair(left, rights[left.stem], left.stem, "folder"))
            if pairs:
                return pairs
    return _pair_lr_suffix(_images(root), "folder")


def build_pairs(root: str | Path, dataset: str, split: str | None = None) -> list[StereoPair]:
    name = dataset.lower()
    if name == "flickr1024":
        return scan_flickr1024(root, split or "Train")
    if name in {"middlebury", "middlebury2014"}:
        return scan_middlebury(root)
    if name in {"kitti", "kitti_depth", "kitti_depth_selection"}:
        return scan_kitti_depth_selection(root, split or "val_selection_cropped")
    if name in {"folder", "custom"}:
        return scan_folder_pairs(root)
    raise ValueError(f"Unsupported dataset: {dataset}")

