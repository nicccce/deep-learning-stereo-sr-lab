from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
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
    lefts: dict[tuple[Path, str], Path] = {}
    rights: dict[tuple[Path, str], Path] = {}

    for path in files:
        stem = path.stem
        if stem.endswith("_L"):
            lefts[(path.parent, stem[:-2])] = path
        elif stem.endswith("_R"):
            rights[(path.parent, stem[:-2])] = path
        elif stem.endswith("_left"):
            lefts[(path.parent, stem[:-5])] = path
        elif stem.endswith("_right"):
            rights[(path.parent, stem[:-6])] = path

    keys = sorted(lefts.keys() & rights.keys(), key=lambda item: (str(item[0]), item[1]))
    name_counts = Counter(name for _, name in keys)
    pairs = []
    for key in keys:
        folder, name = key
        pair_name = name if name_counts[name] == 1 else f"{folder.name}_{name}"
        pairs.append(StereoPair(lefts[key], rights[key], pair_name, source))
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


def scan_kitti_stereo_flow(root: str | Path, split: str = "training") -> list[StereoPair]:
    root = Path(root)
    if split.lower() == "all":
        pairs = []
        for split_name in ("training", "testing"):
            for pair in scan_kitti_stereo_flow(root, split_name):
                pairs.append(StereoPair(pair.left, pair.right, f"{split_name}_{pair.name}", pair.source))
        return pairs

    split_root = _split_root(root, split)
    for left_name, right_name in (("colored_0", "colored_1"), ("image_2", "image_3")):
        left_dir = split_root / left_name
        right_dir = split_root / right_name
        if not left_dir.is_dir() or not right_dir.is_dir():
            continue

        rights = {path.name: path for path in _images(right_dir)}
        pairs = []
        for left in _images(left_dir):
            right = rights.get(left.name)
            if right is not None:
                pairs.append(StereoPair(left, right, left.stem, f"kitti_stereo_flow/{split}"))
        if pairs:
            return pairs
    return []


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
    if name in {"kitti", "kitti_stereo", "kitti_stereo_flow", "kitti2012"}:
        return scan_kitti_stereo_flow(root, split or "training")
    if name in {"kitti_depth", "kitti_depth_selection"}:
        return scan_kitti_depth_selection(root, split or "val_selection_cropped")
    if name in {"folder", "custom"}:
        return scan_folder_pairs(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_pairs_from_source(source: dict) -> list[StereoPair]:
    if "dataset" not in source or "root" not in source:
        raise ValueError("Data source must define 'dataset' and 'root'.")
    split = source.get("split")
    return build_pairs(source["root"], source["dataset"], split)


def build_pairs_from_sources(sources: list[dict]) -> list[StereoPair]:
    pairs = []
    for source in sources:
        pairs.extend(build_pairs_from_source(source))
    return pairs


def split_pairs(
    pairs: list[StereoPair],
    val_ratio: float = 0.1,
    seed: int = 42,
    val_count: int | None = None,
) -> tuple[list[StereoPair], list[StereoPair]]:
    pairs = list(pairs)
    if len(pairs) < 2:
        return pairs, []

    if val_count is None:
        val_count = round(len(pairs) * val_ratio)
        if val_ratio > 0:
            val_count = max(1, val_count)
    val_count = max(0, min(int(val_count), len(pairs) - 1))

    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_count])
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in val_indices]
    val_pairs = [pair for index, pair in enumerate(pairs) if index in val_indices]
    return train_pairs, val_pairs

