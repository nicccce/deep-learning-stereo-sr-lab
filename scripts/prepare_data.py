#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        archive.extractall(destination)


def extract_if_needed(
    zip_path: Path,
    destination: Path,
    force: bool = False,
    marker_name: str = ".extracted",
) -> None:
    if not zip_path.is_file():
        print(f"skip missing {zip_path}")
        return
    marker = destination / marker_name
    if marker.exists() and not force:
        print(f"skip existing {destination}")
        return
    print(f"extract {zip_path} -> {destination}")
    safe_extract(zip_path, destination)
    marker.write_text(str(zip_path), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local stereo SR datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets"))
    parser.add_argument("--only", nargs="*", default=["flickr1024", "middlebury", "kitti"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root
    out_root = args.out_root

    if "flickr1024" in args.only:
        extract_if_needed(
            data_root / "flickr1024" / "Flickr1024.zip",
            out_root / "flickr1024",
            force=args.force,
        )

    if "middlebury" in args.only:
        source_dir = data_root / "middlebury_2014" / "perfect_train"
        target_dir = out_root / "middlebury_2014" / "perfect_train"
        for zip_path in sorted(source_dir.glob("*.zip")):
            extract_if_needed(
                zip_path,
                target_dir,
                force=args.force,
                marker_name=f".{zip_path.stem}.extracted",
            )

    if "kitti" in args.only:
        extract_if_needed(
            data_root / "kitti" / "data_depth_selection.zip",
            out_root / "kitti",
            force=args.force,
        )


if __name__ == "__main__":
    main()

