#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def spectrum(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(array))
    mag = np.log1p(np.abs(fft))
    mag = (mag - mag.min()) / max(mag.max() - mag.min(), 1e-8)
    return Image.fromarray((mag * 255).astype(np.uint8), mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save log Fourier spectra for visual comparison.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("spectra"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        spectrum(image_path).save(args.out_dir / f"{image_path.stem}_spectrum.png")


if __name__ == "__main__":
    main()

