#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "train": (25, 106, 179),
    "validation": (214, 94, 37),
}


def load_points(history: list[dict], section: str, metric: str) -> list[tuple[int, float]]:
    points = []
    for record in history:
        values = record.get(section)
        if values and metric in values:
            points.append((int(record["epoch"]), float(values[metric])))
    return points


def value_range(series: list[tuple[str, tuple[int, int, int], list[tuple[int, float]]]]) -> tuple[float, float]:
    values = [value for _, _, points in series for _, value in points]
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.05, 0.01)
    else:
        pad = (high - low) * 0.12
    return low - pad, high + pad


def epoch_range(series: list[tuple[str, tuple[int, int, int], list[tuple[int, float]]]]) -> tuple[int, int]:
    epochs = [epoch for _, _, points in series for epoch, _ in points]
    if not epochs:
        return 1, 2
    start = min(epochs)
    end = max(epochs)
    return (start, end + 1) if start == end else (start, end)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [low]
    step = (high - low) / (count - 1)
    return [low + step * idx for idx in range(count)]


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    y_label: str,
    series: list[tuple[str, tuple[int, int, int], list[tuple[int, float]]]],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    plot_left = left + 78
    plot_top = top + 48
    plot_right = right - 34
    plot_bottom = bottom - 58
    axis_color = (45, 52, 62)
    grid_color = (220, 226, 234)
    text_color = (31, 36, 44)

    draw.text((left, top), title, fill=text_color, font=font)
    draw.text((left, top + 22), y_label, fill=(88, 96, 107), font=font)

    x_min, x_max = epoch_range(series)
    y_min, y_max = value_range(series)

    def x_pos(epoch: int) -> int:
        return int(plot_left + (epoch - x_min) / max(x_max - x_min, 1) * (plot_right - plot_left))

    def y_pos(value: float) -> int:
        return int(plot_bottom - (value - y_min) / max(y_max - y_min, 1e-12) * (plot_bottom - plot_top))

    for value in nice_ticks(y_min, y_max):
        y = y_pos(value)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        draw.text((left + 4, y - 7), f"{value:.3f}", fill=(88, 96, 107), font=font)

    for epoch in range(x_min, x_max + 1):
        x = x_pos(epoch)
        draw.line((x, plot_bottom, x, plot_bottom + 4), fill=axis_color, width=1)
        draw.text((x - 5, plot_bottom + 12), str(epoch), fill=(88, 96, 107), font=font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)

    legend_x = plot_right - 220
    legend_y = top + 4
    for idx, (label, color, _) in enumerate(series):
        y = legend_y + idx * 22
        draw.line((legend_x, y + 7, legend_x + 28, y + 7), fill=color, width=4)
        draw.text((legend_x + 36, y), label, fill=text_color, font=font)

    for _, color, points in series:
        if not points:
            continue
        pixel_points = [(x_pos(epoch), y_pos(value)) for epoch, value in points]
        if len(pixel_points) > 1:
            draw.line(pixel_points, fill=color, width=4)
        for x, y in pixel_points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/validation PSNR and SSIM curves from history.json.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    history_path = Path(args.history)
    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    image = Image.new("RGB", (width, height), (250, 252, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    psnr_series = [
        ("train split", COLORS["train"], load_points(history, "train_eval", "psnr")),
        ("validation split", COLORS["validation"], load_points(history, "val", "psnr")),
    ]
    ssim_series = [
        ("train split", COLORS["train"], load_points(history, "train_eval", "ssim")),
        ("validation split", COLORS["validation"], load_points(history, "val", "ssim")),
    ]

    draw.text((36, 24), f"Training History: {history_path.parent.name}", fill=(22, 28, 36), font=font)
    draw_panel(draw, (36, 72, 1244, 370), "PSNR by epoch", "higher is better", psnr_series, font)
    draw_panel(draw, (36, 402, 1244, 700), "SSIM by epoch", "higher is better", ssim_series, font)

    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
