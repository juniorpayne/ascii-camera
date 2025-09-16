"""ASCII webcam virtual camera filter.

Capture frames from a webcam, convert them to ASCII art, render to an image, and
publish the result through a virtual camera device using pyvirtualcam.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2  # type: ignore
import numpy as np
import pyvirtualcam  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore

ASCII_CHARS = " .:-=+*#%@"
CHAR_ASPECT = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose an ASCII-art filtered webcam via a virtual camera device."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Zero-based input camera index (default: 0).",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=120,
        help="Number of ASCII characters per row (default: 120).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frame rate for the virtual camera (default: 20).",
    )
    parser.add_argument(
        "--charset",
        default=ASCII_CHARS,
        help=f"Characters used for rendering (default: {ASCII_CHARS!r}).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert brightness so bright pixels map to darker glyphs.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size (pixels) for drawing ASCII characters (default: 14).",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a TTF font file (monospace strongly recommended).",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Width in pixels for the virtual camera stream. Defaults to rendered width.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Height in pixels for the virtual camera stream. Defaults to rendered height.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the ASCII output horizontally (useful for self-view).",
    )
    return parser.parse_args()


def resolve_font(font_path: Optional[Path], font_size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except OSError as exc:
            raise RuntimeError(f"Could not load font at {font_path}: {exc}") from exc
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\consola.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def frame_to_ascii(
    gray_frame: np.ndarray, columns: int, charset: np.ndarray, invert: bool
) -> list[str]:
    height, width = gray_frame.shape
    target_height = max(1, int((height / width) * columns * CHAR_ASPECT))
    resized = cv2.resize(
        gray_frame,
        (columns, target_height),
        interpolation=cv2.INTER_AREA if columns < width else cv2.INTER_LINEAR,
    )
    if invert:
        resized = 255 - resized
    normalized = resized.astype(np.float32) / 255.0
    indices = np.clip(
        np.rint(normalized * (len(charset) - 1)).astype(np.int32),
        0,
        len(charset) - 1,
    )
    ascii_grid = charset[indices]
    return ["".join(row) for row in ascii_grid]


def _text_size(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return font.getsize(text)


def ascii_lines_to_image(
    lines: Iterable[str],
    font: ImageFont.ImageFont,
    foreground: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] = (0, 0, 0),
    line_spacing: Optional[int] = None,
) -> Image.Image:
    lines = [line.rstrip("\n") for line in lines]
    if not lines:
        return Image.new("RGB", (16, 16), background)

    max_line_width = max(_text_size(font, line if line else " ")[0] for line in lines)
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (0, 0)
    base_height = ascent + descent if ascent + descent > 0 else _text_size(font, "M")[1]
    spacing = line_spacing if line_spacing is not None else max(2, base_height // 6)

    margin_x = max(4, max_line_width // 50)
    margin_y = max(4, spacing)
    line_height = base_height + spacing

    img_width = margin_x * 2 + max_line_width
    img_height = margin_y * 2 + len(lines) * line_height

    image = Image.new("RGB", (img_width, img_height), background)
    draw = ImageDraw.Draw(image)

    y = margin_y
    for line in lines:
        draw.text((margin_x, y), line, font=font, fill=foreground)
        y += line_height
    return image


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    args = parse_args()

    charset_raw = args.charset or ASCII_CHARS
    if len(charset_raw) < 2:
        print("The character set must contain at least two characters.", file=sys.stderr)
        return 1
    charset = np.array(list(charset_raw))

    try:
        font = resolve_font(args.font_path, args.font_size)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print(f"Could not open camera device {args.device}.", file=sys.stderr)
        return 1

    frame_interval = 0.0 if args.fps <= 0 else 1.0 / args.fps
    next_frame_time = time.perf_counter()

    try:
        ok, frame = capture.read()
        if not ok:
            print("Failed to read initial frame from camera.", file=sys.stderr)
            return 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ascii_lines = frame_to_ascii(gray, args.columns, charset, args.invert)
        ascii_image = ascii_lines_to_image(ascii_lines, font)
        frame_rgb = np.array(ascii_image)
        height, width, _ = frame_rgb.shape

        cam_width = args.output_width or width
        cam_height = args.output_height or height

        try:
            virtual_cam_ctx = pyvirtualcam.Camera(
                width=cam_width,
                height=cam_height,
                fps=max(1, int(args.fps if args.fps > 0 else 20)),
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
        except RuntimeError as exc:
            print(
                "\nUnable to start a virtual camera.\n"
                "pyvirtualcam could not find a working backend.\n"
                "Make sure a virtual camera driver is installed (OBS Studio 26+\n"
                "with the Virtual Camera component, or another driver listed in\n"
                "the pyvirtualcam documentation). After installation, restart\n"
                "your machine and rerun this script.",
                file=sys.stderr,
            )
            print(f"\nBackend error:\n{exc}\n", file=sys.stderr)
            return 1

        with virtual_cam_ctx as virtual_cam:
            print(
                f"Virtual camera started: {virtual_cam.device}.\n"
                "Select this camera in your video conferencing software."
            )

            while True:
                if frame_interval > 0:
                    now = time.perf_counter()
                    if now < next_frame_time:
                        time.sleep(next_frame_time - now)
                    next_frame_time = max(next_frame_time + frame_interval, time.perf_counter())

                ok, frame = capture.read()
                if not ok:
                    print("Failed to read from camera. Exiting.", file=sys.stderr)
                    return 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ascii_lines = frame_to_ascii(gray, args.columns, charset, args.invert)
                ascii_image = ascii_lines_to_image(ascii_lines, font)
                frame_rgb = np.array(ascii_image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if args.mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                frame_bgr = resize_frame(frame_bgr, cam_width, cam_height)
                virtual_cam.send(frame_bgr)
                virtual_cam.sleep_until_next_frame()
    except KeyboardInterrupt:
        return 0
    finally:
        capture.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
