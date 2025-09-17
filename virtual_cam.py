"""ASCII webcam virtual camera filter with shared render pipeline."""
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

import main as ascii_main

ASCII_CHARS = ascii_main.ASCII_CHARS
CHAR_ASPECT = ascii_main.CHAR_ASPECT
MODE_ORDER = ascii_main.MODE_ORDER
THEME_ROTATION = ascii_main.THEME_ROTATION
COLOR_THEMES = ascii_main.COLOR_THEMES
MATRIX_STATE_FACTORIES = ascii_main.MATRIX_STATE_FACTORIES


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
        "--accelerator",
        choices=["auto", "cpu", "cuda", "torch"],
        default="auto",
        help="Preprocessing backend (default: auto).",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index when using the CUDA accelerator.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Device string for the torch accelerator (e.g. cuda:0, mps, dml:0).",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_ORDER,
        default="classic",
        help="ASCII render mode to apply (classic, matrix, matrix-void).",
    )
    parser.add_argument(
        "--theme",
        choices=["auto", *THEME_ROTATION],
        default="auto",
        help="Color theme for ASCII output (auto selects based on mode).",
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


def _text_size(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return font.getsize(text)


ANSI_BASE_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]


def ansi_256_to_rgb(code: int) -> tuple[int, int, int]:
    if code < 0:
        return (255, 255, 255)
    if code < 16:
        return ANSI_BASE_COLORS[code]
    if code < 232:
        code -= 16
        r = code // 36
        g = (code % 36) // 6
        b = code % 6
        levels = [0, 95, 135, 175, 215, 255]
        return (levels[r], levels[g], levels[b])
    level = 8 + (code - 232) * 10
    return (level, level, level)


def compute_color_codes(
    intensity: np.ndarray,
    theme: str,
    overrides: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    palette = COLOR_THEMES.get(theme) or ()
    if not palette:
        return None
    bins = np.clip(
        np.rint(intensity * (len(palette) - 1)).astype(np.int32),
        0,
        len(palette) - 1,
    )
    palette_array = np.array(palette, dtype=np.int32)
    if overrides is None:
        overrides = np.full_like(bins, -1, dtype=np.int32)
    codes = np.where(overrides >= 0, overrides, palette_array[bins])
    return codes.astype(np.int32)


def color_codes_to_rgb(codes: np.ndarray) -> np.ndarray:
    flat = codes.reshape(-1)
    rgb = np.array([ansi_256_to_rgb(int(code)) for code in flat], dtype=np.uint8)
    return rgb.reshape(codes.shape + (3,))


def grayscale_to_rgb(intensity: np.ndarray) -> np.ndarray:
    values = np.clip((intensity * 255.0).astype(np.uint8), 0, 255)
    return np.stack([values, values, values], axis=-1)


def render_char_grid_to_image(
    chars: np.ndarray,
    intensity: np.ndarray,
    rgb_colors: Optional[np.ndarray],
    theme: str,
    font: ImageFont.ImageFont,
    background: tuple[int, int, int] = (0, 0, 0),
    line_spacing: Optional[int] = None,
) -> Image.Image:
    rows, cols = chars.shape
    if rows == 0 or cols == 0:
        return Image.new("RGB", (16, 16), background)

    char_width, char_height = _text_size(font, "M")
    char_width = max(1, char_width)
    char_height = max(1, char_height)
    spacing = line_spacing if line_spacing is not None else max(2, char_height // 6)
    margin_x = max(4, char_width // 2)
    margin_y = max(4, spacing)

    img_width = margin_x * 2 + char_width * cols
    img_height = margin_y * 2 + rows * (char_height + spacing)

    image = Image.new("RGB", (img_width, img_height), background)
    draw = ImageDraw.Draw(image)
    grayscale_cache = None if rgb_colors is not None else grayscale_to_rgb(intensity)

    for row in range(rows):
        y = margin_y + row * (char_height + spacing)
        for col in range(cols):
            ch = chars[row, col]
            if ch == " ":
                continue
            if rgb_colors is not None:
                color = tuple(int(c) for c in rgb_colors[row, col])
            else:
                gray = grayscale_cache[row, col]
                color = tuple(int(c) for c in gray)
            draw.text((margin_x + col * char_width, y), ch, font=font, fill=color)
    return image


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

    try:
        preprocessor = ascii_main.create_preprocessor(
            args.accelerator,
            args.cuda_device,
            args.torch_device,
        )
    except RuntimeError as exc:
        print(f"Failed to initialize {args.accelerator} accelerator: {exc}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print(f"Could not open camera device {args.device}.", file=sys.stderr)
        return 1

    matrix_states = {name: factory() for name, factory in MATRIX_STATE_FACTORIES.items()}

    theme_locked = args.theme != "auto"
    current_mode = args.mode
    current_theme = args.theme if theme_locked else ascii_main.default_theme_for_mode(current_mode)

    frame_interval = 0.0 if args.fps <= 0 else 1.0 / args.fps
    next_frame_time = time.perf_counter()

    columns = max(2, args.columns)

    controls: Optional[ascii_main.ControlPoller] = None

    try:
        controls = ascii_main.ControlPoller()
        if controls.mode == "none":
            print("Controls: Ctrl+C to quit (runtime mode switching unavailable on this terminal).", file=sys.stderr)
        else:
            print("Controls: q quit | m mode | c cycle colors | g green | a amber | o orange | n none | r reset", file=sys.stderr)
            print("Matrix tuning: [ ] head threshold | , . decay | ; ' activation floor | + - contrast", file=sys.stderr)

        ok, frame = capture.read()
        if not ok:
            print("Failed to read initial frame from camera.", file=sys.stderr)
            return 1

        gray = preprocessor.preprocess(frame, columns)
        ascii_chars, intensity = ascii_main.frame_to_ascii(gray, charset, args.invert)

        display_chars = ascii_chars
        display_intensity = intensity
        overrides: Optional[np.ndarray] = None
        state = matrix_states.get(current_mode)
        if state is not None:
            display_chars, display_intensity, overrides = state.apply(ascii_chars, intensity)

        active_theme = current_theme
        if not theme_locked and args.theme == "auto":
            active_theme = ascii_main.default_theme_for_mode(current_mode)

        color_codes = compute_color_codes(display_intensity, active_theme, overrides)
        rgb_colors = color_codes_to_rgb(color_codes) if color_codes is not None else None

        ascii_image = render_char_grid_to_image(display_chars, display_intensity, rgb_colors, active_theme, font)
        frame_rgb = np.array(ascii_image)
        cam_width = args.output_width or frame_rgb.shape[1]
        cam_height = args.output_height or frame_rgb.shape[0]

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

                gray = preprocessor.preprocess(frame, columns)
                ascii_chars, intensity = ascii_main.frame_to_ascii(
                    gray, charset, args.invert
                )

                display_chars = ascii_chars
                display_intensity = intensity
                overrides = None
                state = matrix_states.get(current_mode)
                if state is not None:
                    display_chars, display_intensity, overrides = state.apply(
                        ascii_chars, intensity
                    )

                active_theme = current_theme
                if not theme_locked and args.theme == "auto":
                    active_theme = ascii_main.default_theme_for_mode(current_mode)

                color_codes = compute_color_codes(display_intensity, active_theme, overrides)
                rgb_colors = (
                    color_codes_to_rgb(color_codes) if color_codes is not None else None
                )

                ascii_image = render_char_grid_to_image(
                    display_chars, display_intensity, rgb_colors, active_theme, font
                )
                frame_rgb = np.array(ascii_image)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if args.mirror:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                frame_bgr = cv2.resize(
                    frame_bgr,
                    (cam_width, cam_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                virtual_cam.send(frame_bgr)
                virtual_cam.sleep_until_next_frame()

                events = controls.poll() if controls else []
                for event in events:
                    if event == "quit":
                        return 0
                    if event == "mode_next":
                        index = (MODE_ORDER.index(current_mode) + 1) % len(MODE_ORDER)
                        previous_mode = current_mode
                        current_mode = MODE_ORDER[index]
                        if (
                            current_mode in matrix_states
                            and previous_mode != current_mode
                        ):
                            matrix_states[current_mode].reset()
                        if not theme_locked and args.theme == "auto":
                            current_theme = ascii_main.default_theme_for_mode(current_mode)
                    elif event == "theme_next":
                        theme_locked = True
                        base = current_theme if current_theme in THEME_ROTATION else "none"
                        idx = (THEME_ROTATION.index(base) + 1) % len(THEME_ROTATION)
                        current_theme = THEME_ROTATION[idx]
                    elif event == "theme_green":
                        theme_locked = True
                        current_theme = "green"
                    elif event == "theme_amber":
                        theme_locked = True
                        current_theme = "amber"
                    elif event == "theme_orange":
                        theme_locked = True
                        current_theme = "orange"
                    elif event == "theme_none":
                        theme_locked = True
                        current_theme = "none"
                    elif event == "theme_reset":
                        theme_locked = False
                        current_theme = ascii_main.default_theme_for_mode(current_mode)
                    elif event == "matrix_threshold_up":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_head_threshold(0.02)
                            print(f"Matrix head threshold: {value:.2f}", file=sys.stderr)
                    elif event == "matrix_threshold_down":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_head_threshold(-0.02)
                            print(f"Matrix head threshold: {value:.2f}", file=sys.stderr)
                    elif event == "matrix_decay_up":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_decay(0.02)
                            print(f"Matrix trail decay: {value:.2f}", file=sys.stderr)
                    elif event == "matrix_decay_down":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_decay(-0.02)
                            print(f"Matrix trail decay: {value:.2f}", file=sys.stderr)
                    elif event == "matrix_floor_up":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_activation_floor(0.002)
                            print(f"Matrix activation floor: {value:.3f}", file=sys.stderr)
                    elif event == "matrix_floor_down":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_activation_floor(-0.002)
                            print(f"Matrix activation floor: {value:.3f}", file=sys.stderr)
                    elif event == "matrix_contrast_up":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_contrast_gain(0.1)
                            print(f"Matrix contrast gain: {value:.2f}", file=sys.stderr)
                    elif event == "matrix_contrast_down":
                        state = matrix_states.get(current_mode)
                        if state is not None:
                            value = state.adjust_contrast_gain(-0.1)
                            print(f"Matrix contrast gain: {value:.2f}", file=sys.stderr)
    except KeyboardInterrupt:
        return 0
    finally:
        if controls is not None:
            controls.close()
        capture.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
