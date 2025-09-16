"""ASCII camera streamer.

Capture frames from a webcam and render them as ASCII art directly in the terminal.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from typing import Optional

import cv2  # type: ignore
import numpy as np

try:  # Windows-specific keyboard polling
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows runtimes
    msvcrt = None  # type: ignore

ASCII_CHARS = " .:-=+*#%@"
CHAR_ASPECT = 0.55  # Typical terminal glyph height/width ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the webcam feed as ASCII art in your terminal."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Zero-based camera index passed to OpenCV (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Maximum character width of each frame. Defaults to terminal width.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Limit refresh rate in frames per second (<=0 disables throttling).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert brightness so bright pixels map to darker characters.",
    )
    parser.add_argument(
        "--charset",
        default=ASCII_CHARS,
        help=f"Characters used for rendering (default: {ASCII_CHARS!r}).",
    )
    return parser.parse_args()


def resolve_width(target_width: Optional[int]) -> int:
    term_width = shutil.get_terminal_size((80, 24)).columns
    if target_width is None:
        return max(2, term_width)
    if term_width > 0:
        return max(2, min(target_width, term_width))
    return max(2, target_width)


def frame_to_ascii(
    gray_frame: np.ndarray, width: int, charset: np.ndarray, invert: bool
) -> str:
    height, original_width = gray_frame.shape
    aspect_height = max(1, int((height / original_width) * width * CHAR_ASPECT))
    resized = cv2.resize(
        gray_frame,
        (width, aspect_height),
        interpolation=cv2.INTER_AREA if width < original_width else cv2.INTER_LINEAR,
    )

    if invert:
        resized = 255 - resized

    normalized = resized.astype(np.float32) / 255.0
    indices = np.clip(
        np.rint(normalized * (len(charset) - 1)).astype(np.int32),
        0,
        len(charset) - 1,
    )
    ascii_frame = charset[indices]
    return "\n".join("".join(row) for row in ascii_frame)


def clear_screen() -> None:
    sys.stdout.write("\x1b[2J")
    sys.stdout.flush()


def move_cursor_home() -> None:
    sys.stdout.write("\x1b[H")
    sys.stdout.flush()


def hide_cursor() -> None:
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()


def should_quit() -> bool:
    if msvcrt:
        while msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b"q", b"Q"):
                return True
    return False


def main() -> int:
    if not sys.stdout.isatty():
        print("This program needs an interactive terminal to display ASCII video.", file=sys.stderr)
        return 1

    args = parse_args()
    width = resolve_width(args.width)

    charset_raw = args.charset or ASCII_CHARS
    if len(charset_raw) < 2:
        print("The character set must contain at least two characters.", file=sys.stderr)
        return 1
    charset = np.array(list(charset_raw))

    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print(f"Could not open camera device {args.device}.", file=sys.stderr)
        return 1

    frame_interval = 0.0 if args.fps <= 0 else 1.0 / args.fps
    next_frame_time = time.perf_counter()

    clear_screen()
    hide_cursor()

    try:
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
            width = resolve_width(args.width)
            ascii_frame = frame_to_ascii(gray, width, charset, args.invert)

            move_cursor_home()
            sys.stdout.write(ascii_frame)
            sys.stdout.write("\x1b[J")  # Clear anything below the current frame
            sys.stdout.flush()

            if should_quit():
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        show_cursor()
        move_cursor_home()
        capture.release()
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
