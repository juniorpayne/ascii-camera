"""ASCII camera streamer.

Capture frames from a webcam and render them as ASCII art directly in the terminal.
"""
from __future__ import annotations

import argparse
import select
import shutil
import sys
import time
from typing import List, Optional

import cv2  # type: ignore
import numpy as np

try:  # Windows-specific keyboard polling
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - non-Windows runtimes
    msvcrt = None  # type: ignore

try:  # POSIX terminal helpers for runtime controls
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows runtimes
    termios = None  # type: ignore
    tty = None  # type: ignore

ASCII_CHARS = " .:-=+*#%@"
MATRIX_CHAR_POOL = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ#$%"))
CHAR_ASPECT = 0.55  # Typical terminal glyph height/width ratio
MODE_ORDER = ["classic", "matrix", "matrix-void"]
THEME_ROTATION = ["none", "green", "amber", "orange"]
COLOR_THEMES = {
    "none": (),
    "green": (22, 28, 34, 40, 46, 82),
    "amber": (94, 130, 136, 142, 214, 220),
    "orange": (94, 130, 166, 202, 208, 214),
    "matrix": (22, 28, 34, 40, 46, 82, 118),
}


class ControlPoller:
    """Non-blocking key polling that works on Windows and most POSIX shells."""

    def __init__(self) -> None:
        self.mode = "none"
        self.fd: Optional[int] = None
        self.old_settings: Optional[List[int]] = None
        if msvcrt:
            self.mode = "windows"
        elif termios and tty and sys.stdin.isatty():
            self.mode = "posix"
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            self.mode = "none"

    def close(self) -> None:
        if self.mode == "posix" and self.fd is not None and self.old_settings is not None and termios:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.old_settings = None

    def poll(self) -> List[str]:
        if self.mode == "windows":
            return self._poll_windows()
        if self.mode == "posix":
            return self._poll_posix()
        return []

    def _poll_windows(self) -> List[str]:
        events: List[str] = []
        if not msvcrt:
            return events
        while msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b"\x00", b"\xe0"):
                continue  # Skip function-key prefixes
            events.extend(self._translate(key.decode("latin1", errors="ignore")))
        return events

    def _poll_posix(self) -> List[str]:
        events: List[str] = []
        if self.fd is None:
            return events
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            events.extend(self._translate(ch))
        return events

    @staticmethod
    def _translate(ch: str) -> List[str]:
        events: List[str] = []
        if not ch:
            return events
        if ch in ("q", "Q"):
            events.append("quit")
        elif ch in ("m", "M"):
            events.append("mode_next")
        elif ch in ("c", "C"):
            events.append("theme_next")
        elif ch in ("g", "G"):
            events.append("theme_green")
        elif ch in ("a", "A", "y", "Y"):
            events.append("theme_amber")
        elif ch in ("o", "O"):
            events.append("theme_orange")
        elif ch in ("n", "N"):
            events.append("theme_none")
        elif ch in ("r", "R"):
            events.append("theme_reset")
        elif ch in ("[", "{"):
            events.append("matrix_threshold_down")
        elif ch in ("]", "}"):
            events.append("matrix_threshold_up")
        elif ch in (",", "<"):
            events.append("matrix_decay_down")
        elif ch in (".", ">"):
            events.append("matrix_decay_up")
        elif ch in (";", ":"):
            events.append("matrix_floor_down")
        elif ch in ("'", '"'):
            events.append("matrix_floor_up")
        return events


class MatrixModeState:
    """Matrix renderer that overlays animated rain on a static glyph image."""

    def __init__(self) -> None:
        self.rain = np.zeros((0, 0), dtype=np.float32)
        self.trail_chars = np.full((0, 0), " ", dtype="<U1")
        self.head_rows = np.zeros(0, dtype=np.float32)
        self.velocities = np.zeros(0, dtype=np.float32)
        self.rng = np.random.default_rng()
        self.head_threshold = 0.28
        self.decay = 0.86
        self.activation_floor = 0.05
        self.smoothed = np.zeros((0, 0), dtype=np.float32)

    def reset(self) -> None:
        self.rain = np.zeros((0, 0), dtype=np.float32)
        self.trail_chars = np.full((0, 0), " ", dtype="<U1")
        self.head_rows = np.zeros(0, dtype=np.float32)
        self.velocities = np.zeros(0, dtype=np.float32)
        self.smoothed = np.zeros((0, 0), dtype=np.float32)

    def adjust_head_threshold(self, delta: float) -> float:
        self.head_threshold = float(np.clip(self.head_threshold + delta, 0.05, 0.9))
        return self.head_threshold

    def adjust_decay(self, delta: float) -> float:
        self.decay = float(np.clip(self.decay + delta, 0.6, 0.98))
        return self.decay

    def adjust_activation_floor(self, delta: float) -> float:
        self.activation_floor = float(np.clip(self.activation_floor + delta, 0.0, 0.3))
        return self.activation_floor

    def apply(
        self, base_chars: np.ndarray, intensity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = intensity.shape
        normalized = np.clip(intensity, 0.0, 1.0).astype(np.float32)
        if self.smoothed.shape != normalized.shape:
            self.smoothed = normalized.copy()
        else:
            self.smoothed = np.clip(self.smoothed * 0.82 + normalized * 0.18, 0.0, 1.0)
        contrast_source = self.smoothed
        contrast = np.power(contrast_source, 0.7, dtype=np.float32)
        indices = np.clip(
            np.rint(contrast * (len(MATRIX_CHAR_POOL) - 1)).astype(np.int32),
            0,
            len(MATRIX_CHAR_POOL) - 1,
        )
        display_chars = MATRIX_CHAR_POOL[indices]
        overrides = np.full(intensity.shape, -1, dtype=np.int16)

        if self.rain.shape != intensity.shape:
            self.rain = np.zeros_like(intensity, dtype=np.float32)
            self.trail_chars = np.full(intensity.shape, " ", dtype="<U1")
            self.head_rows = -self.rng.uniform(0, height, size=width).astype(np.float32)
            self.velocities = self.rng.uniform(0.28, 0.65, size=width).astype(np.float32)

        self.rain *= self.decay
        fade_mask = self.rain < (self.activation_floor * 0.9)
        if np.any(fade_mask):
            self.trail_chars[fade_mask] = " "

        column_profile = 0.15 + normalized.mean(axis=0) * 0.85

        for col in range(width):
            speed = self.velocities[col] + column_profile[col] * 0.55 + 0.35
            self.head_rows[col] += speed
            if self.head_rows[col] < -2.0:
                self.head_rows[col] = -2.0
            if self.head_rows[col] >= height + 10:
                self.head_rows[col] = -self.rng.uniform(0, height * 0.5)
                self.velocities[col] = self.rng.uniform(0.28, 0.65)

            head_row = int(self.head_rows[col])
            tail_len = max(8, min(height, int(9 + column_profile[col] * 24)))
            start_row = max(0, min(height - 1, head_row))

            for offset in range(tail_len):
                row = start_row - offset
                if row >= height:
                    continue
                if row < 0:
                    break

                pixel_value = float(normalized[row, col])
                depth_fade = 1.0 if offset == 0 else 0.82 ** offset
                strength = (0.25 + pixel_value * 0.7) * (0.55 + depth_fade * 0.45)

                if offset == 0:
                    glyph = self.rng.choice(MATRIX_CHAR_POOL)
                    self.trail_chars[row, col] = glyph
                    if pixel_value > self.head_threshold:
                        overrides[row, col] = 15
                        strength = max(strength, 0.78 + pixel_value * 0.3)
                    else:
                        strength = max(strength, 0.38 + pixel_value * 0.4)
                else:
                    glyph = self.trail_chars[row, col]
                    if glyph == " " or self.rng.random() < 0.1:
                        glyph = self.rng.choice(MATRIX_CHAR_POOL)
                        self.trail_chars[row, col] = glyph

                if strength > self.rain[row, col]:
                    self.rain[row, col] = strength

        active_mask = self.rain > self.activation_floor
        if np.any(active_mask):
            display_chars[active_mask] = self.trail_chars[active_mask]

        combined_intensity = np.clip(
            contrast * 0.25 + self.rain * 0.85 + self.smoothed * 0.2,
            0.0,
            1.0,
        )
        return display_chars, combined_intensity, overrides


class MatrixVoidModeState(MatrixModeState):
    """Matrix rain variant that leaves gaps for darker regions with hysteresis."""

    def __init__(self) -> None:
        super().__init__()
        self.activity = np.zeros((0, 0), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.activity = np.zeros((0, 0), dtype=np.float32)

    def apply(
        self, base_chars: np.ndarray, intensity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        display_chars, combined_intensity, overrides = super().apply(base_chars, intensity)

        if self.activity.shape != self.rain.shape:
            self.activity = np.zeros_like(self.rain, dtype=np.float32)

        active_map = (self.rain > self.activation_floor).astype(np.float32)
        self.activity = np.maximum(active_map, self.activity * 0.88)
        void_threshold = max(0.18, self.activation_floor * 1.5)
        void_mask = self.activity < void_threshold
        if np.any(void_mask):
            display_chars = display_chars.copy()
            combined_intensity = combined_intensity.copy()
            overrides = overrides.copy() if overrides is not None else None
            display_chars[void_mask] = " "
            combined_intensity[void_mask] *= 0.35
            if overrides is not None:
                overrides[void_mask] = -1
        return display_chars, combined_intensity, overrides


MATRIX_MODE_NAMES = {"matrix", "matrix-void"}
MATRIX_STATE_FACTORIES = {
    "matrix": MatrixModeState,
    "matrix-void": MatrixVoidModeState,
}


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
    parser.add_argument(
        "--mode",
        choices=MODE_ORDER,
        default="classic",
        help="Initial display mode (classic grayscale or matrix variants).",
    )
    parser.add_argument(
        "--theme",
        choices=["auto", *THEME_ROTATION],
        default="auto",
        help="Color theme for ASCII output (auto selects based on the mode).",
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
) -> tuple[np.ndarray, np.ndarray]:
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
    return ascii_frame, normalized


def render_ascii(
    chars: np.ndarray,
    intensity: np.ndarray,
    theme: str,
    overrides: Optional[np.ndarray] = None,
) -> str:
    palette = COLOR_THEMES.get(theme) or ()
    if not palette:
        return "\n".join("".join(row.tolist()) for row in chars)

    palette_array = np.array(palette, dtype=np.int16)
    bins = np.clip(
        (intensity * (len(palette_array) - 1)).astype(np.int32),
        0,
        len(palette_array) - 1,
    )
    if overrides is None:
        overrides = np.full_like(bins, -1, dtype=np.int16)

    lines: list[str] = []
    for char_row, bin_row, override_row in zip(chars, bins, overrides):
        parts: list[str] = []
        active_code: Optional[int] = None
        for ch, bin_idx, override in zip(char_row.tolist(), bin_row.tolist(), override_row.tolist()):
            code = int(override) if override >= 0 else int(palette_array[bin_idx])
            if ch == " ":
                if active_code is not None:
                    parts.append("\x1b[0m")
                    active_code = None
                parts.append(" ")
                continue
            if active_code != code:
                if active_code is not None:
                    parts.append("\x1b[0m")
                parts.append(f"\x1b[38;5;{code}m")
                active_code = code
            parts.append(ch)
        if active_code is not None:
            parts.append("\x1b[0m")
        lines.append("".join(parts))
    return "\n".join(lines)


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


def default_theme_for_mode(mode: str) -> str:
    return "matrix" if mode in MATRIX_MODE_NAMES else "none"


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

    controls = ControlPoller()
    matrix_states = {name: factory() for name, factory in MATRIX_STATE_FACTORIES.items()}

    theme_locked = args.theme != "auto"
    current_mode = args.mode
    current_theme = args.theme if theme_locked else default_theme_for_mode(current_mode)

    if controls.mode == "none":
        print("Controls: Ctrl+C to quit (runtime mode switching unavailable on this terminal).", file=sys.stderr)
    else:
        print(
            "Controls: q quit | m mode | c cycle colors | g green | a amber | o orange | n none | r reset",
            file=sys.stderr,
        )
        print(
            "Matrix tuning: [ ] head threshold | , . decay | ; ' activation floor",
            file=sys.stderr,
        )

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
            ascii_chars, intensity = frame_to_ascii(gray, width, charset, args.invert)

            overrides = None
            display_chars = ascii_chars
            display_intensity = intensity
            state = matrix_states.get(current_mode)
            if state is not None:
                display_chars, display_intensity, overrides = state.apply(ascii_chars, intensity)

            active_theme = current_theme
            if not theme_locked and args.theme == "auto":
                active_theme = default_theme_for_mode(current_mode)

            ascii_frame = render_ascii(display_chars, display_intensity, active_theme, overrides)

            move_cursor_home()
            sys.stdout.write(ascii_frame)
            sys.stdout.write("\x1b[0m")
            sys.stdout.write("\x1b[J")  # Clear anything below the current frame
            sys.stdout.flush()

            events = controls.poll()
            for event in events:
                if event == "quit":
                    return 0
                if event == "mode_next":
                    index = (MODE_ORDER.index(current_mode) + 1) % len(MODE_ORDER)
                    previous_mode = current_mode
                    current_mode = MODE_ORDER[index]
                    if current_mode in matrix_states and previous_mode != current_mode:
                        matrix_states[current_mode].reset()
                    if not theme_locked and args.theme == "auto":
                        current_theme = default_theme_for_mode(current_mode)
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
                    current_theme = default_theme_for_mode(current_mode)
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
                        value = state.adjust_activation_floor(0.01)
                        print(f"Matrix activation floor: {value:.2f}", file=sys.stderr)
                elif event == "matrix_floor_down":
                    state = matrix_states.get(current_mode)
                    if state is not None:
                        value = state.adjust_activation_floor(-0.01)
                        print(f"Matrix activation floor: {value:.2f}", file=sys.stderr)
    except KeyboardInterrupt:
        return 0
    finally:
        controls.close()
        show_cursor()
        move_cursor_home()
        capture.release()
        sys.stdout.write("\x1b[0m\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
