"""ASCII camera streamer.

Capture frames from a webcam and render them as ASCII art directly in the terminal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import select
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # type: ignore

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
MODE_ORDER = ["classic", "matrix", "matrix-void", "matrix-show"]
THEME_ROTATION = ["none", "green", "amber", "orange"]
COLOR_THEMES = {
    "none": (),
    "green": (22, 28, 34, 40, 46, 82),
    "amber": (94, 130, 136, 142, 214, 220),
    "orange": (94, 130, 166, 202, 208, 214),
    "matrix": (22, 28, 34, 40, 46, 82, 118),
}


@dataclass
class AsciiFrameData:
    chars: np.ndarray
    intensity: np.ndarray
    intensity_tensor: Optional[Any] = None
    indices_tensor: Optional[Any] = None


TORCH_RENDER_DEVICE: Optional[Any] = None
_TORCH_RENDERERS: Dict[Tuple[int, int, int, int, int, str], Any] = {}


class FramePreprocessor:
    """Resize and convert frames to grayscale for ASCII rendering."""

    def preprocess(self, frame: np.ndarray, width: int) -> Any:
        raise NotImplementedError


class CPUFramePreprocessor(FramePreprocessor):
    def preprocess(self, frame: np.ndarray, width: int) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, original_width = gray.shape
        target_height, interpolation = compute_resize_params(height, original_width, width)
        if width == original_width and target_height == height:
            return gray
        return cv2.resize(gray, (width, target_height), interpolation=interpolation)


class CUDAFramePreprocessor(FramePreprocessor):
    def __init__(self, device_index: Optional[int] = None) -> None:
        if not hasattr(cv2, "cuda"):
            raise RuntimeError("OpenCV was built without CUDA support.")
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count <= 0:
            build_has_cuda = False
            build_info = getattr(cv2, "getBuildInformation", None)
            if callable(build_info):
                info = build_info()
                build_has_cuda = "NVIDIA CUDA: YES" in info or "CUDA: YES" in info
            if not build_has_cuda:
                raise RuntimeError("OpenCV build lacks CUDA support. Install a CUDA-enabled OpenCV package or choose --accelerator torch.")
            raise RuntimeError("No CUDA-enabled devices detected. Ensure NVIDIA drivers are installed and the GPU is not in use.")
        if device_index is not None:
            if device_index < 0 or device_index >= device_count:
                raise RuntimeError(
                    f"Requested CUDA device {device_index} is out of range (0-{device_count - 1})."
                )
            cv2.cuda.setDevice(device_index)
        self.device_index = cv2.cuda.getDevice()
        self.stream = cv2.cuda.Stream()
        self.gpu_src = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()
        self.gpu_resized = cv2.cuda_GpuMat()

    def preprocess(self, frame: np.ndarray, width: int) -> np.ndarray:
        original_height, original_width = frame.shape[:2]
        target_height, interpolation = compute_resize_params(original_height, original_width, width)
        self.gpu_src.upload(frame, stream=self.stream)
        cv2.cuda.cvtColor(self.gpu_src, cv2.COLOR_BGR2GRAY, dst=self.gpu_gray, stream=self.stream)
        if width == original_width and target_height == original_height:
            result_gpu = self.gpu_gray
        else:
            cv2.cuda.resize(
                self.gpu_gray,
                (width, target_height),
                dst=self.gpu_resized,
                interpolation=interpolation,
                stream=self.stream,
            )
            result_gpu = self.gpu_resized
        result = result_gpu.download(stream=self.stream)
        self.stream.waitForCompletion()
        return result






class TorchFramePreprocessor(FramePreprocessor):
    def __init__(self, device_spec: Optional[str] = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is not installed.") from exc

        self.torch = torch
        self.directml_device = None

        if device_spec:
            spec = device_spec.strip().lower()
            if spec.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Requested CUDA device but PyTorch was not compiled with CUDA support. "
                        "Install a CUDA-enabled PyTorch build from https://pytorch.org/."
                    )
                self.device = torch.device(device_spec)
            elif spec.startswith("mps"):
                if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                    raise RuntimeError("Requested MPS device but the backend is unavailable on this system.")
                self.device = torch.device("mps")
            elif spec.startswith("dml"):
                self.directml_device = self._init_directml(device_spec)
                if self.directml_device is None:
                    raise RuntimeError(
                        "Requested DirectML device but torch-directml is not installed. "
                        "Install it with `pip install torch-directml`."
                    )
                self.device = self.directml_device
            else:
                self.device = torch.device(device_spec)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.directml_device = self._init_directml(None)
                if self.directml_device is None:
                    raise RuntimeError(
                        "No GPU-capable torch backend available. Install a CUDA-enabled PyTorch build or torch-directml."
                    )
                self.device = self.directml_device

        global TORCH_RENDER_DEVICE
        TORCH_RENDER_DEVICE = self.device

        self.device_label = self._describe_device()

        self.gray_weights = torch.tensor([0.114, 0.587, 0.299], device=self.device, dtype=torch.float32)

    def _init_directml(self, spec: Optional[str]):
        try:
            import torch_directml  # type: ignore
        except ImportError:
            return None
        if spec and ":" in spec:
            _, index = spec.split(":", 1)
            return torch_directml.device(int(index))
        return torch_directml.device()

    def _describe_device(self) -> str:
        torch = self.torch
        device = self.device
        device_type = getattr(device, "type", None)
        if device_type == "cuda":
            if getattr(device, "index", None) is not None:
                index = device.index
            else:
                try:
                    index = torch.cuda.current_device()
                except Exception:
                    index = 0
            try:
                name = torch.cuda.get_device_name(index)
                return f"cuda:{index} ({name})"
            except Exception:
                return f"cuda:{index}"
        if device_type == "mps":
            return "mps"
        if self.directml_device is not None and device_type == "dml":
            index = getattr(device, "index", 0) or 0
            return f"dml:{index}"
        return str(device)

    def preprocess(self, frame: np.ndarray, width: int) -> Any:
        torch = self.torch
        original_height, original_width = frame.shape[:2]
        target_height, _ = compute_resize_params(original_height, original_width, width)
        with torch.no_grad():
            frame_tensor = torch.as_tensor(frame, device=self.device, dtype=torch.float32)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            gray = torch.tensordot(frame_tensor, self.gray_weights, dims=([1], [0]))
            gray = gray.unsqueeze(1)
            mode = "area" if (target_height < original_height or width < original_width) else "bilinear"
            interp_kwargs = {"align_corners": False} if mode != "area" else {}
            resized = torch.nn.functional.interpolate(
                gray,
                size=(target_height, width),
                mode=mode,
                **interp_kwargs,
            )
            normalized = resized.squeeze(0).squeeze(0).clamp(0.0, 1.0).contiguous()
        return normalized

def compute_resize_params(original_height: int, original_width: int, target_width: int) -> tuple[int, int]:
    if original_width <= 0:
        original_width = 1
    target_height = max(1, int((original_height / original_width) * target_width * CHAR_ASPECT))
    interpolation = cv2.INTER_AREA if target_width < original_width else cv2.INTER_LINEAR
    return target_height, interpolation




def create_preprocessor(
    mode: str,
    cuda_device: Optional[int],
    torch_device: Optional[str],
) -> FramePreprocessor:
    if mode == "cpu":
        return CPUFramePreprocessor()
    if mode == "cuda":
        try:
            preprocessor = CUDAFramePreprocessor(cuda_device)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Unable to initialize the OpenCV CUDA accelerator. Install a CUDA-enabled OpenCV build or use --accelerator torch. (Reason: {exc})"
            ) from exc
        device_idx = getattr(preprocessor, "device_index", cuda_device if cuda_device is not None else 0)
        print(f"Using OpenCV CUDA accelerator on device {device_idx}.", file=sys.stderr)
        return preprocessor
    if mode == "torch":
        preprocessor = TorchFramePreprocessor(torch_device)
        print(f"Using torch accelerator on {preprocessor.device_label}.", file=sys.stderr)
        return preprocessor
    if mode == "auto":
        try:
            preprocessor = CUDAFramePreprocessor(cuda_device)
            device_idx = getattr(preprocessor, "device_index", cuda_device if cuda_device is not None else 0)
            print(f"Using OpenCV CUDA accelerator on device {device_idx}.", file=sys.stderr)
            return preprocessor
        except RuntimeError as exc:
            print(f"OpenCV CUDA accelerator unavailable ({exc}). Trying torch backend...", file=sys.stderr)
        try:
            preprocessor = TorchFramePreprocessor(torch_device)
            print(f"Using torch accelerator on {preprocessor.device_label}.", file=sys.stderr)
            return preprocessor
        except RuntimeError as exc2:
            print(f"Torch acceleration unavailable ({exc2}). Falling back to CPU.", file=sys.stderr)
        return CPUFramePreprocessor()
    raise ValueError(f"Unknown accelerator mode: {mode}")


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
        elif ch in ("+", "="):
            events.append("matrix_contrast_up")
        elif ch in ("-", "_"):
            events.append("matrix_contrast_down")
        return events



class MatrixModeState:
    """Matrix renderer that overlays animated rain on a static glyph image."""

    def __init__(self) -> None:
        self.rain = np.zeros((0, 0), dtype=np.float32)
        self.rain_tensor: Any = None
        self.trail_chars = np.full((0, 0), " ", dtype="<U1")
        self.trail_indices_tensor: Any = None
        self.head_rows = np.zeros(0, dtype=np.float32)
        self.head_rows_tensor: Any = None
        self.velocities = np.zeros(0, dtype=np.float32)
        self.velocities_tensor: Any = None
        self.rng = np.random.default_rng()
        self.head_threshold = 0.28
        self.decay = 0.86
        self.activation_floor = 0.05
        self.smoothed = np.zeros((0, 0), dtype=np.float32)
        self.smoothed_tensor: Any = None
        self.contrast_gain = 1.35

    def reset(self) -> None:
        self.rain = np.zeros((0, 0), dtype=np.float32)
        self.rain_tensor = None
        self.trail_chars = np.full((0, 0), " ", dtype="<U1")
        self.trail_indices_tensor = None
        self.head_rows = np.zeros(0, dtype=np.float32)
        self.head_rows_tensor = None
        self.velocities = np.zeros(0, dtype=np.float32)
        self.velocities_tensor = None
        self.smoothed = np.zeros((0, 0), dtype=np.float32)
        self.smoothed_tensor = None

    def adjust_head_threshold(self, delta: float) -> float:
        self.head_threshold = float(np.clip(self.head_threshold + delta, 0.05, 0.9))
        return self.head_threshold

    def adjust_decay(self, delta: float) -> float:
        self.decay = float(np.clip(self.decay + delta, 0.2, 0.98))
        return self.decay

    def adjust_activation_floor(self, delta: float) -> float:
        self.activation_floor = float(np.clip(self.activation_floor + delta, 0.0, 0.5))
        return self.activation_floor

    def adjust_contrast_gain(self, delta: float) -> float:
        self.contrast_gain = float(np.clip(self.contrast_gain + delta, 0.2, 5.0))
        return self.contrast_gain

    def _ensure_torch_buffers(self, torch_mod: Any, device: Any, height: int, width: int) -> None:
        if (
            self.rain_tensor is None
            or self.rain_tensor.device != device
            or self.rain_tensor.shape != (height, width)
        ):
            self.rain_tensor = torch_mod.zeros((height, width), device=device, dtype=torch_mod.float32)
            self.trail_indices_tensor = torch_mod.full(
                (height, width), -1, device=device, dtype=torch_mod.int64
            )
        if (
            self.head_rows_tensor is None
            or self.head_rows_tensor.device != device
            or self.head_rows_tensor.shape[0] != width
        ):
            self.head_rows_tensor = -torch_mod.rand(width, device=device, dtype=torch_mod.float32) * height
            self.velocities_tensor = (
                torch_mod.rand(width, device=device, dtype=torch_mod.float32) * (0.65 - 0.28) + 0.28
            )
        if (
            self.smoothed_tensor is None
            or self.smoothed_tensor.device != device
            or self.smoothed_tensor.shape != (height, width)
        ):
            self.smoothed_tensor = torch_mod.zeros((height, width), device=device, dtype=torch_mod.float32)

    def apply(
        self, ascii_frame: AsciiFrameData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        intensity_np = np.clip(ascii_frame.intensity, 0.0, 1.0).astype(np.float32, copy=False)
        intensity_tensor = ascii_frame.intensity_tensor

        if intensity_tensor is not None:
            try:
                import torch as torch_mod  # type: ignore

                return self._apply_torch(ascii_frame, intensity_np, intensity_tensor, torch_mod)
            except Exception:
                self.rain_tensor = None
                self.trail_indices_tensor = None
                self.head_rows_tensor = None
                self.velocities_tensor = None
                self.smoothed_tensor = None

        return self._apply_cpu(intensity_np)

    def _apply_cpu(self, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = intensity.shape
        normalized = np.clip(intensity, 0.0, 1.0).astype(np.float32)
        if self.smoothed.shape != normalized.shape:
            self.smoothed = normalized.copy()
        else:
            self.smoothed = np.clip(self.smoothed * 0.82 + normalized * 0.18, 0.0, 1.0)
        contrast_source = np.clip(self.smoothed * self.contrast_gain, 0.0, 1.0)
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
        return (
            display_chars,
            combined_intensity.astype(np.float32, copy=False),
            overrides.astype(np.int16, copy=False),
        )

    def _apply_torch(
        self,
        ascii_frame: AsciiFrameData,
        intensity_np: np.ndarray,
        intensity_tensor: Any,
        torch_mod: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        device = intensity_tensor.device
        height, width = intensity_tensor.shape
        self._ensure_torch_buffers(torch_mod, device, height, width)

        normalized = intensity_tensor.clamp(0.0, 1.0)
        if self.smoothed_tensor is None or self.smoothed_tensor.shape != normalized.shape:
            self.smoothed_tensor = normalized.detach().clone()
        else:
            self.smoothed_tensor = (
                self.smoothed_tensor * 0.82 + normalized * 0.18
            ).clamp_(0.0, 1.0)
        torch_smoothed = self.smoothed_tensor
        self.smoothed = torch_smoothed.detach().cpu().numpy().astype(np.float32, copy=False)

        contrast_tensor = torch_smoothed.mul(self.contrast_gain).clamp_(0.0, 1.0).pow_(0.7)
        self.rain_tensor.mul_(self.decay)

        fade_mask = self.rain_tensor < (self.activation_floor * 0.9)
        if torch_mod.any(fade_mask):
            self.trail_indices_tensor = self.trail_indices_tensor.masked_fill(fade_mask, -1)

        column_profile = 0.15 + normalized.mean(dim=0) * 0.85
        speeds = self.velocities_tensor + column_profile * 0.55 + 0.35
        self.head_rows_tensor = torch_mod.maximum(
            self.head_rows_tensor + speeds,
            torch_mod.full_like(self.head_rows_tensor, -2.0),
        )
        reset_mask = self.head_rows_tensor >= (height + 10)
        if torch_mod.any(reset_mask):
            self.head_rows_tensor[reset_mask] = -torch_mod.rand_like(self.head_rows_tensor[reset_mask]) * (height * 0.5)
            self.velocities_tensor[reset_mask] = (
                torch_mod.rand_like(self.velocities_tensor[reset_mask]) * (0.65 - 0.28) + 0.28
            )

        heads = torch_mod.floor(self.head_rows_tensor).to(torch_mod.int64)
        tail_lengths = torch_mod.clamp((9 + column_profile * 24).to(torch_mod.int64), 8, height)
        overrides_tensor = torch_mod.full((height, width), -1, dtype=torch_mod.int16, device=device)
        len_pool = len(MATRIX_CHAR_POOL)
        max_tail = int(tail_lengths.max().item()) if tail_lengths.numel() else 0

        for offset in range(max_tail):
            valid = tail_lengths > offset
            if not torch_mod.any(valid):
                break
            rows = heads - offset
            mask = valid & (rows >= 0) & (rows < height)
            if not torch_mod.any(mask):
                continue
            cols = torch_mod.nonzero(mask, as_tuple=False).squeeze(1)
            rows_sel = rows[mask]
            pixel_value = normalized[rows_sel, cols]
            depth_fade = 1.0 if offset == 0 else 0.82 ** offset
            strength = (0.25 + pixel_value * 0.7) * (0.55 + depth_fade * 0.45)

            if offset == 0:
                glyph_choices = torch_mod.randint(len_pool, (cols.shape[0],), device=device)
                self.trail_indices_tensor[rows_sel, cols] = glyph_choices
                bright_mask = pixel_value > self.head_threshold
                if torch_mod.any(bright_mask):
                    bright_idx = torch_mod.nonzero(bright_mask, as_tuple=False).squeeze(1)
                    overrides_tensor[rows_sel[bright_idx], cols[bright_idx]] = 15
                    strength[bright_mask] = torch_mod.maximum(
                        strength[bright_mask], 0.78 + pixel_value[bright_mask] * 0.3
                    )
                dim_mask = ~bright_mask
                if torch_mod.any(dim_mask):
                    strength[dim_mask] = torch_mod.maximum(
                        strength[dim_mask], 0.38 + pixel_value[dim_mask] * 0.4
                    )
            else:
                existing = self.trail_indices_tensor[rows_sel, cols]
                regen_mask = (existing < 0) | (torch_mod.rand_like(pixel_value) < 0.1)
                if torch_mod.any(regen_mask):
                    regen_idx = torch_mod.nonzero(regen_mask, as_tuple=False).squeeze(1)
                    new_vals = torch_mod.randint(len_pool, (regen_idx.shape[0],), device=device)
                    self.trail_indices_tensor[rows_sel[regen_idx], cols[regen_idx]] = new_vals

            current = self.rain_tensor[rows_sel, cols]
            self.rain_tensor[rows_sel, cols] = torch_mod.maximum(current, strength)

        active_mask = self.rain_tensor > self.activation_floor
        base_indices = torch_mod.clamp(
            (contrast_tensor * (len_pool - 1) + 0.5).to(torch_mod.int64),
            0,
            len_pool - 1,
        )
        display_indices = base_indices.clone()
        trail_valid = (self.trail_indices_tensor >= 0) & active_mask
        if torch_mod.any(trail_valid):
            display_indices[trail_valid] = self.trail_indices_tensor[trail_valid]

        display_indices_np = display_indices.detach().cpu().numpy()
        display_chars = MATRIX_CHAR_POOL[display_indices_np]

        self.rain = self.rain_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        combined_tensor = torch_mod.clamp(
            contrast_tensor * 0.25 + self.rain_tensor * 0.85 + self.smoothed_tensor * 0.2,
            0.0,
            1.0,
        )
        combined_np = combined_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

        overrides_np = overrides_tensor.detach().cpu().numpy().astype(np.int16, copy=False)
        trail_np = np.full(display_chars.shape, " ", dtype="<U1")
        trail_indices_np = self.trail_indices_tensor.detach().cpu().numpy()
        valid_mask = trail_indices_np >= 0
        if np.any(valid_mask):
            trail_np[valid_mask] = MATRIX_CHAR_POOL[trail_indices_np[valid_mask]]
        self.trail_chars = trail_np
        self.head_rows = self.head_rows_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        self.velocities = self.velocities_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

        return display_chars, combined_np, overrides_np


class MatrixVoidModeState(MatrixModeState):
    """Matrix rain variant that leaves gaps for darker regions with hysteresis."""

    def __init__(self) -> None:
        super().__init__()
        self.contrast_gain = 1.6
        self.activity = np.zeros((0, 0), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.activity = np.zeros((0, 0), dtype=np.float32)

    def apply(
        self, ascii_frame: AsciiFrameData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        display_chars, combined_intensity, overrides = super().apply(ascii_frame)

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
            combined_intensity[void_mask] *= 0.55
            if overrides is not None:
                overrides[void_mask] = -1
        return display_chars, combined_intensity, overrides



class MatrixShowModeState(MatrixModeState):
    '''Matrix rain with background ASCII details for richer silhouettes.'''

    def __init__(self) -> None:
        super().__init__()
        self.detail_persistence = 0.92
        self.detail_threshold = 0.38
        self.activity = np.zeros((0, 0), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.activity = np.zeros((0, 0), dtype=np.float32)

    def apply(
        self, ascii_frame: AsciiFrameData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_chars, base_intensity, overrides = super().apply(ascii_frame)

        detail_intensity = np.clip(ascii_frame.intensity, 0.0, 1.0).astype(np.float32, copy=False)
        if self.activity.shape != self.rain.shape:
            self.activity = np.zeros_like(self.rain, dtype=np.float32)
        active_map = (self.rain > self.activation_floor).astype(np.float32)
        self.activity = np.maximum(active_map, self.activity * self.detail_persistence)

        detail_mask = self.activity < self.detail_threshold
        boosted = np.clip(base_intensity * 1.08, 0.0, 1.0)
        if not np.any(detail_mask):
            return base_chars, boosted, overrides

        grad_y, grad_x = np.gradient(detail_intensity)
        edge_strength = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)
        detail_mix = np.clip(detail_intensity * 0.85 + edge_strength * 1.2, 0.0, 1.0)
        detail_mix = np.clip(detail_mix * 0.8 + boosted * 0.2 + 0.05, 0.0, 1.0)

        display_chars = base_chars.copy()
        detail_indices = np.clip(
            np.rint(detail_mix * (len(MATRIX_CHAR_POOL) - 1)).astype(np.int32),
            0,
            len(MATRIX_CHAR_POOL) - 1,
        )
        display_chars[detail_mask] = MATRIX_CHAR_POOL[detail_indices[detail_mask]]

        boosted = np.clip(boosted * 1.12, 0.0, 1.0)
        boosted[detail_mask] = np.clip(detail_mix[detail_mask] * 1.1, 0.0, 1.0)

        if overrides is None:
            overrides = np.full(display_chars.shape, -1, dtype=np.int16)
        else:
            overrides = overrides.copy()
        overrides[detail_mask] = 82

        return display_chars, boosted, overrides


MATRIX_MODE_NAMES = {"matrix", "matrix-void", "matrix-show"}
MATRIX_STATE_FACTORIES = {
    "matrix": MatrixModeState,
    "matrix-void": MatrixVoidModeState,
    "matrix-show": MatrixShowModeState,
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
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "torch"],
        default="auto",
        help="Preprocessing backend (auto prefers OpenCV CUDA, then torch GPU/DirectML, otherwise CPU).",
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
    return parser.parse_args()


def resolve_width(
    frame_shape: tuple[int, ...], target_width: Optional[int]
) -> int:
    term_size = shutil.get_terminal_size((80, 24))
    term_width = term_size.columns

    if target_width is None:
        width = term_width if term_width > 0 else 80
    else:
        width = target_width

    if term_width > 0:
        width = min(width, term_width)
    width = max(2, int(width))

    if len(frame_shape) >= 2:
        original_height = int(frame_shape[0])
        original_width = int(frame_shape[1])
    else:
        original_height = 1
        original_width = 1

    if term_size.lines > 0 and original_width > 0:
        max_rows = max(2, term_size.lines - 1)
        aspect = (original_height / original_width) * CHAR_ASPECT
        if aspect > 0:
            max_width_from_rows = int(max_rows / aspect)
            if max_width_from_rows >= 2:
                width = min(width, max_width_from_rows)
            else:
                width = 2
    return max(2, width)


def frame_to_ascii(
    gray_frame: Any, charset: np.ndarray, invert: bool
) -> AsciiFrameData:
    try:
        import torch as torch_mod  # type: ignore
    except ImportError:  # pragma: no cover - CPU-only environments
        torch_mod = None
    else:
        if isinstance(gray_frame, torch_mod.Tensor):
            working = 1.0 - gray_frame if invert else gray_frame
            normalized = working.clamp(0.0, 1.0)
            num_chars = len(charset)
            indices = torch_mod.clamp(
                (normalized * (num_chars - 1) + 0.5).to(torch_mod.int64),
                0,
                num_chars - 1,
            )
            ascii_frame = charset[indices.detach().cpu().numpy()]
            intensity_np = normalized.detach().cpu().numpy().astype(np.float32, copy=False)
            return AsciiFrameData(
                ascii_frame,
                intensity_np,
                intensity_tensor=normalized,
                indices_tensor=indices,
            )

    if invert:
        if np.issubdtype(gray_frame.dtype, np.floating):
            working = 1.0 - gray_frame
        else:
            working = 255 - gray_frame
    else:
        working = gray_frame

    if np.issubdtype(working.dtype, np.floating):
        max_val = float(np.nanmax(working)) if working.size else 0.0
        min_val = float(np.nanmin(working)) if working.size else 0.0
        if max_val > 1.0 or min_val < 0.0:
            base = np.clip(working, 0.0, 255.0).astype(np.float32) / 255.0
        else:
            base = np.clip(working, 0.0, 1.0).astype(np.float32)
    else:
        base = working.astype(np.float32) / 255.0

    normalized_np = np.clip(base, 0.0, 1.0)
    indices_np = np.clip(
        np.rint(normalized_np * (len(charset) - 1)).astype(np.int32),
        0,
        len(charset) - 1,
    )

    ascii_frame = charset[indices_np]
    return AsciiFrameData(ascii_frame, normalized_np)


def _text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return font.getsize(text)


def _compute_font_metrics(
    font: ImageFont.ImageFont, line_spacing: Optional[int]
) -> Tuple[int, int, int, int, int]:
    char_width, char_height = _text_size(font, "M")
    char_width = max(1, char_width)
    char_height = max(1, char_height)
    spacing = line_spacing if line_spacing is not None else max(2, char_height // 6)
    margin_x = max(4, char_width // 2)
    margin_y = max(4, spacing)
    return char_width, char_height, spacing, margin_x, margin_y


class TorchGlyphRenderer:
    def __init__(self, font: ImageFont.ImageFont, metrics: Tuple[int, int, int, int, int], device: Any) -> None:
        import torch

        self.torch = torch
        self.font = font
        self.char_width, self.char_height, self.spacing, self.margin_x, self.margin_y = metrics
        self.device = torch.device(device)
        self.char_to_index: Dict[str, int] = {}
        self.space_index = 0
        self.atlas: Optional[torch.Tensor] = None
        self._ensure_glyph(" ")

    def _ensure_glyph(self, ch: str) -> None:
        if ch in self.char_to_index:
            return
        glyph_mask = self._render_glyph_mask(ch)
        glyph_tensor = self.torch.tensor(glyph_mask, device=self.device, dtype=self.torch.float32)
        glyph_tensor = glyph_tensor.unsqueeze(0)
        if self.atlas is None:
            self.atlas = glyph_tensor
        else:
            self.atlas = self.torch.cat([self.atlas, glyph_tensor], dim=0)
        index = int(self.atlas.shape[0] - 1)
        self.char_to_index[ch] = index
        if ch == " ":
            self.space_index = index

    def _render_glyph_mask(self, ch: str) -> np.ndarray:
        image = Image.new("L", (self.char_width, self.char_height), 0)
        draw = ImageDraw.Draw(image)
        offset_x = offset_y = 0
        if hasattr(self.font, "getoffset"):
            try:
                offset_x, offset_y = self.font.getoffset(ch)
            except Exception:
                offset_x = offset_y = 0
        draw.text((-offset_x, -offset_y), ch, font=self.font, fill=255)
        return np.array(image, dtype=np.float32) / 255.0

    def ensure_glyphs(self, chars: Iterable[str]) -> None:
        for ch in chars:
            self._ensure_glyph(str(ch))

    def map_chars(self, chars: np.ndarray) -> Any:
        flat = chars.reshape(-1)
        indices: list[int] = []
        for ch in flat:
            ch_str = str(ch)
            idx = self.char_to_index.get(ch_str)
            if idx is None:
                self._ensure_glyph(ch_str)
                idx = self.char_to_index[ch_str]
            indices.append(idx)
        return self.torch.tensor(indices, device=self.device, dtype=self.torch.long).view(chars.shape)

    def render(self, chars: np.ndarray, color_tensor: Any) -> Any:
        torch = self.torch
        rows, cols = chars.shape
        if rows == 0 or cols == 0:
            return torch.zeros((self.margin_y * 2 + 1, self.margin_x * 2 + 1, 3), device=self.device)
        indices = self.map_chars(chars)
        glyph_masks = self.atlas[indices]
        tiles = glyph_masks.unsqueeze(-1) * color_tensor[:, :, None, None, :]
        tiles = tiles.permute(0, 2, 1, 3, 4).contiguous()
        tile_rows = tiles.view(rows, self.char_height, cols * self.char_width, 3)
        segments: list[Any] = []
        gap = None
        if self.spacing > 0:
            gap = torch.zeros((self.spacing, tile_rows.shape[2], 3), device=self.device)
        for row_idx in range(rows):
            segments.append(tile_rows[row_idx])
            if gap is not None and row_idx < rows - 1:
                segments.append(gap)
        body = torch.cat(segments, dim=0) if segments else torch.zeros((self.char_height, self.char_width, 3), device=self.device)
        canvas_height = body.shape[0] + self.margin_y * 2
        canvas_width = body.shape[1] + self.margin_x * 2
        canvas = torch.zeros((canvas_height, canvas_width, 3), device=self.device)
        canvas[self.margin_y : self.margin_y + body.shape[0], self.margin_x : self.margin_x + body.shape[1], :] = body
        return canvas


def _get_torch_renderer_key(
    font: ImageFont.ImageFont, metrics: Tuple[int, int, int, int, int], device: Any
) -> Tuple[Any, ...]:
    font_source = getattr(font, "path", None)
    font_size = getattr(font, "size", None)
    device_type = getattr(device, "type", str(device))
    device_index = getattr(device, "index", None)
    return (
        id(font),
        str(font_source),
        font_size,
        *metrics,
        device_type,
        -1 if device_index is None else int(device_index),
    )


def _get_torch_renderer(
    font: ImageFont.ImageFont, metrics: Tuple[int, int, int, int, int], device: Any
) -> TorchGlyphRenderer:
    key = _get_torch_renderer_key(font, metrics, device)
    renderer = _TORCH_RENDERERS.get(key)
    if renderer is None:
        renderer = TorchGlyphRenderer(font, metrics, device)
        _TORCH_RENDERERS[key] = renderer
    return renderer


def _render_char_grid_to_image_torch(
    chars: np.ndarray,
    intensity: np.ndarray,
    rgb_colors: Optional[np.ndarray],
    theme: str,
    font: ImageFont.ImageFont,
    ascii_frame: Optional[AsciiFrameData],
    metrics: Tuple[int, int, int, int, int],
) -> Optional[Image.Image]:
    try:
        import torch
    except ImportError:
        return None

    device = None
    if ascii_frame is not None and ascii_frame.intensity_tensor is not None:
        device = ascii_frame.intensity_tensor.device
    elif TORCH_RENDER_DEVICE is not None:
        device = TORCH_RENDER_DEVICE
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        return None

    device = torch.device(device)

    rows, cols = chars.shape
    if rows == 0 or cols == 0:
        return None

    try:
        renderer = _get_torch_renderer(font, metrics, device)
        unique_chars = np.unique(chars)
        renderer.ensure_glyphs(unique_chars.tolist())

        if rgb_colors is not None:
            color_tensor = torch.as_tensor(rgb_colors, dtype=torch.float32, device=device) / 255.0
        else:
            color_tensor = torch.as_tensor(intensity, dtype=torch.float32, device=device)
            color_tensor = color_tensor.clamp(0.0, 1.0).unsqueeze(-1).repeat(1, 1, 3)

        color_tensor = color_tensor.to(device)
        canvas = renderer.render(chars, color_tensor)
        canvas = canvas.clamp_(0.0, 1.0)
        image_array = (canvas * 255.0).round().to(torch.uint8).cpu().numpy()
        return Image.fromarray(image_array, mode="RGB")
    except Exception:
        return None


def _render_char_grid_to_image_cpu(
    chars: np.ndarray,
    intensity: np.ndarray,
    rgb_colors: Optional[np.ndarray],
    theme: str,
    font: ImageFont.ImageFont,
    metrics: Tuple[int, int, int, int, int],
    background: tuple[int, int, int],
) -> Image.Image:
    rows, cols = chars.shape
    if rows == 0 or cols == 0:
        return Image.new("RGB", (16, 16), background)

    char_width, char_height, spacing, margin_x, margin_y = metrics

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


def render_char_grid_to_image(
    chars: np.ndarray,
    intensity: np.ndarray,
    rgb_colors: Optional[np.ndarray],
    theme: str,
    font: ImageFont.ImageFont,
    background: tuple[int, int, int] = (0, 0, 0),
    line_spacing: Optional[int] = None,
    ascii_frame: Optional[AsciiFrameData] = None,
) -> Image.Image:
    metrics = _compute_font_metrics(font, line_spacing)
    gpu_image = _render_char_grid_to_image_torch(chars, intensity, rgb_colors, theme, font, ascii_frame, metrics)
    if gpu_image is not None:
        return gpu_image
    return _render_char_grid_to_image_cpu(chars, intensity, rgb_colors, theme, font, metrics, background)
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

    charset_raw = args.charset or ASCII_CHARS
    if len(charset_raw) < 2:
        print("The character set must contain at least two characters.", file=sys.stderr)
        return 1
    charset = np.array(list(charset_raw))

    capture = cv2.VideoCapture(args.device)
    if not capture.isOpened():
        print(f"Could not open camera device {args.device}.", file=sys.stderr)
        return 1

    try:
        preprocessor = create_preprocessor(args.accelerator, args.cuda_device, args.torch_device)
    except RuntimeError as exc:
        print(f"Failed to initialize {args.accelerator} accelerator: {exc}", file=sys.stderr)
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
            "Matrix tuning: [ ] head threshold | , . decay | ; ' activation floor | + - contrast",
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

            width = resolve_width(frame.shape, args.width)
            gray = preprocessor.preprocess(frame, width)
            ascii_frame = frame_to_ascii(gray, charset, args.invert)

            overrides = None
            display_chars = ascii_frame.chars
            display_intensity = ascii_frame.intensity
            state = matrix_states.get(current_mode)
            if state is not None:
                display_chars, display_intensity, overrides = state.apply(ascii_frame)

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
        controls.close()
        show_cursor()
        move_cursor_home()
        capture.release()
        sys.stdout.write("\x1b[0m\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
