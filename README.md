# ASCII Camera

A small Python toolkit that can stream your camera feed as ASCII art either directly in the terminal or by publishing to a virtual camera for video conferencing apps.

## Prerequisites

- Python 3.9 or newer.
- A working webcam.
- (Optional) A terminal that supports ANSI escape sequences for smoother output.
- For the virtual camera filter on Windows, install OBS Studio (v26+) so the bundled OBS Virtual Camera driver is available. On macOS/Linux ensure you meet the [pyvirtualcam](https://github.com/jremmons/pyvirtualcam) backend prerequisites.

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

## Terminal ASCII Stream

```powershell
python main.py [--device N] [--width 120] [--fps 15]
```

- `--device` selects the zero-based camera index. Use `0` for the default webcam.
- `--width` controls the number of character columns rendered. The script will automatically constrain the frame to the terminal width.
- `--fps` clamps the refresh rate if your camera outputs faster than you want.

Press `q` to quit while the script is running. On Windows consoles that do not support ANSI escape codes, the image may flicker because the screen has to be cleared between frames.

## Virtual Camera ASCII Filter

```powershell
python virtual_cam.py [--device N] [--columns 120] [--fps 20]
```

- The script opens your webcam, converts each frame to ASCII, renders it to an image, and publishes it via `pyvirtualcam`.
- Use `--columns` to control the ASCII resolution and `--font-path` if you want to point at a specific monospace `.ttf` font (Consolas is used by default on Windows).
- `--output-width` / `--output-height` let you force a specific resolution (for example 1280×720) if your meeting software expects it.
- `--mirror` toggles a mirrored view for natural self-viewing.

Once the script prints the virtual camera name, open your conferencing tool (Teams, Zoom, Meet, etc.) and choose that device as the video source. Stop the script with `Ctrl+C` when you are done.

## Notes

- For the best contrast, run the script in a dark themed terminal and maximize the window.
- If the capture feed cannot open, verify that no other application is using the camera and that privacy settings allow console applications to access it.
- The virtual camera stream depends on OS-level support: ensure the relevant driver/backend is installed before running `virtual_cam.py`.
