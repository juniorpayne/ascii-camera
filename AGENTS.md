# Repository Guidelines

## Project Structure & Module Organization
Keep the root tidy: `main.py` streams ASCII directly to the terminal while `virtual_cam.py` publishes frames through `pyvirtualcam`. Shared helpers such as `frame_to_ascii` live in `main.py`; reuse them rather than duplicating logic. Dependency pins stay in `requirements.txt`. Place any future automated tests in a top-level `tests/` package and keep generated assets or recordings out of version control.

## Build, Test, and Development Commands
Create an isolated environment before hacking:
```
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```
Run `python main.py --width 120 --fps 15` for the terminal renderer and `python virtual_cam.py --columns 120 --fps 20` to exercise the virtual camera path. When adjusting image processing logic, capture a short session and verify latency and contrast manually.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive snake_case function names (`frame_to_ascii`, `resolve_width`). Maintain type hints and docstrings similar to the existing modules. Prefer f-strings for logging, keep imports standard-library/third-party/local grouped, and avoid hard-coding platform-specific paths outside the CLI argument layer.

## Testing Guidelines
There is no automated suite yet; add `pytest`-based coverage when contributing substantial logic. Name test files `test_<feature>.py` under `tests/` and focus on pure functions such as width resolution and ASCII conversion. For hardware-dependent flows, include reproducible manual steps (device index, resolution, frame rate) in the pull request description.

## Commit & Pull Request Guidelines
Write commits in present-tense imperatives (e.g., "Add HSV prefilter") and keep the subject under ~72 characters; expand on reasoning in the body when behavior changes. Pull requests should summarize user-facing effects, list tested commands, mention any new dependencies, and attach screenshots or GIFs when modifying rendering parameters. Reference related issues so changelog entries are easy to generate.