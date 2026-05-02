# 📸 Open Source Photo Booth

> An interactive desktop photo booth powered by Python, OpenCV, and MediaPipe — place FOSS stickers on your camera feed and capture photos using hand gestures. No mouse. No keyboard. Just your hands.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-00897B?style=flat-square&logo=google&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Hand Gestures](#hand-gestures)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Stickers](#stickers)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Open Source Photo Booth is a touchless, interactive desktop application that uses your webcam and real-time hand tracking to let you browse, place, and drag Free and Open Source Software (FOSS) stickers over your camera feed — then snap a clean photo with a peace sign gesture.

Built entirely with open source tools: Python 3.11, OpenCV, and Google's MediaPipe Hands. No proprietary SDKs, no internet connection required at runtime.

---

## Features

- **Real-time hand tracking** via MediaPipe Hands with hand skeleton overlay at 30+ FPS
- **Gesture-triggered photo capture** — hold ✌️ for ~0.7 s; a cyan arc fills around the fingertip confirming intent before the countdown starts
- **Draggable FOSS stickers** — white-tinted SVG and PNG logos you can freely drag anywhere on screen
- **Sticker selection strip** — paginated gallery pinned to the **top** of the frame, showing 4 stickers at a time
- **Dwell-based button activation** — hover your index finger over a button for ~0.4 s to navigate pages; accidental flicks are ignored
- **Visual countdown** — prominent 3–2–1 overlay before capture
- **Clean captures** — saved photos include the camera frame and stickers, but never the UI controls
- **Kokoa logo watermark** — brand logo stamped in the bottom-right corner on both the live view and saved photos
- **Auto-clear stickers** — placed stickers are wiped after each capture so the next shot starts fresh
- **White flash confirmation** — a fade-out flash confirms every capture
- **SVG sticker support** — stickers can be `.svg` or `.png`; SVGs are rasterised in memory via cairosvg
- **Fully configurable** — tweak FPS, dwell frames, sticker sizes, and logo placement via `config.py`

---

## Project Structure

```
open-source-photo-booth/
│
├── .python-version          # Pins Python 3.11 for uv
├── pyproject.toml           # Project metadata and dependencies
├── README.md
│
├── src/
│   ├── app.py               # Main loop, state machine, capture logic
│   ├── vision_engine.py     # MediaPipe processing, gesture classification, alpha blending
│   ├── ui_manager.py        # Sticker state, button ROI, dwell collision logic
│   └── config.py            # Global constants (FPS, sizes, thresholds)
│
├── assets/
│   ├── stickers/            # SVG/PNG sticker files (loaded automatically at startup)
│   │   ├── archlinux.svg
│   │   ├── firefoxbrowser.svg
│   │   ├── git.svg
│   │   ├── gnome.svg
│   │   ├── gnu.svg
│   │   ├── linux.svg
│   │   ├── python.svg
│   │   ├── rust.svg
│   │   └── ubuntu.svg
│   └── footer/
│       └── kokoa_logo.png   # Brand watermark — bottom-right on live view and captures
│
├── captures/                # Auto-created — saved photos land here
│
└── tests/
    ├── conftest.py          # Adds src/ to sys.path for imports
    ├── test_vision_engine.py
    └── test_ui_manager.py
```

---

## Requirements

| Dependency     | Version   | Purpose                                              |
|----------------|-----------|------------------------------------------------------|
| Python         | 3.11+     | Runtime — uses `match-case` syntax                   |
| opencv-python  | ≥ 4.13    | Camera capture, frame manipulation, display          |
| mediapipe      | 0.10.9    | Real-time hand landmark detection (pinned version)   |
| numpy          | ≥ 1.26    | Vectorised alpha blending operations                 |
| cairosvg       | latest    | SVG → PNG rasterisation for sticker loading          |
| pillow         | ≥ 10.0    | Robust PNG loading with alpha                        |
| uv             | latest    | Environment and package management                   |

> **Note:** MediaPipe is pinned to `0.10.9` because newer versions removed the `solutions` API used by this project.

**Hardware:** Any webcam supported by OpenCV. A CPU capable of running MediaPipe at ≥ 25 FPS is recommended (most modern laptops qualify).

---

## Installation

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository

```bash
git clone https://github.com/your-username/open-source-photo-booth.git
cd open-source-photo-booth
```

### 3. Set up the environment

`uv` reads `.python-version` and `pyproject.toml` automatically — no manual venv creation needed.

```bash
uv sync
```

This will:
- Download and pin Python 3.11 if not already available
- Create a local virtual environment
- Install all dependencies with locked, reproducible versions

### 4. Add sticker assets

Place your sticker files inside `assets/stickers/`. Both `.svg` and `.png` (with transparency) formats are supported. The app loads all matching files at startup, resizes them to `STICKER_CANVAS_SIZE`, and tints them white. See the [Stickers](#stickers) section for recommended sources.

---

## Usage

```bash
# Run the application
uv run python src/app.py

# Run tests
uv run pytest tests/

# Lint with ruff
uv run ruff check src/
```

Press `Q` to quit at any time. Captured photos are saved automatically to `captures/` with a timestamp filename such as `photo_20240315_143022.png`.

---

## Hand Gestures

The application tracks your dominant hand and recognises the following gestures:

| Gesture | Symbol | Action |
|---------|--------|--------|
| Peace sign (held) | ✌️ | Arc progress fills over ~0.7 s, then triggers 3-second countdown |
| Pointing (index only) | ☝️ | Hover over UI buttons / drag stickers |
| No hand detected | — | Idle — all dwell counters reset |

### How gesture detection works

MediaPipe Hands returns 21 landmarks per hand. Each finger is considered **extended** when the tip landmark's Y coordinate is above (less than) its MCP (knuckle) joint Y coordinate. Python 3.11 `match-case` maps the combination of four finger states to a named gesture:

```
(index, middle, ring, pinky) extended:
  (True,  True,  False, False) → PEACE       ✌️
  (True,  False, False, False) → POINTING    ☝️
  any other combination        → NONE
```

---

## How It Works

### Alpha Blending

OpenCV works in BGR. Sticker images are loaded with `cv2.IMREAD_UNCHANGED` to preserve the alpha channel (BGRA format). SVG files are first rasterised in memory by `cairosvg.svg2png()` then decoded by `cv2.imdecode()`. All stickers have their colour channels forced to white (`img[..., :3] = 255`) while the alpha shape is kept, giving a consistent white-on-transparent look against any background. Blending operates in `float32` to avoid integer overflow:

```
output = α · sticker_BGR + (1 − α) · background_BGR
```

Where `α` is the sticker's alpha channel normalised to `[0.0, 1.0]`. Stickers that partially overflow the screen edges are silently clipped.

### Dwell-Based Collision (ROI with Frame Delay)

Raw landmark coordinates jitter naturally, even when the user holds still. A single-frame "touch" check would fire constantly. Instead, each button maintains a **dwell counter**:

1. Every frame where the index fingertip is **inside** the button ROI, the counter increments.
2. If the finger **leaves** the ROI for even one frame, the counter resets to zero.
3. When the counter reaches `COLLISION_DWELL_FRAMES` (default: 12 frames ≈ 0.4 s at 30 FPS), the action fires **once**.
4. A **cooldown** of 20 frames prevents the same button from firing again immediately.

A progress bar rendered inside each button gives real-time visual feedback on dwell progress.

### PEACE Gesture Dwell (Anti False-Positive)

The capture trigger applies the same dwell concept to the gesture itself. A single frame of ✌️ classification does not start the countdown — the gesture must be held for `PEACE_DWELL_FRAMES` (default: 20 frames ≈ 0.7 s). A cyan arc drawn around the index fingertip fills as the gesture is held, giving clear visual feedback. If the gesture is broken before the threshold, the counter resets to zero.

### Capture Pipeline

When a capture gesture is confirmed:

1. The app enters `COUNTDOWN` state and records the start timestamp.
2. Each frame renders `3 → 2 → 1` centred on screen with a black halo for readability.
3. At `t = 0`, `save_clean_capture()` is called with the **raw frame** (before any UI overlay is drawn).
4. The function composites only the placed stickers onto the raw frame using `alpha_blend()`, then stamps the Kokoa logo watermark in the bottom-right corner.
5. The result is saved as a PNG to `captures/` — UI elements are never included.
6. Placed stickers are cleared so the next session starts with a clean canvas.
7. The app enters `CAPTURED` state: a white flash fades over ~12 frames, then returns to `IDLE`.

### State Machine

```
         ✌️ held (PEACE_DWELL_FRAMES)
IDLE ──────────────────────────────► COUNTDOWN
  ▲                                       │
  │    flash fades out                    │  countdown reaches 0
  │                                       ▼
CAPTURED ◄────────────────────── (photo saved, stickers cleared)
```

---

## Configuration

All tuneable parameters live in `src/config.py`. Edit this file to adjust behaviour without touching application logic.

```python
# Camera resolution
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30

# Sticker UI
STRIP_HEIGHT        = 130    # Height of selection strip pinned to the top of the frame
STICKERS_PER_PAGE   = 4      # Max stickers shown at once
STICKER_THUMB_SIZE  = 90     # Thumbnail size (px) in the strip
STICKER_CANVAS_SIZE = 160    # Size (px) when placed on screen

# Collision detection
COLLISION_DWELL_FRAMES = 12  # Frames finger must stay in ROI (~0.4 s at 30 FPS)
BUTTON_W               = 60  # Button width (px)
BUTTON_H               = 80  # Button height (px)

# Capture
COUNTDOWN_SECONDS  = 3
PEACE_DWELL_FRAMES = 20      # Frames ✌️ must be held before countdown starts (~0.7 s at 30 FPS)

# MediaPipe
MP_MAX_HANDS          = 1
MP_MIN_DETECTION_CONF = 0.7
MP_MIN_TRACKING_CONF  = 0.6
MP_DETECT_SCALE       = 0.5  # Resize factor before MediaPipe inference (halves pixels processed)

# Footer logo
FOOTER_LOGO        = Path("assets/footer/kokoa_logo.png")
FOOTER_LOGO_HEIGHT = 200     # px — logo height in bottom-right corner
FOOTER_MARGIN      = 16      # px — gap from frame edges
```

---

## Architecture

The codebase is split into three focused modules:

### `vision_engine.py`
Pure perception layer. No UI state, no side effects on the application.

- `VisionEngine.process_frame(bgr_frame)` — downscales the frame to 50% before running MediaPipe inference (halves pixels processed without reducing accuracy), stores results, then returns a `HandState` dataclass with landmarks, classified gesture, and index fingertip pixel coordinates.
- `VisionEngine.draw_landmarks(frame)` — draws the full hand skeleton (dots + connections) on the frame using MediaPipe's drawing utilities. Called every IDLE frame so the user can see what the system is tracking.
- `VisionEngine.alpha_blend(canvas, sticker, x, y)` — static method. Composites a BGRA sticker over a BGR canvas at the given position. Handles boundary clipping.
- `VisionEngine._classify(landmarks)` — maps 4 finger extension booleans to a `Gesture` enum value using `match-case`.

### `ui_manager.py`
Presentation and interaction state. Knows nothing about MediaPipe internals.

- `UIManager.__init__` — loads all stickers (PNG and SVG via cairosvg), applies white tint (`img[..., :3] = 255`), pre-computes thumbnail-sized copies to avoid per-frame resizing. Pre-allocates the dark strip background array.
- `UIManager.setup_layout(w, h)` — computes all button ROI positions from the actual frame dimensions. Called once per frame **before** `update_collisions()` so collision checks always match the rendered positions.
- `ButtonROI` — dataclass that encapsulates a rectangular hit area plus dwell counter, cooldown, and `progress()` for the visual feedback bar.
- `UIManager.update_collisions(index_tip, is_pointing)` — advances or resets dwell counters; guards the drag logic against the UI strip region to prevent conflicts with thumbnail hover.
- `UIManager.render_ui(frame)` — draws the selection strip (pinned to the **top** of the frame) and nav buttons using pre-computed thumbnails. Does **not** draw placed stickers (those are drawn in the main loop under the UI layer).
- `UIManager.clear_stickers()` — removes all placed stickers. Called automatically after each capture.

### `app.py`
Thin orchestrator. Owns the OpenCV window and the application state machine.

- `_load_logo(path, height)` — loads the footer logo as BGRA, scaled to the target height while preserving aspect ratio.
- `_stamp_logo(frame, logo)` — composites the logo onto the bottom-right corner of any frame using `VisionEngine.alpha_blend`.
- `save_clean_capture(raw_frame, ui, engine, logo)` — composites placed stickers onto a clean copy of the raw frame, stamps the logo, and saves to `captures/`.
- `render_countdown(frame, n)` — draws the large countdown number with halo.
- `main()` — initialises camera, `VisionEngine`, `UIManager`, and logo; runs the main loop; skips MediaPipe inference during COUNTDOWN and CAPTURED states to save CPU.

---

## Stickers

Sticker files can be `.svg` or `.png` with a transparent background. Minimum recommended resolution: **256 × 256 px** for PNGs. SVGs are rasterised at `STICKER_CANVAS_SIZE` automatically. All stickers are rendered in white, giving a consistent clean look against any background.

Recommended sources for official FOSS logos:

- **Simple Icons** — `simpleicons.org` (monochrome SVGs, consistent style — this project's current sticker set)
- **Wikimedia Commons** — `commons.wikimedia.org` (SVG → export as PNG)
- **Project official repos** — most projects include a `logo/` or `branding/` directory

To add a new sticker, drop the `.svg` or `.png` file into `assets/stickers/`. It will appear in the selection strip automatically on the next run.

---

## Contributing

Contributions are welcome! Here are some areas where help is appreciated:

- Additional FOSS sticker packs
- Improved gesture classifier (e.g., thumbs up, fist for drag)
- Multi-hand support for simultaneous sticker placement
- Sticker scaling via pinch gesture
- Undo/redo for sticker placement
- GUI settings panel (resolution, dwell threshold)
- Packaging as a standalone executable with PyInstaller

### Development setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/your-username/open-source-photo-booth.git
cd open-source-photo-booth
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

Please open an issue before submitting large pull requests to discuss the approach.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Individual sticker assets may carry their own licenses (typically the trademark/brand guidelines of each respective project). The sticker images are used for decorative, non-commercial purposes. Always check the licensing terms of each logo before redistribution.

---

<p align="center">
  Made with 🐧 and open source tools.<br>
  <em>No proprietary software was harmed in the making of this project.</em>
</p>
