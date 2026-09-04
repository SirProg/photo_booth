from __future__ import annotations
import cv2
import numpy as np
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from config import *
from vision_engine import VisionEngine, Gesture
from ui_manager import UIManager


class AppState(Enum):
    IDLE      = auto()
    COUNTDOWN = auto()
    CAPTURED  = auto()


def _load_logo(path: Path, height: int) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w    = img.shape[:2]
    new_w   = max(1, int(w * height / h))
    return cv2.resize(img, (new_w, height))


def _stamp_logo(frame: np.ndarray, logo: np.ndarray) -> np.ndarray:
    fh, fw = frame.shape[:2]
    lh, lw = logo.shape[:2]
    x = fw - lw - FOOTER_MARGIN
    y = fh - lh - FOOTER_MARGIN
    return VisionEngine.alpha_blend(frame, logo, x, y)


def render_countdown(frame: np.ndarray, n: int) -> None:
    h, w   = frame.shape[:2]
    text   = str(n)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 8.0
    thick  = 16
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thick + 8)
    cv2.putText(frame, text, org, font, scale, (0, 230, 255), thick)


def _render_debug(
    frame:  np.ndarray,
    engine: VisionEngine,
    hand,
    fps:    float,
) -> None:
    """Overlay the tracking window and rate — press 'd' to tune reach on site."""
    roi = engine.roi
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)

    if hand is None:
        status = "SEARCHING"
    elif hand.stale:
        status = "COASTING"
    else:
        status = f"LOCKED {hand.gesture.name}"

    lines = [f"{fps:4.1f} FPS", status, f"roi={roi}"]
    for i, text in enumerate(lines):
        org = (12, 28 + i * 26)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 1)


def save_clean_capture(
    raw_frame: np.ndarray,
    ui:        UIManager,
    engine:    VisionEngine,
    logo:      np.ndarray | None = None,
) -> Path:
    clean = raw_frame.copy()
    for s in ui.placed_stickers:
        clean = VisionEngine.alpha_blend(clean, s.img, s.x, s.y)
    if logo is not None:
        clean = _stamp_logo(clean, logo)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURES_DIR / f"photo_{ts}.png"
    cv2.imwrite(str(path), clean)
    return path


def main() -> None:
    cap = cv2.VideoCapture(0)
    # MJPG before the resolution request: most UVC webcams only reach 720p30
    # on MJPG, and fall back to a soft-focus 10 FPS YUYV mode otherwise.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    # A one-frame buffer keeps the hand position current — a deeper queue makes
    # the tracker chase where the hand was several frames ago.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    engine = VisionEngine()
    ui     = UIManager(STICKER_FILES)
    logo   = _load_logo(FOOTER_LOGO, FOOTER_LOGO_HEIGHT)

    state:           AppState        = AppState.IDLE
    countdown_start: float           = 0.0
    flash_alpha:     float           = 0.0
    last_raw:        np.ndarray | None = None
    peace_dwell:     int             = 0
    show_debug:      bool            = False
    fps:             float           = 0.0
    last_tick:       float           = time.time()

    cv2.namedWindow("Open Source Photo Booth", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame    = cv2.flip(frame, 1)
        last_raw = frame.copy()
        # Only run MediaPipe in IDLE — COUNTDOWN and CAPTURED don't need hand data
        hand = engine.process_frame(frame) if state == AppState.IDLE else None

        # Draw placed stickers under UI layer
        for s in ui.placed_stickers:
            frame = VisionEngine.alpha_blend(frame, s.img, s.x, s.y)

        match state:
            case AppState.IDLE:
                # Landmarks drawn before UI so the strip renders on top
                engine.draw_landmarks(frame)

                # Compute button positions from the actual frame size first,
                # then check collisions against those fresh positions.
                fh, fw = frame.shape[:2]
                ui.setup_layout(fw, fh)
                ui.update_collisions(
                    hand.index_tip_px if hand else None,
                    is_pointing=hand is not None and hand.gesture == Gesture.POINTING,
                )
                frame = ui.render_ui(frame)

                # Finger cursor — drawn on top of UI so the user can see exactly
                # where the system is tracking the index tip.
                if hand:
                    ix, iy = hand.index_tip_px
                    cv2.circle(frame, (ix, iy), 10, (0, 0, 0),    -1)
                    cv2.circle(frame, (ix, iy),  8, (0, 230, 255), -1)

                # Require PEACE_DWELL_FRAMES consecutive frames before triggering.
                # Draw a progress arc around the fingertip as visual feedback.
                if hand and hand.gesture == Gesture.PEACE:
                    peace_dwell += 1
                    ix, iy  = hand.index_tip_px
                    progress = peace_dwell / PEACE_DWELL_FRAMES
                    sweep    = int(360 * progress)
                    cv2.ellipse(frame, (ix, iy), (24, 24), -90, 0,     360,   (50,  50,  50),  2)
                    cv2.ellipse(frame, (ix, iy), (24, 24), -90, 0,     sweep, (0,  230, 255),  3)
                    if peace_dwell >= PEACE_DWELL_FRAMES:
                        state           = AppState.COUNTDOWN
                        countdown_start = time.time()
                        peace_dwell     = 0
                        # No frames reach the engine until IDLE resumes; drop
                        # the ROI so it re-acquires instead of coasting on a
                        # pose that is seconds old.
                        engine.reset()
                else:
                    peace_dwell = 0

            case AppState.COUNTDOWN:
                elapsed = time.time() - countdown_start
                n       = COUNTDOWN_SECONDS - int(elapsed)
                if n > 0:
                    render_countdown(frame, n)
                else:
                    path = save_clean_capture(last_raw, ui, engine, logo)
                    print(f"Saved: {path}")
                    ui.clear_stickers()
                    state       = AppState.CAPTURED
                    flash_alpha = 1.0

            case AppState.CAPTURED:
                if flash_alpha > 0:
                    white = np.ones_like(frame, dtype=np.uint8) * 255
                    cv2.addWeighted(white, flash_alpha, frame, 1 - flash_alpha, 0, frame)
                    flash_alpha -= 0.08
                else:
                    state = AppState.IDLE
                    ui.render_ui(frame)

        if logo is not None:
            frame = _stamp_logo(frame, logo)

        now       = time.time()
        dt        = now - last_tick
        last_tick = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

        if show_debug:
            _render_debug(frame, engine, hand, fps)

        cv2.imshow("Open Source Photo Booth", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
