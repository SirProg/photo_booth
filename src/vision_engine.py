from __future__ import annotations
import math
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto
from config import *


class Gesture(Enum):
    NONE     = auto()
    PEACE    = auto()   # ✌ photo trigger
    POINTING = auto()   # ☝ navigation / drag


@dataclass
class Landmark:
    """A single hand landmark in full-frame normalized coordinates."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class HandState:
    landmarks:    list[Landmark]
    gesture:      Gesture
    index_tip_px: tuple[int, int]
    stale:        bool = False   # coasted from an earlier frame, not detected now


# ── temporal smoothing ────────────────────────────────────────────────────────

class OneEuroArray:
    """One Euro filter over an (N, 3) landmark array.

    Cuts jitter when the hand is still — the dominant failure mode far from the
    camera, where a couple of pixels of noise swing the normalized coordinates —
    while staying responsive during fast motion, unlike a fixed-alpha EMA.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._d_cutoff   = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x_prev:  np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev:  float | None      = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev  = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev  = t
            return x

        dt = t - self._t_prev
        if not 0.0 < dt <= 1.0:          # first frame after a stall / clock jump
            dt = 1.0 / TARGET_FPS
        self._t_prev = t

        dx     = (x - self._x_prev) / dt
        a_d    = self._alpha(self._d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self._min_cutoff + self._beta * np.abs(dx_hat)
        a      = self._alpha(cutoff, dt)
        x_hat  = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


# ── vision engine ─────────────────────────────────────────────────────────────

class VisionEngine:
    # (tip, mcp) pairs used for the extension test
    _FINGERS = ((8, 5), (12, 9), (16, 13), (20, 17))

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands

        def _make(static: bool):
            return self._mp_hands.Hands(
                static_image_mode=static,
                max_num_hands=MP_MAX_HANDS,
                model_complexity=MP_MODEL_COMPLEXITY,
                min_detection_confidence=MP_MIN_DETECTION_CONF,
                min_tracking_confidence=MP_MIN_TRACKING_CONF,
            )

        self._finder  = _make(True)
        self._tracker = _make(not MP_TRACKER_VIDEO_MODE)
        self._clahe = cv2.createCLAHE(
            clipLimit=MP_CLAHE_CLIP,
            tileGridSize=(MP_CLAHE_GRID, MP_CLAHE_GRID),
        )
        self._gamma_luts: dict[int, np.ndarray] = {}
        self._filter = OneEuroArray(
            MP_SMOOTH_MIN_CUTOFF, MP_SMOOTH_BETA, MP_SMOOTH_D_CUTOFF,
        )
        self.reset()

    def reset(self) -> None:
        """Drop all tracking state — call when the app stops feeding frames."""
        self._drop_lock()
        self._search_idx = 0    # index 0 is the full frame — nearby hands lock

    def _drop_lock(self) -> None:
        """Forget the tracked hand but keep sweeping the search grid.

        The search cursor deliberately survives: resetting it on every empty
        frame pinned acquisition to the first window, so the grid that finds a
        distant hand was never reached.
        """
        self._roi:        tuple[int, int, int, int] | None = None
        self._last_pts:   np.ndarray | None = None
        self._draw_px:    np.ndarray | None = None
        self._misses      = 0
        self._roi_misses  = 0
        self._stale       = False
        self._votes: deque[Gesture] = deque(maxlen=GESTURE_VOTE_WINDOW)
        self._stable_gesture = Gesture.NONE
        self._filter.reset()

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """Current tracking window in full-frame pixels — for debug overlays."""
        return self._roi

    # ── main entry point ──────────────────────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray) -> HandState | None:
        h, w = bgr_frame.shape[:2]
        now  = time.perf_counter()

        # Exactly one inference per frame. A same-frame fallback from the ROI to
        # a search window doubled the cost and put the loop under 22 FPS, which
        # cost more tracking than the extra attempt ever recovered.
        if self._roi is not None:
            # Retry a missed frame with the static detector on the same window.
            # Video-mode tracking is twice as fast but noticeably worse at
            # re-acquiring once it has lost the thread; the static pass
            # re-anchors it. Widening the window instead — the obvious move —
            # made things worse: the resized window failed the size check on
            # the next hit, so the crop churned and the tracker never settled.
            model = self._tracker if self._roi_misses == 0 else self._finder
            pts = self._detect(bgr_frame, self._roi, MP_ROI_INPUT_SIDE, model)
            if pts is None:
                self._roi_misses += 1
                if self._roi_misses > MP_ROI_MAX_MISSES:
                    self._roi = None
        else:
            pts = self._detect(
                bgr_frame, self._next_search_rect(w, h), MP_SEARCH_INPUT_SIDE,
                self._finder,
            )

        if pts is None:
            # Coast on the last known pose for a few frames. A single dropped
            # detection no longer yanks the cursor away or resets a dwell.
            self._misses += 1
            if self._last_pts is None or self._misses > MP_LOST_GRACE_FRAMES:
                self._drop_lock()
                return None
            pts         = self._last_pts
            self._stale = True
        else:
            self._misses     = 0
            self._roi_misses = 0
            self._stale      = False
            self._last_pts   = pts
            self._roi        = self._settle_roi(self._roi, pts, w, h)

        smooth = self._filter(pts, now) if MP_SMOOTH_ENABLED else pts

        self._draw_px = np.column_stack((
            smooth[:, 0] * w, smooth[:, 1] * h,
        )).astype(np.int32)

        lms = [Landmark(float(p[0]), float(p[1]), float(p[2])) for p in smooth]
        tip = lms[8]

        return HandState(
            landmarks=lms,
            gesture=self._vote(self._classify(lms)),
            index_tip_px=(int(tip.x * w), int(tip.y * h)),
            stale=self._stale,
        )

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect(
        self,
        bgr:    np.ndarray,
        rect:   tuple[int, int, int, int],
        target: int,
        model,
    ) -> np.ndarray | None:
        """Run MediaPipe on one crop; return (21, 3) full-frame normalized pts."""
        fh, fw = bgr.shape[:2]
        x1, y1, x2, y2 = rect
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        crop     = bgr[y1:y2, x1:x2]
        ch, cw   = crop.shape[:2]
        scale    = target / min(cw, ch)
        if abs(scale - 1.0) > 0.05:
            interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            crop   = cv2.resize(crop, (max(1, round(cw * scale)),
                                       max(1, round(ch * scale))), interpolation=interp)

        rgb = cv2.cvtColor(self._enhance(crop), cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = model.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        # Landmarks are normalized to the crop — map them back to the full frame.
        lm  = results.multi_hand_landmarks[0].landmark
        out = np.empty((len(lm), 3), dtype=np.float64)
        for i, p in enumerate(lm):
            out[i, 0] = (x1 + p.x * cw) / fw
            out[i, 1] = (y1 + p.y * ch) / fh
            out[i, 2] = p.z
        return out

    def _enhance(self, bgr: np.ndarray) -> np.ndarray:
        """Normalize contrast and brightness of the model input only.

        The user never sees this image; it exists so a backlit or dimly lit hand
        still carries the local contrast the palm detector keys on.
        """
        if not MP_ENHANCE_ENABLED:
            return bgr
        lab       = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lum, a, b = cv2.split(lab)
        lum       = self._clahe.apply(lum)
        mean      = float(lum.mean())
        if mean < MP_GAMMA_TARGET:
            gamma = max(0.4, mean / MP_GAMMA_TARGET)   # < 1 brightens
            lum   = cv2.LUT(lum, self._gamma_lut(gamma))
        return cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        key = int(round(gamma * 20))          # cache in 0.05 steps
        lut = self._gamma_luts.get(key)
        if lut is None:
            g   = key / 20.0
            lut = np.clip(
                ((np.arange(256) / 255.0) ** g) * 255.0, 0, 255,
            ).astype(np.uint8)
            self._gamma_luts[key] = lut
        return lut

    # ── tracking windows ──────────────────────────────────────────────────────

    def _next_search_rect(self, w: int, h: int) -> tuple[int, int, int, int]:
        rel = MP_SEARCH_WINDOWS[self._search_idx % len(MP_SEARCH_WINDOWS)]
        self._search_idx += 1
        rx1, ry1, rx2, ry2 = rel
        return (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))

    def _settle_roi(
        self,
        roi:  tuple[int, int, int, int] | None,
        pts:  np.ndarray,
        w:    int,
        h:    int,
    ) -> tuple[int, int, int, int]:
        """Keep the current window unless the hand has drifted or resized."""
        fresh = self._roi_from(pts, w, h)
        if roi is None:
            return fresh

        rx1, ry1, rx2, ry2 = roi
        side   = rx2 - rx1
        margin = side * MP_ROI_HYSTERESIS
        xs, ys = pts[:, 0] * w, pts[:, 1] * h

        centred = (
            xs.min() >= rx1 + margin and xs.max() <= rx2 - margin
            and ys.min() >= ry1 + margin and ys.max() <= ry2 - margin
        )
        sized = abs((fresh[2] - fresh[0]) - side) <= side * MP_ROI_RESIZE_TOL
        return roi if (centred and sized) else fresh

    @staticmethod
    def _roi_from(
        pts: np.ndarray, w: int, h: int,
    ) -> tuple[int, int, int, int]:
        """Square window around the hand, padded and clamped to the frame."""
        xs = pts[:, 0] * w
        ys = pts[:, 1] * h
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())

        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * MP_ROI_PAD)
        side = min(max(side, MP_ROI_MIN_PX), float(min(w, h)))
        half = side / 2.0

        cx = min(max((x1 + x2) / 2.0, half), w - half)
        cy = min(max((y1 + y2) / 2.0, half), h - half)
        return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))

    # ── rendering ─────────────────────────────────────────────────────────────

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw the hand skeleton from the smoothed, full-frame landmarks."""
        if self._draw_px is None:
            return
        dot  = (0, 200, 200) if self._stale else (0, 255, 120)
        line = (170, 170, 170) if self._stale else (255, 255, 255)
        pts  = self._draw_px
        for a, b in self._mp_hands.HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), line, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, dot, -1, cv2.LINE_AA)

    # ── gesture classification ────────────────────────────────────────────────

    @staticmethod
    def _palm_axis(lm) -> tuple[float, float, float]:
        """Unit vector wrist → middle-finger MCP, plus the palm span.

        Measuring extension along this axis instead of screen-vertical keeps a
        tilted or rotated hand classifying correctly. Degenerate input falls
        back to screen-up so the test doubles behave like the old comparison.
        """
        vx = lm[9].x - lm[0].x
        vy = lm[9].y - lm[0].y
        n  = math.hypot(vx, vy)
        if n < 1e-6:
            return 0.0, -1.0, 0.0
        return vx / n, vy / n, n

    def _classify(self, lm) -> Gesture:
        ax, ay, span = self._palm_axis(lm)
        margin = MP_FINGER_EXT_MARGIN * span

        def extended(tip: int, mcp: int) -> bool:
            dx = lm[tip].x - lm[mcp].x
            dy = lm[tip].y - lm[mcp].y
            return (dx * ax + dy * ay) > margin

        idx, mid, ring, pinky = (extended(t, m) for t, m in self._FINGERS)

        match (idx, mid, ring, pinky):
            case (True,  True,  False, False): return Gesture.PEACE
            case (True,  False, False, False): return Gesture.POINTING
            case _:                            return Gesture.NONE

    def _vote(self, gesture: Gesture) -> Gesture:
        """Report a gesture only once it wins a majority of the recent window.

        Without this a single misread frame far from the camera resets the peace
        dwell counter, making the trigger feel unreachable at distance.
        """
        self._votes.append(gesture)
        winner, count = Counter(self._votes).most_common(1)[0]
        if count > len(self._votes) // 2:
            self._stable_gesture = winner
        return self._stable_gesture

    # ── compositing ───────────────────────────────────────────────────────────

    @staticmethod
    def alpha_blend(
        canvas:  np.ndarray,
        sticker: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        sh, sw = sticker.shape[:2]
        ch, cw = canvas.shape[:2]

        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + sw, cw), min(y + sh, ch)

        if x2 <= x1 or y2 <= y1:
            return canvas

        sx1 = x1 - x; sy1 = y1 - y
        sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)

        roi    = canvas[y1:y2, x1:x2].astype(np.float32)
        s_crop = sticker[sy1:sy2, sx1:sx2]
        alpha  = s_crop[..., 3:] / 255.0
        s_bgr  = s_crop[..., :3].astype(np.float32)

        blended = alpha * s_bgr + (1.0 - alpha) * roi
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        return canvas
