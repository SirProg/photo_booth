from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum, auto
from config import *


class Gesture(Enum):
    NONE     = auto()
    PEACE    = auto()   # ✌ photo trigger
    POINTING = auto()   # ☝ navigation / drag


@dataclass
class HandState:
    landmarks:    list
    gesture:      Gesture
    index_tip_px: tuple[int, int]


class VisionEngine:
    def __init__(self) -> None:
        self._mp_hands   = mp.solutions.hands
        self._draw_utils = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )
        self._last_results = None

    def process_frame(self, bgr_frame: np.ndarray) -> HandState | None:
        h, w = bgr_frame.shape[:2]

        # Downscale before sending to MediaPipe — halves the number of pixels
        # the model processes without meaningfully reducing detection quality.
        small   = cv2.resize(bgr_frame, (int(w * MP_DETECT_SCALE), int(h * MP_DETECT_SCALE)))
        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        self._last_results = results

        if not results.multi_hand_landmarks:
            return None

        lm  = results.multi_hand_landmarks[0].landmark
        tip = lm[8]  # INDEX_FINGER_TIP
        # Landmarks are normalized [0,1] — multiply by full-res dims directly
        index_px = (int(tip.x * w), int(tip.y * h))

        return HandState(
            landmarks=lm,
            gesture=self._classify(lm),
            index_tip_px=index_px,
        )

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw hand skeleton on frame for all detected hands."""
        if self._last_results is None or not self._last_results.multi_hand_landmarks:
            return
        dot  = self._draw_utils.DrawingSpec(color=(0, 255, 120), thickness=1, circle_radius=3)
        line = self._draw_utils.DrawingSpec(color=(255, 255, 255), thickness=1)
        for hand_lm in self._last_results.multi_hand_landmarks:
            self._draw_utils.draw_landmarks(
                frame, hand_lm, self._mp_hands.HAND_CONNECTIONS, dot, line,
            )

    def _classify(self, lm) -> Gesture:
        def extended(tip, mcp): return lm[tip].y < lm[mcp].y

        idx   = extended(8,  5)
        mid   = extended(12, 9)
        ring  = extended(16, 13)
        pinky = extended(20, 17)

        match (idx, mid, ring, pinky):
            case (True,  True,  False, False): return Gesture.PEACE
            case (True,  False, False, False): return Gesture.POINTING
            case _:                            return Gesture.NONE

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
