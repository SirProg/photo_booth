import numpy as np
import pytest
from vision_engine import VisionEngine, Gesture


# ── alpha_blend ────────────────────────────────────────────────────────────────

def _make_sticker(size: int, bgr: tuple, alpha: int) -> np.ndarray:
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[..., :3] = bgr
    img[...,  3] = alpha
    return img


def test_alpha_blend_returns_same_shape():
    canvas  = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _make_sticker(40, (200, 100, 50), 128)
    result  = VisionEngine.alpha_blend(canvas, sticker, 10, 10)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_alpha_blend_fully_opaque_overwrites_canvas():
    canvas  = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _make_sticker(40, (100, 150, 200), 255)
    result  = VisionEngine.alpha_blend(canvas, sticker, 0, 0)
    # Center of the sticker region must equal sticker color (BGR)
    assert result[20, 20, 0] == 100
    assert result[20, 20, 1] == 150
    assert result[20, 20, 2] == 200


def test_alpha_blend_fully_transparent_leaves_canvas():
    canvas  = np.full((100, 100, 3), 42, dtype=np.uint8)
    sticker = _make_sticker(40, (255, 0, 0), 0)
    result  = VisionEngine.alpha_blend(canvas, sticker, 0, 0)
    assert result[10, 10, 0] == 42


def test_alpha_blend_completely_outside_does_not_raise():
    canvas  = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _make_sticker(40, (255, 255, 255), 255)
    result  = VisionEngine.alpha_blend(canvas, sticker, 200, 200)
    assert result.shape == (100, 100, 3)
    assert result.sum() == 0  # canvas untouched


def test_alpha_blend_partial_clip_does_not_raise():
    canvas  = np.zeros((100, 100, 3), dtype=np.uint8)
    sticker = _make_sticker(40, (255, 255, 255), 255)
    result  = VisionEngine.alpha_blend(canvas, sticker, 85, 85)
    assert result.shape == (100, 100, 3)


# ── gesture classifier ────────────────────────────────────────────────────────

class _FakeLM:
    """Minimal landmark mock: only .y matters for extension check."""
    def __init__(self, y: float):
        self.y = y
        self.x = 0.0


def _build_landmarks(finger_states: dict[int, bool]) -> list:
    """Build a 21-element fake landmark list.

    finger_states maps tip_id → True (extended) / False (closed).
    Extended means tip.y < mcp.y.
    """
    pairs = {8: 5, 12: 9, 16: 13, 20: 17}
    lm = [_FakeLM(0.5)] * 21

    for tip_id, mcp_id in pairs.items():
        extended = finger_states.get(tip_id, False)
        lm = list(lm)
        lm[mcp_id] = _FakeLM(0.5)
        lm[tip_id] = _FakeLM(0.3 if extended else 0.7)  # 0.3 < 0.5 → extended

    return lm


def test_gesture_peace():
    engine = VisionEngine.__new__(VisionEngine)
    lm = _build_landmarks({8: True, 12: True, 16: False, 20: False})
    assert engine._classify(lm) == Gesture.PEACE


def test_gesture_open_palm_is_none():
    # All four fingers extended is no longer a recognized gesture
    engine = VisionEngine.__new__(VisionEngine)
    lm = _build_landmarks({8: True, 12: True, 16: True, 20: True})
    assert engine._classify(lm) == Gesture.NONE


def test_gesture_pointing():
    engine = VisionEngine.__new__(VisionEngine)
    lm = _build_landmarks({8: True, 12: False, 16: False, 20: False})
    assert engine._classify(lm) == Gesture.POINTING


def test_gesture_none():
    engine = VisionEngine.__new__(VisionEngine)
    lm = _build_landmarks({8: False, 12: False, 16: False, 20: False})
    assert engine._classify(lm) == Gesture.NONE


def test_open_palm_removed_from_gesture_enum():
    assert not hasattr(Gesture, "OPEN_PALM")
