import cv2
import numpy as np
import pytest
from collections import deque
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


# ── ROI geometry ──────────────────────────────────────────────────────────────

import config
from vision_engine import OneEuroArray


def _pts(cx: float, cy: float, half: float) -> np.ndarray:
    """21 landmarks spread over a square of half-width `half`, normalized."""
    p = np.full((21, 3), [cx, cy, 0.0], dtype=np.float64)
    p[0]  = [cx - half, cy - half, 0.0]
    p[20] = [cx + half, cy + half, 0.0]
    return p


def test_roi_is_square_and_padded():
    roi = VisionEngine._roi_from(_pts(0.5, 0.5, 0.05), 1280, 720)
    x1, y1, x2, y2 = roi
    assert (x2 - x1) == (y2 - y1)
    # hand spans 0.10 * 1280 = 128 px; padding widens it by (1 + 2*PAD)
    assert (x2 - x1) == pytest.approx(128 * (1 + 2 * config.MP_ROI_PAD), abs=2)


def test_roi_stays_inside_the_frame():
    for cx, cy in ((0.02, 0.02), (0.98, 0.98), (0.5, 0.0), (0.0, 0.5)):
        x1, y1, x2, y2 = VisionEngine._roi_from(_pts(cx, cy, 0.05), 1280, 720)
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720


def test_roi_never_smaller_than_the_floor():
    x1, y1, x2, y2 = VisionEngine._roi_from(_pts(0.5, 0.5, 0.001), 1280, 720)
    assert (x2 - x1) >= config.MP_ROI_MIN_PX


def test_roi_never_exceeds_the_short_side():
    x1, y1, x2, y2 = VisionEngine._roi_from(_pts(0.5, 0.5, 0.5), 1280, 720)
    assert (x2 - x1) <= 720


# ── ROI hysteresis ────────────────────────────────────────────────────────────

def _engine_shell() -> VisionEngine:
    e = VisionEngine.__new__(VisionEngine)
    e._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    e._gamma_luts = {}
    return e


def test_settle_roi_holds_a_centred_hand():
    e   = _engine_shell()
    pts = _pts(0.5, 0.5, 0.05)
    roi = VisionEngine._roi_from(pts, 1280, 720)
    # Same hand, same place — the window must not move, or the video-mode
    # tracker loses the geometry it depends on.
    assert e._settle_roi(roi, pts, 1280, 720) == roi


def test_settle_roi_recentres_when_the_hand_drifts_out():
    e     = _engine_shell()
    roi   = VisionEngine._roi_from(_pts(0.5, 0.5, 0.05), 1280, 720)
    moved = _pts(0.75, 0.5, 0.05)
    assert e._settle_roi(roi, moved, 1280, 720) != roi


def test_settle_roi_recentres_when_the_hand_grows():
    e   = _engine_shell()
    roi = VisionEngine._roi_from(_pts(0.5, 0.5, 0.03), 1280, 720)
    assert e._settle_roi(roi, _pts(0.5, 0.5, 0.12), 1280, 720) != roi


# ── search grid ───────────────────────────────────────────────────────────────

def test_search_starts_with_the_full_frame():
    e = VisionEngine.__new__(VisionEngine)
    e._search_idx = 0
    assert e._next_search_rect(1280, 720) == (0, 0, 1280, 720)


def test_search_sweeps_every_window_before_repeating():
    e = VisionEngine.__new__(VisionEngine)
    e._search_idx = 0
    n    = len(config.MP_SEARCH_WINDOWS)
    seen = {e._next_search_rect(1280, 720) for _ in range(n)}
    assert len(seen) == n
    # and it wraps around rather than running off the end
    assert e._next_search_rect(1280, 720) == (0, 0, 1280, 720)


def test_search_tiles_cover_the_frame():
    xs = [w[2] for w in config.MP_SEARCH_WINDOWS]
    ys = [w[3] for w in config.MP_SEARCH_WINDOWS]
    assert max(xs) == pytest.approx(1.0)
    assert max(ys) == pytest.approx(1.0)
    assert min(w[0] for w in config.MP_SEARCH_WINDOWS) == pytest.approx(0.0)


# ── crop → full-frame coordinate mapping ──────────────────────────────────────

class _FakePoint:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _FakeModel:
    """Stands in for MediaPipe: always reports a hand at fixed crop coords."""
    def __init__(self, x, y):
        self._pts = [_FakePoint(x, y) for _ in range(21)]

    def process(self, rgb):
        hand = type("H", (), {"landmark": self._pts})()
        return type("R", (), {"multi_hand_landmarks": [hand]})()


def test_detect_maps_crop_coords_back_to_the_full_frame():
    e     = _engine_shell()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rect  = (400, 200, 656, 456)          # 256 x 256 window
    out   = e._detect(frame, rect, 256, _FakeModel(0.5, 0.5))
    # centre of that window is (528, 328) in full-frame pixels
    assert out[0, 0] == pytest.approx(528 / 1280, abs=1e-6)
    assert out[0, 1] == pytest.approx(328 / 720, abs=1e-6)


def test_detect_maps_corners_of_a_resized_crop():
    e     = _engine_shell()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rect  = (100, 50, 612, 434)           # 512 x 384, downscaled to reach target
    out   = e._detect(frame, rect, 256, _FakeModel(0.0, 1.0))
    assert out[0, 0] == pytest.approx(100 / 1280, abs=1e-3)
    assert out[0, 1] == pytest.approx(434 / 720, abs=1e-3)


def test_detect_rejects_a_degenerate_window():
    e = _engine_shell()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert e._detect(frame, (10, 10, 11, 11), 256, _FakeModel(0.5, 0.5)) is None


# ── gesture debounce ──────────────────────────────────────────────────────────

def _voter() -> VisionEngine:
    e = VisionEngine.__new__(VisionEngine)
    e._votes = deque(maxlen=config.GESTURE_VOTE_WINDOW)
    e._stable_gesture = Gesture.NONE
    return e


def test_vote_survives_a_single_bad_frame():
    e = _voter()
    for _ in range(config.GESTURE_VOTE_WINDOW):
        e._vote(Gesture.PEACE)
    # one misread must not drop the gesture — that is what reset the dwell
    assert e._vote(Gesture.NONE) == Gesture.PEACE


def test_vote_switches_once_the_new_gesture_wins():
    e = _voter()
    for _ in range(config.GESTURE_VOTE_WINDOW):
        e._vote(Gesture.PEACE)
    out = [e._vote(Gesture.POINTING) for _ in range(config.GESTURE_VOTE_WINDOW)]
    assert out[-1] == Gesture.POINTING


# ── rotation-invariant classification ─────────────────────────────────────────

def _hand(extended: dict[int, bool]) -> list:
    """A geometrically plausible 21-point hand, palm pointing up."""
    lm = [_FakeLM(0.5) for _ in range(21)]
    for i in range(21):
        lm[i] = _FakeLM(0.5); lm[i].x = 0.5
    lm[0].x, lm[0].y = 0.50, 0.80                     # wrist
    mcp = {5: 0.44, 9: 0.50, 13: 0.56, 17: 0.62}
    for mcp_id, x in mcp.items():
        lm[mcp_id].x, lm[mcp_id].y = x, 0.55
    for tip, mcp_id in ((8, 5), (12, 9), (16, 13), (20, 17)):
        lm[tip].x = mcp[mcp_id]
        lm[tip].y = 0.35 if extended.get(tip, False) else 0.62
    return lm


def _rotate(lm: list, deg: float) -> list:
    import math
    a, ox, oy = math.radians(deg), lm[0].x, lm[0].y
    out = []
    for p in lm:
        dx, dy = p.x - ox, p.y - oy
        q = _FakeLM(oy + dx * math.sin(a) + dy * math.cos(a))
        q.x = ox + dx * math.cos(a) - dy * math.sin(a)
        out.append(q)
    return out


def test_geometric_hand_classifies_peace():
    engine = VisionEngine.__new__(VisionEngine)
    assert engine._classify(_hand({8: True, 12: True})) == Gesture.PEACE


@pytest.mark.parametrize("deg", [0, 30, 90, 180, -60])
def test_classification_is_rotation_invariant(deg):
    engine = VisionEngine.__new__(VisionEngine)
    lm = _rotate(_hand({8: True, 12: True}), deg)
    assert engine._classify(lm) == Gesture.PEACE


@pytest.mark.parametrize("deg", [0, 45, 90, 135])
def test_pointing_is_rotation_invariant(deg):
    engine = VisionEngine.__new__(VisionEngine)
    lm = _rotate(_hand({8: True}), deg)
    assert engine._classify(lm) == Gesture.POINTING


# ── One Euro filter ───────────────────────────────────────────────────────────

def test_filter_passes_the_first_sample_through():
    f = OneEuroArray(1.5, 1.5, 1.0)
    x = np.full((21, 3), 0.4)
    assert np.allclose(f(x, 0.0), x)


def test_filter_converges_on_a_still_hand():
    f = OneEuroArray(1.5, 1.5, 1.0)
    x = np.full((21, 3), 0.4)
    f(x, 0.0)
    for i in range(1, 40):
        out = f(x, i / 30.0)
    assert np.allclose(out, x, atol=1e-3)


def test_filter_damps_jitter():
    f     = OneEuroArray(1.5, 1.5, 1.0)
    base  = np.full((21, 3), 0.4)
    rng   = np.random.default_rng(0)
    noisy = []
    out   = []
    for i in range(60):
        s = base + rng.normal(0, 0.01, base.shape)
        noisy.append(s[0, 0])
        out.append(f(s, i / 30.0)[0, 0])
    # the filtered signal must vary less than the raw one it came from
    assert np.std(out[10:]) < np.std(noisy[10:])


def test_filter_reset_forgets_history():
    f = OneEuroArray(1.5, 1.5, 1.0)
    f(np.full((21, 3), 0.9), 0.0)
    f.reset()
    x = np.full((21, 3), 0.1)
    assert np.allclose(f(x, 1.0), x)
