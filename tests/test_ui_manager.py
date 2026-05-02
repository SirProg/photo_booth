import pytest
from ui_manager import ButtonROI
from config import COLLISION_DWELL_FRAMES


# ── ButtonROI.contains ─────────────────────────────────────────────────────────

def test_contains_inside():
    btn = ButtonROI("x", 10, 20, 50, 60)
    assert btn.contains((30, 40))


def test_contains_on_edge():
    btn = ButtonROI("x", 10, 20, 50, 60)
    assert btn.contains((10, 20))
    assert btn.contains((50, 60))


def test_contains_outside():
    btn = ButtonROI("x", 10, 20, 50, 60)
    assert not btn.contains((5,  40))
    assert not btn.contains((30, 65))


# ── ButtonROI.update / dwell ───────────────────────────────────────────────────

def test_fires_after_dwell_threshold():
    btn   = ButtonROI("x", 0, 0, 100, 100)
    fired = sum(1 for _ in range(COLLISION_DWELL_FRAMES) if btn.update(True))
    assert fired == 1


def test_dwell_resets_on_one_frame_outside():
    btn = ButtonROI("x", 0, 0, 100, 100)
    for _ in range(COLLISION_DWELL_FRAMES - 1):
        btn.update(True)
    btn.update(False)          # leave for just one frame
    assert btn.dwell == 0


def test_cooldown_prevents_immediate_refire():
    btn   = ButtonROI("x", 0, 0, 100, 100)
    # Run enough frames to trigger once + exhaust cooldown + trigger again
    fires = sum(1 for _ in range(COLLISION_DWELL_FRAMES * 3) if btn.update(True))
    assert fires == 1  # cooldown (20 frames) blocks second fire within this window


def test_no_fire_without_reaching_threshold():
    btn = ButtonROI("x", 0, 0, 100, 100)
    for _ in range(COLLISION_DWELL_FRAMES - 1):
        btn.update(True)
    assert btn.dwell == COLLISION_DWELL_FRAMES - 1
    assert btn.progress() < 1.0


# ── ButtonROI.progress ────────────────────────────────────────────────────────

def test_progress_starts_at_zero():
    btn = ButtonROI("x", 0, 0, 100, 100)
    assert btn.progress() == 0.0


def test_progress_halfway():
    btn = ButtonROI("x", 0, 0, 100, 100)
    half = COLLISION_DWELL_FRAMES // 2
    for _ in range(half):
        btn.update(True)
    assert abs(btn.progress() - half / COLLISION_DWELL_FRAMES) < 0.01


def test_progress_caps_at_one():
    btn = ButtonROI("x", 0, 0, 100, 100)
    btn.dwell = COLLISION_DWELL_FRAMES + 5  # force beyond threshold
    assert btn.progress() == 1.0


# ── ButtonROI.reset ───────────────────────────────────────────────────────────

def test_reset_clears_state():
    btn = ButtonROI("x", 0, 0, 100, 100)
    for _ in range(COLLISION_DWELL_FRAMES - 1):
        btn.update(True)
    btn.reset()
    assert btn.dwell == 0
    assert btn.cooldown == 0
