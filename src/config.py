from pathlib import Path

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
TARGET_FPS   = 30

STRIP_HEIGHT        = 130
STICKERS_PER_PAGE   = 4
STICKER_THUMB_SIZE  = 90
STICKER_CANVAS_SIZE = 160

COLLISION_DWELL_FRAMES = 12
BUTTON_W = 60
BUTTON_H = 80

COUNTDOWN_SECONDS  = 3
PEACE_DWELL_FRAMES = 20   # frames ✌️ must be held before countdown starts (~0.67s at 30 FPS)
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

# ── Hand tracking ─────────────────────────────────────────────────────────────
# MediaPipe's palm detector rescales whatever it is given to 192x192 internally.
# Feeding it the whole 1280x720 frame turns a hand at 2 m into ~15 px of input,
# which is why detection used to drop out at moderate distance. The pipeline
# below crops a square ROI around the last known hand instead, so the hand fills
# the detector's input regardless of how far away the user stands.

MP_MAX_HANDS          = 1
MP_MODEL_COMPLEXITY   = 1     # 1 = full landmark model (0 = lite, faster/worse far away)
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF  = 0.5

# Two MediaPipe instances, because the two jobs want opposite settings.
#
# The searcher runs in static mode: every search window is unrelated to the
# last, so there is no cross-frame state worth keeping and a stale hand rect
# would only mislead it.
#
# The tracker runs in video mode, which reuses the previous frame's landmarks
# and skips the palm detector entirely. Measured on 256x256 ROI crops with a
# hand present: 39.2 ms/frame static vs 19.4 ms video, at 100 % vs 98 %
# re-detection. That 2x is what keeps the loop above 30 FPS. It only holds
# while the crop geometry is stable frame to frame — hence MP_ROI_HYSTERESIS.
MP_TRACKER_VIDEO_MODE = True

# ROI tracking — the square window fed to MediaPipe once a hand is locked on.
# Padding was swept against the booth captures: 0.4 re-detects 57 % of frames,
# 0.6 gets 74 %, and 0.9 reaches 93 % and plateaus. Crops tighter than that
# clip the wrist context the palm detector keys on.
MP_ROI_PAD        = 0.9
MP_ROI_MIN_PX     = 160   # never crop tighter than this in source pixels
MP_ROI_INPUT_SIDE = 256   # ROI is resized to this before inference
MP_ROI_MAX_MISSES = 2     # ROI retries before falling back to a full search

# Hold the window still until the hand nears its edge or changes size. Chasing
# the hand every frame would shift the crop under the video-mode tracker, whose
# whole speed advantage comes from the geometry staying put.
MP_ROI_HYSTERESIS = 0.22  # re-centre once the hand enters this margin of the window
MP_ROI_RESIZE_TOL = 0.35  # or once the ideal window differs this much in size
MP_LOST_GRACE_FRAMES = 4  # keep coasting on the last pose for this many misses


def _search_grid(cols: int, rows: int, overlap: float) -> list[tuple]:
    """Overlapping tiles covering the frame, nearest the centre first."""
    tw = 1.0 / (cols - (cols - 1) * overlap)
    th = 1.0 / (rows - (rows - 1) * overlap)
    sx, sy = tw * (1.0 - overlap), th * (1.0 - overlap)
    tiles = [
        (c * sx, r * sy, min(c * sx + tw, 1.0), min(r * sy + th, 1.0))
        for r in range(rows) for c in range(cols)
    ]
    # A raised hand tends to sit near the middle of the booth frame.
    tiles.sort(key=lambda t: ((t[0] + t[2]) / 2 - 0.5) ** 2
                           + ((t[1] + t[3]) / 2 - 0.5) ** 2)
    return tiles


# Acquisition search — one window per frame while no hand is locked on.
# The palm detector squashes its input to 192x192, so searching the whole
# 1280x720 frame shrinks a hand at arm's length past the point of detection.
# Measured on the booth captures with the subject scaled down to simulate
# distance (hand width in the source frame in brackets):
#
#            95 px    66 px    47 px
#   full     22.7 %   13.6 %   13.6 %
#   2x2      90.9 %   36.4 %   36.4 %
#   3x2      95.5 %   68.2 %   36.4 %
#   4x3     100.0 %   95.5 %   63.6 %
#
# The full frame stays first so a nearby hand still locks on in a single frame;
# the 4x3 tiles then sweep in ~430 ms at 30 FPS for anyone standing back.
MP_SEARCH_TILES      = (4, 3)
MP_SEARCH_OVERLAP    = 0.5
MP_SEARCH_INPUT_SIDE = 256   # short side of the search crop before inference
MP_SEARCH_WINDOWS    = (
    (0.0, 0.0, 1.0, 1.0),
    *_search_grid(*MP_SEARCH_TILES, MP_SEARCH_OVERLAP),
)

# Illumination normalization — applied only to the MediaPipe input, never to
# the frame the user sees. Rescues backlit and dim-room detection.
MP_ENHANCE_ENABLED = True
MP_CLAHE_CLIP      = 2.0
MP_CLAHE_GRID      = 8
MP_GAMMA_TARGET    = 110   # mean L below this triggers a gamma lift

# One Euro smoothing of the landmarks (normalized units, so speed is in 1/s).
MP_SMOOTH_ENABLED    = True
MP_SMOOTH_MIN_CUTOFF = 1.5   # lower = smoother when the hand is still
MP_SMOOTH_BETA       = 1.5   # higher = less lag when the hand moves fast
MP_SMOOTH_D_CUTOFF   = 1.0

# Finger extension is measured along the palm axis rather than screen-vertical,
# so a tilted hand still classifies correctly. Margin scales with hand size.
MP_FINGER_EXT_MARGIN = 0.15

# Gesture debounce — a gesture must win a majority of this many frames before
# it is reported, so a single bad frame no longer resets the peace dwell.
GESTURE_VOTE_WINDOW = 5

ASSETS_DIR    = Path("assets/stickers")
STICKER_FILES = (
    sorted(ASSETS_DIR.glob("*.png")) + sorted(ASSETS_DIR.glob("*.svg"))
) if ASSETS_DIR.exists() else []

FOOTER_LOGO        = Path("assets/footer/kokoa_logo.png")
FOOTER_LOGO_HEIGHT = 200   # px — logo height in the bottom-right corner
FOOTER_MARGIN      = 16   # px — gap from frame edges
