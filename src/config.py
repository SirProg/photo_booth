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

MP_MAX_HANDS          = 1
MP_MIN_DETECTION_CONF = 0.7
MP_MIN_TRACKING_CONF  = 0.6
MP_DETECT_SCALE       = 0.5   # resize factor before MediaPipe — halves pixels processed

ASSETS_DIR    = Path("assets/stickers")
STICKER_FILES = (
    sorted(ASSETS_DIR.glob("*.png")) + sorted(ASSETS_DIR.glob("*.svg"))
) if ASSETS_DIR.exists() else []

FOOTER_LOGO        = Path("assets/footer/kokoa_logo.png")
FOOTER_LOGO_HEIGHT = 200   # px — logo height in the bottom-right corner
FOOTER_MARGIN      = 16   # px — gap from frame edges
