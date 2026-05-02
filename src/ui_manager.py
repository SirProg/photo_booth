from __future__ import annotations
import cv2
import numpy as np
import cairosvg
from dataclasses import dataclass
from pathlib import Path
from config import *


@dataclass
class StickerInstance:
    img:      np.ndarray
    x:        int
    y:        int
    name:     str  = ""
    dragging: bool = False


@dataclass
class ButtonROI:
    label:    str
    x1:       int
    y1:       int
    x2:       int
    y2:       int
    dwell:    int = 0
    cooldown: int = 0

    def contains(self, pt: tuple[int, int]) -> bool:
        return self.x1 <= pt[0] <= self.x2 and self.y1 <= pt[1] <= self.y2

    def update(self, inside: bool) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
            self.dwell = 0
            return False
        if inside:
            self.dwell += 1
            if self.dwell >= COLLISION_DWELL_FRAMES:
                self.dwell    = 0
                self.cooldown = 20
                return True
        else:
            self.dwell = 0
        return False

    def progress(self) -> float:
        return min(self.dwell / COLLISION_DWELL_FRAMES, 1.0)

    def reset(self) -> None:
        self.dwell    = 0
        self.cooldown = 0


class UIManager:
    def __init__(self, sticker_paths: list[Path]) -> None:
        self._library = [self._load(p) for p in sticker_paths]
        self._names   = [p.stem for p in sticker_paths]
        self._page    = 0
        self._placed: list[StickerInstance] = []
        self._drag_idx: int | None = None
        self._last_page = 0
        self._frame_w   = FRAME_WIDTH
        self._frame_h   = FRAME_HEIGHT

        self._btn_prev = ButtonROI("◀", 0, 0, 0, 0)
        self._btn_next = ButtonROI("▶", 0, 0, 0, 0)
        self._thumb_buttons = [
            ButtonROI(f"t{i}", 0, 0, 0, 0) for i in range(STICKERS_PER_PAGE)
        ]

        # Pre-compute thumbnail-sized versions — avoids cv2.resize every frame
        ts = STICKER_THUMB_SIZE
        self._thumbnails = [cv2.resize(img, (ts, ts)) for img in self._library]

        # Pre-allocated dark strip background — reused every frame
        self._strip_bg: np.ndarray | None = None

        self.setup_layout(FRAME_WIDTH, FRAME_HEIGHT)

    def setup_layout(self, w: int, h: int) -> None:
        """Compute all button ROI positions for a frame of size (w, h).
        Must be called once per frame before update_collisions."""
        self._frame_w = w
        self._frame_h = h

        # Strip sits at the TOP of the frame
        bx  = 10
        by  = (STRIP_HEIGHT - BUTTON_H) // 2
        bx2 = w - BUTTON_W - 10
        self._btn_prev.__dict__.update(x1=bx,  y1=by, x2=bx  + BUTTON_W, y2=by + BUTTON_H)
        self._btn_next.__dict__.update(x1=bx2, y1=by, x2=bx2 + BUTTON_W, y2=by + BUTTON_H)

        thumb_x1 = bx  + BUTTON_W + 10
        thumb_x2 = bx2 - 10
        spacing  = (thumb_x2 - thumb_x1) // max(STICKERS_PER_PAGE, 1)
        ts       = STICKER_THUMB_SIZE

        for i in range(STICKERS_PER_PAGE):
            tx = thumb_x1 + i * spacing + (spacing - ts) // 2
            ty = (STRIP_HEIGHT - ts) // 2
            self._thumb_buttons[i].__dict__.update(x1=tx, y1=ty, x2=tx + ts, y2=ty + ts)

    def _load(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".svg":
            png_bytes = cairosvg.svg2png(
                url=str(path.resolve()),
                output_width=STICKER_CANVAS_SIZE,
                output_height=STICKER_CANVAS_SIZE,
            )
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[..., :3] = 255  # tint all pixels white, alpha channel unchanged
        return cv2.resize(img, (STICKER_CANVAS_SIZE,) * 2)

    def _total_pages(self) -> int:
        return max(1, -(-len(self._library) // STICKERS_PER_PAGE))

    def current_page_stickers(self) -> list[np.ndarray]:
        start = self._page * STICKERS_PER_PAGE
        return self._library[start:start + STICKERS_PER_PAGE]

    def update_collisions(
        self,
        index_tip: tuple[int, int] | None,
        is_pointing: bool = False,
    ) -> None:
        pt = index_tip or (-1, -1)

        # Reset thumb dwell on page change
        if self._page != self._last_page:
            for btn in self._thumb_buttons:
                btn.reset()
            self._last_page = self._page

        # Nav buttons
        if self._btn_prev.update(self._btn_prev.contains(pt)):
            self._page = (self._page - 1) % self._total_pages()
        if self._btn_next.update(self._btn_next.contains(pt)):
            self._page = (self._page + 1) % self._total_pages()

        # Thumbnail buttons — dwell to place sticker at screen center
        page_stickers = self.current_page_stickers()
        for i, btn in enumerate(self._thumb_buttons[:len(page_stickers)]):
            if btn.update(btn.contains(pt)):
                lib_idx = self._page * STICKERS_PER_PAGE + i
                self._placed.append(StickerInstance(
                    img=self._library[lib_idx].copy(),
                    x=self._frame_w // 2 - STICKER_CANVAS_SIZE // 2,
                    y=self._frame_h // 2 - STICKER_CANVAS_SIZE // 2,
                    name=self._names[lib_idx],
                ))

        # Drag: POINTING picks up the closest placed sticker.
        # Skip drag when the finger is inside the top UI strip to avoid conflict
        # with thumbnail hover.
        in_ui_strip = index_tip is not None and index_tip[1] < STRIP_HEIGHT
        if index_tip and is_pointing and not in_ui_strip:
            if self._drag_idx is None:
                best_i, best_d = None, STICKER_CANVAS_SIZE  # full sticker width as radius
                for i, s in enumerate(self._placed):
                    cx = s.x + STICKER_CANVAS_SIZE // 2
                    cy = s.y + STICKER_CANVAS_SIZE // 2
                    d = ((index_tip[0] - cx) ** 2 + (index_tip[1] - cy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None:
                    self._drag_idx = best_i
                    self._placed[best_i].dragging = True
            else:
                s = self._placed[self._drag_idx]
                s.x = index_tip[0] - STICKER_CANVAS_SIZE // 2
                s.y = index_tip[1] - STICKER_CANVAS_SIZE // 2
        else:
            if self._drag_idx is not None:
                self._placed[self._drag_idx].dragging = False
                self._drag_idx = None

    def render_ui(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # Rebuild strip background only when frame width changes (e.g. first frame)
        if self._strip_bg is None or self._strip_bg.shape[1] != w:
            self._strip_bg = np.full((STRIP_HEIGHT, w, 3), (15, 15, 30), dtype=np.uint8)

        # Blend only the strip rows — avoids copying the entire frame
        strip_copy = frame[:STRIP_HEIGHT].copy()
        cv2.addWeighted(self._strip_bg, 0.75, strip_copy, 0.25, 0, frame[:STRIP_HEIGHT])

        # Nav buttons — positions come from setup_layout, just draw them
        for btn in (self._btn_prev, self._btn_next):
            p = btn.progress()
            cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2), (40, 40, 60), -1)
            if p > 0:
                fill_w = int((btn.x2 - btn.x1) * p)
                cv2.rectangle(frame, (btn.x1, btn.y1),
                              (btn.x1 + fill_w, btn.y1 + 4), (0, 220, 180), -1)
            cv2.putText(frame, btn.label,
                        (btn.x1 + 12, btn.y1 + BUTTON_H // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 240), 2)

        # Sticker thumbnails — use pre-computed thumbnails, no resize per frame
        ts = STICKER_THUMB_SIZE
        start = self._page * STICKERS_PER_PAGE
        for i, lib_idx in enumerate(range(start, min(start + STICKERS_PER_PAGE, len(self._library)))):
            btn   = self._thumb_buttons[i]
            tx, ty = btn.x1, btn.y1
            thumb = self._thumbnails[lib_idx]

            roi = frame[ty:ty + ts, tx:tx + ts]
            if roi.shape[:2] == (ts, ts):
                alpha   = thumb[..., 3:] / 255.0
                blended = alpha * thumb[..., :3].astype(np.float32) + (1.0 - alpha) * roi.astype(np.float32)
                frame[ty:ty + ts, tx:tx + ts] = blended.astype(np.uint8)

            p = btn.progress()
            if p > 0:
                fill_w = int(ts * p)
                cv2.rectangle(frame, (tx, ty + ts - 4), (tx + fill_w, ty + ts), (0, 220, 180), -1)

        return frame

    def clear_stickers(self) -> None:
        self._placed.clear()
        self._drag_idx = None

    @property
    def placed_stickers(self) -> list[StickerInstance]:
        return self._placed
