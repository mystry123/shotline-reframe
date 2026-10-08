"""Burned-in text outside the real picture (titles, captions in the bars).

When a video is an export with the picture inside bars, the bars often carry
text: a title above, animated captions below. Reframing to another shape cuts
the bars away, so the user is asked whether to keep that text. This module
finds it and describes it for that question:

- region:  top or bottom of the picture
- kind:    "title" (the same text the whole time) or "captions" (changing)
- active:  share of the video it's on screen
- keyable: the text sits on a plain dark background, so it can be lifted out
           and laid over the new frame cleanly; if not, the editor suggests
           Shotline captions instead
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .probe import Rect, VideoInfo, sample_frames

TEXT_ROW_MIN = 0.01        # share of lit pixels in a row that counts as "has text"
MIN_ON_SCREEN_S = 1.0      # text must be on screen at least this long (titles are often brief)
DARK = 24                  # background at or below this luma is "plain dark"


@dataclass
class TextBand:
    id: str
    region: str            # "top" | "bottom"
    kind: str              # "title" | "captions"
    rect: Rect             # source pixels
    active: float          # 0..1 share of the video
    keyable: bool

    def describe(self) -> str:
        where = "at the top" if self.region == "top" else "at the bottom"
        return f"{'A title' if self.kind == 'title' else 'Captions'} {where}"


def _bands(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return []
    bands = [[rows[0], rows[0] + 1]]
    for r in rows[1:]:
        if r - bands[-1][1] <= max_gap:
            bands[-1][1] = r + 1
        else:
            bands.append([r, r + 1])
    return [(a, b) for a, b in bands]


class _Rows:
    """A strip of frames (rows lo.. of the full frame) indexed in full-frame rows."""

    def __init__(self, strip: np.ndarray, lo: int):
        self.strip, self.lo = strip, lo

    def __getitem__(self, key):
        frames, rows, *rest = key if isinstance(key, tuple) else (key, slice(None))
        if isinstance(rows, slice):
            rows = slice(None if rows.start is None else rows.start - self.lo,
                         None if rows.stop is None else rows.stop - self.lo, rows.step)
        else:
            rows = rows - self.lo
        return self.strip[(frames, rows, *rest)]


def detect_text_bands(info: VideoInfo, area: Rect) -> list[TextBand]:
    if area.h >= info.height - 4 and area.w >= info.width - 4:
        return []  # no bars, nothing outside the picture
    # 2 samples a second (at most 600) so a 1-second title is never missed.
    # Only the bars are decoded (a strip per bar), so long videos stay small.
    count = int(min(600, max(20, info.duration * 2)))
    H, W = info.height, info.width
    out: list[TextBand] = []
    for region, (lo, hi) in (("top", (0, area.y)), ("bottom", (area.y + area.h, H))):
        if hi - lo < 6:
            continue
        strip = sample_frames(info, count=count, rows=(lo, hi))
        if strip.size == 0:
            continue
        # The strip stands in for rows lo..hi of the frame: index it as if it were the frame.
        frames = _Rows(strip, lo)
        n = strip.shape[0]
        sample_s = info.duration / n
        min_samples = max(1, int(np.ceil(MIN_ON_SCREEN_S / sample_s)))
        lit = _Rows(strip > 60, lo)  # text is bright against the bars
        rows_lit = lit[:, lo:hi, :].mean(axis=2)                  # (n, rows)
        has_text = (rows_lit > TEXT_ROW_MIN).sum(axis=0) >= min_samples
        for a, b in _bands(has_text, max_gap=max(3, H // 60)):
            if b - a < 6:
                continue
            y0, y1 = lo + a, lo + b
            band = frames[:, y0:y1, :]
            on = (lit[:, y0:y1, :].mean(axis=(1, 2)) > TEXT_ROW_MIN / 2)
            if on.sum() < min_samples:
                continue
            cols = np.flatnonzero(lit[on, y0:y1, :].any(axis=(0, 1)))
            x0, x1 = int(cols.min()), int(cols.max()) + 1
            # Title: the lit pixels barely change between samples where it's shown.
            shown = band[on] > 60
            change = np.mean([np.mean(shown[i] != shown[i + 1]) for i in range(len(shown) - 1)]) if len(shown) > 1 else 0.0
            kind = "title" if change < 0.01 else "captions"
            # Keyable: everything that isn't text (or its soft anti-aliased
            # edge, 2 px around each letter) is near-black.
            kernel = np.ones((5, 5), np.uint8)
            bg_pixels = []
            for frame in band[on][:: max(1, int(on.sum()) // 20)]:
                text_mask = cv2.dilate((frame > 60).astype(np.uint8), kernel) > 0
                bg_pixels.append(frame[~text_mask])
            background = np.concatenate(bg_pixels) if bg_pixels else np.array([], np.uint8)
            keyable = bool(background.size == 0 or np.percentile(background, 98) <= DARK)
            pad = 4
            rx0, ry0 = max(0, x0 - pad), max(0, int(y0) - pad)
            rect = Rect(int(rx0), int(ry0), int(min(W, x1 + pad) - rx0), int(min(H, int(y1) + pad) - ry0))
            out.append(TextBand(f"{region}-{len(out)}", region, kind, rect, round(float(on.mean()), 2), keyable))
    # Merge bands of the same kind in the same region (multi-line captions).
    merged: list[TextBand] = []
    for t in out:
        prev = next((m for m in merged if m.region == t.region and m.kind == t.kind), None)
        if prev is None:
            merged.append(t)
            continue
        x0, y0 = min(prev.rect.x, t.rect.x), min(prev.rect.y, t.rect.y)
        x1 = max(prev.rect.x + prev.rect.w, t.rect.x + t.rect.w)
        y1 = max(prev.rect.y + prev.rect.h, t.rect.y + t.rect.h)
        prev.rect = Rect(x0, y0, x1 - x0, y1 - y0)
        prev.active = max(prev.active, t.active)
        prev.keyable = prev.keyable and t.keyable
    for i, t in enumerate(merged):
        t.id = f"{t.region}-{t.kind}"
    return merged
