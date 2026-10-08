"""Probe a video and find the area that holds the real picture.

Many uploads are re-exports with the real picture inside black bars, often
with text burned into the bars (titles above, animated captions below). Plain
black-bar detection is fooled by that text, so rows and columns are scored by
how *picture-like* they are across many frames: a picture row has most of its
pixels lit in most frames; a caption row has a few bright glyph pixels or is
black most of the time.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int          # display width (after rotation)
    height: int         # display height (after rotation)
    fps: float
    duration: float
    has_audio: bool
    rotation: int


@dataclass(frozen=True)
class Rect:
    """Pixel rectangle in display coordinates."""
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict:
        return asdict(self)


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("The file has no video stream.")
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    rotation = int(video.get("tags", {}).get("rotate", rotation)) % 360
    w, h = int(video["width"]), int(video["height"])
    if rotation in (90, 270, -90, -270):
        w, h = h, w
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    return VideoInfo(path, w, h, fps, duration, has_audio, rotation)


def sample_frames(info: VideoInfo, count: int = 40, width: int = 0) -> np.ndarray:
    """`count` frames spread over the video, grayscale, at display orientation
    (ffmpeg applies rotation). Optionally scaled to `width`."""
    fps = max(count / max(info.duration, 0.1), 0.01)
    scale = f",scale={width}:-2" if width else ""
    w = width or info.width
    h = int(round(info.height * (w / info.width) / 2) * 2) if width else info.height
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", info.path, "-an",
         "-vf", f"fps={fps:.6f}{scale}", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
        check=True, capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // (w * h)
    return frames[: n * w * h].reshape(n, h, w)


def _longest_true_run(mask: np.ndarray, max_gap: int = 0) -> tuple[int, int]:
    """Longest run of True values, bridging gaps of up to `max_gap` False values
    (a dark stripe inside the picture shouldn't split it in two)."""
    mask = mask.copy()
    if max_gap > 0:
        idx = np.flatnonzero(mask)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= max_gap + 1:
                mask[a:b] = True
    best = (0, 0)
    start = None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def find_picture_area(info: VideoInfo, frames: np.ndarray | None = None) -> Rect:
    """The rectangle holding the real picture (full frame when there are no bars)."""
    if frames is None:
        frames = sample_frames(info)
    if frames.size == 0:
        return Rect(0, 0, info.width, info.height)

    # Above near-black: bars encode as 0-6 even after compression, while dark,
    # moody footage still has most pixels above this.
    lit = frames > 10
    # Share of lit pixels per row/column, per frame; then the median over frames.
    row_fill = np.median(lit.mean(axis=2), axis=0)
    col_fill = np.median(lit.mean(axis=1), axis=0)
    # Picture rows/cols are mostly lit in most frames. Caption/title rows are
    # sparse (glyphs) or blank most of the time, so their median stays low.
    h_, w_ = frames.shape[1:]
    rows = _longest_true_run(row_fill > 0.45, max_gap=max(1, h_ // 33))
    cols = _longest_true_run(col_fill > 0.45, max_gap=max(1, w_ // 33))

    h, w = frames.shape[1:]
    sy, sx = info.height / h, info.width / w
    y0, y1 = rows if rows[1] - rows[0] >= h * 0.25 else (0, h)
    x0, x1 = cols if cols[1] - cols[0] >= w * 0.25 else (0, w)
    rect = Rect(int(round(x0 * sx)), int(round(y0 * sy)), int(round((x1 - x0) * sx)), int(round((y1 - y0) * sy)))
    # Even sizes keep ffmpeg's crop and scale filters happy.
    return Rect(rect.x, rect.y, rect.w - rect.w % 2, rect.h - rect.h % 2)
