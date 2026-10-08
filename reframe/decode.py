"""Decode the picture area of a video into frames with ffmpeg.

ffmpeg does rotation, cropping to the picture area, frame-rate reduction and
scaling in C; Python only sees the frames it analyses. Timing is by the
output frame rate (ffmpeg's fps filter resamples variable frame rate input
onto a fixed grid), so frame i is at time i / fps.

Frames are read from ffmpeg one at a time. They are kept as raw arrays while
the whole decode fits in RAW_BUDGET bytes (every short clip); longer videos
keep each frame as a high-quality JPEG and decode it on access, so a
6-minute 1080p phone video needs a few hundred MB instead of many GB.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .probe import Rect, VideoInfo

RAW_BUDGET = 1_000_000_000      # bytes of raw frames kept uncompressed
JPEG_QUALITY = 95


class FrameStore(Sequence):
    """A list of frames: raw arrays, or JPEG bytes decoded on access."""

    def __init__(self, compressed: bool, gray: bool):
        self.compressed = compressed
        self.gray = gray
        self._items: list = []

    def append(self, frame: np.ndarray) -> None:
        if self.compressed:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            self._items.append(buf.tobytes() if ok else frame.copy())
        else:
            self._items.append(frame)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        item = self._items[i]
        if isinstance(item, bytes):
            flag = cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR
            return cv2.imdecode(np.frombuffer(item, np.uint8), flag)
        return item

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]


@dataclass
class Frames:
    fps: float
    width: int
    height: int
    data: FrameStore        # frames: (h, w, 3) BGR uint8, or (h, w) gray
    scale: float            # analysis pixels per source pixel (of the picture area)

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.fps


def decode(info: VideoInfo, area: Rect, fps: float, max_width: int = 640, gray: bool = False, upscale_to: int = 0) -> Frames:
    """`upscale_to`: analyse small sources at this width (helps detectors find
    small faces in low-resolution uploads)."""
    width = min(max_width, max(area.w, upscale_to))
    width -= width % 2
    height = int(round(area.h * width / area.w / 2) * 2)
    channels = 1 if gray else 3
    frame_bytes = width * height * channels
    expected = int(info.duration * fps) + 2
    store = FrameStore(compressed=expected * frame_bytes > RAW_BUDGET, gray=gray)
    vf = f"crop={area.w}:{area.h}:{area.x}:{area.y},fps={fps},scale={width}:{height}:flags=area"
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", info.path, "-an", "-vf", vf,
         "-pix_fmt", "gray" if gray else "bgr24", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    shape = (height, width) if gray else (height, width, 3)
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        store.append(np.frombuffer(buf, dtype=np.uint8).reshape(shape))
    err = proc.stderr.read() if proc.stderr else b""
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, "ffmpeg", stderr=err)
    return Frames(fps=fps, width=width, height=height, data=store, scale=width / area.w)
