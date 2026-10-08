"""Decode the picture area of a video into numpy frames with ffmpeg.

ffmpeg does rotation, cropping to the picture area, frame-rate reduction and
scaling in C; Python only sees the frames it analyses. Timing is by the
output frame rate (ffmpeg's fps filter resamples variable frame rate input
onto a fixed grid), so frame i is at time i / fps.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from .probe import Rect, VideoInfo


@dataclass
class Frames:
    fps: float
    width: int
    height: int
    data: np.ndarray        # (n, h, w, 3) BGR uint8, or (n, h, w) gray
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
    vf = f"crop={area.w}:{area.h}:{area.x}:{area.y},fps={fps},scale={width}:{height}:flags=area"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", info.path, "-an", "-vf", vf,
         "-pix_fmt", "gray" if gray else "bgr24", "-f", "rawvideo", "pipe:1"],
        check=True, capture_output=True,
    ).stdout
    frame_bytes = width * height * channels
    n = len(raw) // frame_bytes
    shape = (n, height, width) if gray else (n, height, width, 3)
    data = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(shape)
    return Frames(fps=fps, width=width, height=height, data=data, scale=width / area.w)
