"""Shot boundaries.

Same idea as PySceneDetect's AdaptiveDetector: a per-frame content change score
(HSV + edges), compared with the average change of the frames around it, so
fast camera motion inside a shot (high scores everywhere) doesn't count, while a
cut (one isolated spike) does. Runs on the picture area only, so captions or
emoji animating in black bars can't create cuts.

Also handled:
- flash frames / strobes: a spike immediately followed by the reverse spike
  back to the previous look is not a cut;
- very short shots: cuts closer than `min_shot` seconds are merged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .decode import Frames


@dataclass(frozen=True)
class Shot:
    index: int
    start: float   # seconds, inclusive
    end: float     # seconds, exclusive

    @property
    def duration(self) -> float:
        return self.end - self.start


def _frame_features(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200).astype(np.float32)
    return np.dstack([hsv, edges])


def content_scores(frames: Frames) -> np.ndarray:
    """Change score between each frame and the previous one (0 for frame 0)."""
    scores = np.zeros(len(frames.data), dtype=np.float32)
    prev = None
    for i, frame in enumerate(frames.data):
        feat = _frame_features(frame)
        if prev is not None:
            # Mean absolute difference per channel: hue, saturation, value, edges
            # (weights as PySceneDetect's ContentDetector: 1, 1, 1, 0.5 for edges is
            # its "luma + edges" style; we keep edges lighter).
            diff = np.abs(feat - prev).reshape(-1, 4).mean(axis=0)
            diff[0] = min(diff[0], 180 - diff[0])  # hue wraps around
            scores[i] = float(diff[0] + diff[1] + diff[2] + 0.25 * diff[3]) / 3.25
        prev = feat
    return scores


def detect_shots(
    frames: Frames,
    duration: float,
    adaptive_ratio: float = 3.0,
    min_content: float = 15.0,
    window: int = 2,
    min_shot: float = 0.5,
) -> list[Shot]:
    scores = content_scores(frames)
    n = len(scores)
    cut_frames: list[int] = []
    for i in range(1, n):
        lo, hi = max(1, i - window), min(n, i + window + 1)
        neighbours = [scores[j] for j in range(lo, hi) if j != i]
        avg = float(np.mean(neighbours)) if neighbours else 0.0
        ratio = scores[i] / max(avg, 1e-3)
        if scores[i] < min_content or ratio < adaptive_ratio:
            continue
        # Flash frame: the frame after jumps straight back to the old look.
        if i + 1 < n and i >= 1:
            before = frames.data[i - 1].astype(np.int16)
            after = frames.data[i + 1].astype(np.int16)
            if np.abs(after - before).mean() < 6:
                continue
        cut_frames.append(i)

    times = [0.0]
    for f in cut_frames:
        t = f / frames.fps
        if t - times[-1] >= min_shot and duration - t >= min_shot:
            times.append(t)
    times.append(duration)
    return [Shot(k, times[k], times[k + 1]) for k in range(len(times) - 1)]
