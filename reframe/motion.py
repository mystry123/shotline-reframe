"""Camera motion in the source: how the original camera pans, tilts and zooms
between consecutive analysis frames.

Corners are tracked between frames (pyramidal Lucas-Kanade) and a similarity
transform (pan, tilt, zoom, roll) is fitted with RANSAC. Points on people
(their faces, widened to cover the body) are left out, so a moving subject
isn't taken for a moving camera. When too few points agree, a phase
correlation gives the translation alone, marked unreliable. Nothing is
estimated across a cut.

Units: dx, dy are camera movement in picture widths / heights between the
two frames (positive = the camera moves right / down, so the scene slides
left / up); scale > 1 = zooming in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .decode import Frames

WORK_WIDTH = 640
MAX_CORNERS = 300
MIN_INLIERS = 30
MIN_INLIER_SHARE = 0.5
PHASE_RELIABLE = 0.3   # phase-correlation peak response needed to trust the fallback


@dataclass
class Motion:
    t: float          # time of the later frame
    dx: float
    dy: float
    scale: float
    rotation: float   # radians
    inliers: float    # share of tracked points agreeing (0 for the fallback)
    reliable: bool


def _mask(shape, faces) -> np.ndarray | None:
    """255 where corners may be taken: everywhere but people."""
    if not faces:
        return None
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    for d in faces:
        x0, x1 = int((d.x - d.w) * w), int((d.x + 2 * d.w) * w)
        y0, y1 = int((d.y - 0.5 * d.h) * h), int((d.y + 5 * d.h) * h)
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 0
    return mask if mask.mean() > 25 else None      # people everywhere: use the whole frame


def _gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = g.shape
    if w > WORK_WIDTH:
        g = cv2.resize(g, (WORK_WIDTH, int(round(h * WORK_WIDTH / w))), interpolation=cv2.INTER_AREA)
    return g


def _phase(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """Translation of the content (pixels) and the peak response."""
    win = cv2.createHanningWindow(prev.shape[::-1], cv2.CV_32F)
    (tx, ty), response = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32), win)
    return tx, ty, response


def estimate(frames: Frames, faces_per_frame: list[list], shot_starts: list[int]) -> list[Motion | None]:
    """One entry per frame: the camera motion from the previous frame, or None
    at the first frame and at the first frame of each shot."""
    out: list[Motion | None] = [None] * len(frames.data)
    starts = set(shot_starts)
    prev = None
    for i, frame in enumerate(frames.data):
        cur = _gray(frame)
        if prev is None or i in starts:
            prev = cur
            continue
        h, w = cur.shape
        t = i / frames.fps
        pts = cv2.goodFeaturesToTrack(prev, MAX_CORNERS, 0.01, 8, mask=_mask(prev.shape, faces_per_frame[i - 1]))
        motion = None
        if pts is not None and len(pts) >= MIN_INLIERS:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            ok = status.ravel() == 1
            p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
            if len(p0) >= MIN_INLIERS:
                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None and inl is not None:
                    share = float(inl.mean())
                    if inl.sum() >= MIN_INLIERS and share >= MIN_INLIER_SHARE:
                        a, b = M[0, 0], M[1, 0]
                        scale = float(np.hypot(a, b))
                        # How the frame centre moves with the content = the pan (zoom removed).
                        c = np.array([w / 2, h / 2])
                        moved = M[:, :2] @ c + M[:, 2]
                        dx, dy = -(moved - c) / np.array([w, h])
                        motion = Motion(t, float(dx), float(dy), scale, float(np.arctan2(b, a)), share, True)
        if motion is None:
            tx, ty, response = _phase(prev, cur)
            motion = Motion(t, float(-tx / w), float(-ty / h), 1.0, 0.0, 0.0, response > PHASE_RELIABLE)
        out[i] = motion
        prev = cur
    return out


def shot_path(motion: list[Motion | None], first: int, last: int) -> np.ndarray:
    """Cumulative camera position over frames first..last-1 of one shot:
    rows (x, y, log zoom), starting at 0."""
    rows = [np.zeros(3)]
    for i in range(first + 1, last):
        m = motion[i] if i < len(motion) else None
        step = np.array([m.dx, m.dy, np.log(m.scale)]) if m is not None else np.zeros(3)
        rows.append(rows[-1] + step)
    return np.array(rows)
