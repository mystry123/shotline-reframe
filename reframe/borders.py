"""Blurred bars: a picture placed over a blurred, enlarged copy of itself
(a phone video in a 16:9 frame, or a 16:9 video in a 9:16 one, as CapCut-
style tools make them). Black bars are found in probe.py; these aren't black,
so they need their own test.

From grayscale frames sampled across the video:

1. Sharpness profile: per column (side bars) or row (top/bottom bars), the
   median over frames of the mean |Laplacian|. The bands must be at least
   SHARPNESS_RATIO times blurrier than the picture. Candidate pictures have a
   standard shape (9:16, 4:5, 1:1, 3:4, 2:3, 16:9, 4:3) within 3%.
2. Echo test: the bands look like the picture scaled up to cover the frame
   and blurred: Pearson correlation over ECHO_CORR on ECHO_SHARE of frames.
3. Straight boundary: a gradient line at the picture's edges, much stronger
   than just beside it, in BOUNDARY_SHARE of frames.

Accepted when 1 and (2 or 3) hold, or 2 and 3 (an unblurred echo). Shallow
depth of field mimics test 1 but fails 2 and 3: a soft background is not a
copy of the subject and has no straight edge.
"""

from __future__ import annotations

import cv2
import numpy as np

STANDARD_ASPECTS = (9 / 16, 4 / 5, 1.0, 3 / 4, 2 / 3, 16 / 9, 4 / 3)
ASPECT_TOLERANCE = 0.03
SHARPNESS_RATIO = 4.0
ECHO_CORR = 0.6
ECHO_SHARE = 0.6
BOUNDARY_SHARE = 0.5
MIN_BAND = 0.03          # each side band at least this share of the frame (if present)
WORK_WIDTH = 640


def _profile(frames: np.ndarray, axis: str) -> np.ndarray:
    """Median over frames of mean |Laplacian| per column ("x") or row ("y")."""
    per = []
    for f in frames:
        lap = np.abs(cv2.Laplacian(f.astype(np.float32), cv2.CV_32F, ksize=3))
        per.append(lap.mean(axis=0) if axis == "x" else lap.mean(axis=1))
    return np.median(np.array(per), axis=0)


def _windows(profile: np.ndarray, other: int, axis: str, top: int = 6) -> list[tuple[int, int, float]]:
    """The `top` standard-shaped windows (start, length, sharpness ratio),
    ranked by how much sharper inside than outside they are (a difference:
    a ratio is fooled by perfectly flat areas outside, which score 0)."""
    size = len(profile)
    cs = np.r_[0.0, np.cumsum(profile)]
    total = cs[-1]
    floor = 0.05 * float(np.mean(profile)) + 1e-3
    found = []
    for aspect in STANDARD_ASPECTS:
        inner = aspect * other if axis == "x" else other / aspect
        if not (0.25 * size <= inner <= (1 - 2 * MIN_BAND) * size):
            continue
        for length in range(int(inner * (1 - ASPECT_TOLERANCE)), int(inner * (1 + ASPECT_TOLERANCE)) + 1):
            for start in range(0, size - length + 1):
                before, after = start, size - start - length
                if (0 < before < MIN_BAND * size) or (0 < after < MIN_BAND * size):
                    continue
                inside = (cs[start + length] - cs[start]) / length
                outside = (total - (cs[start + length] - cs[start])) / max(size - length, 1)
                found.append((inside - outside, start, length, inside / (outside + floor)))
    found.sort(reverse=True)
    picked: list[tuple[int, int, float]] = []
    for _, start, length, ratio in found:          # distinct windows only
        if all(abs(start - a) > 4 or abs(length - b) > 4 for a, b, _ in picked):
            picked.append((start, length, ratio))
        if len(picked) == top:
            break
    return picked


def _echo_share(frames: np.ndarray, start: int, length: int, axis: str) -> float:
    """Share of frames whose bands match the picture scaled up to cover the frame."""
    hits = valid = 0
    for f in frames:
        h, w = f.shape
        if axis == "x":
            inner = f[:, start:start + length]
            scale = w / length
            cover = cv2.resize(inner, (w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            oy = (cover.shape[0] - h) // 2
            cover = cover[oy:oy + h, :w]
        else:
            inner = f[start:start + length, :]
            scale = h / length
            cover = cv2.resize(inner, (int(round(w * scale)), h), interpolation=cv2.INTER_AREA)
            ox = (cover.shape[1] - w) // 2
            cover = cover[:h, ox:ox + w]
        small = (64, max(8, int(64 * h / w)))
        a = cv2.GaussianBlur(cv2.resize(f, small, interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.5)
        b = cv2.GaussianBlur(cv2.resize(cover, small, interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.5)
        mask = np.ones_like(a, dtype=bool)
        k0, k1 = (int(start / w * small[0]), int((start + length) / w * small[0])) if axis == "x" else \
                 (int(start / h * small[1]), int((start + length) / h * small[1]))
        if axis == "x":
            mask[:, max(0, k0 - 1):k1 + 1] = False
        else:
            mask[max(0, k0 - 1):k1 + 1, :] = False
        av, bv = a[mask], b[mask]
        if av.size < 16 or av.std() < 2 or bv.std() < 2:
            continue                       # flat bands (or flat picture): can't tell
        valid += 1
        hits += float(np.corrcoef(av, bv)[0, 1]) > ECHO_CORR
    return hits / valid if valid else 0.0


def _boundary_share(frames: np.ndarray, start: int, length: int, axis: str) -> float:
    """Share of frames with a straight edge at both ends of the picture."""
    hits = 0
    for f in frames:
        g = f.astype(np.float32)
        grad = np.abs(np.diff(g, axis=1 if axis == "x" else 0))
        line = grad.mean(axis=0 if axis == "x" else 1)
        ok = True
        for edge in (start - 1, start + length - 1):
            if edge <= 4 or edge >= len(line) - 5:
                continue                   # a picture touching the frame edge has no boundary there
            beside = (line[edge - 4] + line[edge + 4]) / 2
            ok &= line[edge] >= 2 * max(beside, 0.5)
        hits += ok
    return hits / max(len(frames), 1)


def find_blurred_bars(frames: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the real picture in the frames' pixels, or None."""
    if frames.size == 0 or len(frames) < 3:
        return None
    n, h, w = frames.shape
    if w > WORK_WIDTH:
        s = WORK_WIDTH / w
        frames = np.array([cv2.resize(f, (WORK_WIDTH, int(round(h * s))), interpolation=cv2.INTER_AREA) for f in frames])
    else:
        s = 1.0
    _, hh, ww = frames.shape
    found = None
    for axis in ("x", "y"):
        profile = _profile(frames, axis)
        for start, length, ratio in _windows(profile, hh if axis == "x" else ww, axis):
            echo = _echo_share(frames, start, length, axis)
            boundary = _boundary_share(frames, start, length, axis) >= BOUNDARY_SHARE
            sharp = ratio >= SHARPNESS_RATIO
            if (sharp and (echo >= ECHO_SHARE or boundary)) or (echo >= ECHO_SHARE and boundary):
                score = 10 * echo + min(ratio, 50) / 10
                if found is None or score > found[0]:
                    found = (score, axis, start, length)
    if found is None:
        return None
    _, axis, start, length = found
    # Snap to the exact standard shape it matched, keeping its centre.
    other = hh if axis == "x" else ww
    size = ww if axis == "x" else hh
    aspect = min(STANDARD_ASPECTS, key=lambda a: abs((length / other if axis == "x" else other / length) - a))
    exact = aspect * other if axis == "x" else other / aspect
    centre = start + length / 2
    start = float(np.clip(centre - exact / 2, 0, size - exact))
    length = exact
    if axis == "x":
        return int(round(start / s)), 0, int(round(length / s)), h
    return 0, int(round(start / s)), w, int(round(length / s))
