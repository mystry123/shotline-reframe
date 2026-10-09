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

Gradual transitions (detect_transitions):
- fades through black (or white): contrast falls steadily to near zero and
  rises again; the shot boundary is the darkest frame;
- dissolves: the frame in the middle of the window is the average of the
  frames at its ends (the blend test), and the colours either side differ
  (a different shot). A camera pan changes the picture as much but fails the
  blend test, so it isn't taken for a dissolve.
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


# ---------------------------------------------------------------------------
# Gradual transitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    kind: str       # "cut", "dissolve" or "fade"
    start: float    # seconds
    end: float      # seconds (== start for a cut)
    confidence: float = 1.0

    @property
    def middle(self) -> float:
        return (self.start + self.end) / 2


FADE_DARK = 20.0          # mean luma below this (or above 255 - this) with...
FADE_FLAT = 8.0           # ...contrast (luma std) below this = a black or white frame
FADE_MIN_FRAMES = 4       # a fade takes at least this long on one side
FADE_STEP = 0.2           # contrast changes at least this much per frame while fading
DISSOLVE_LENGTHS_S = (0.32, 0.5, 0.75, 1.0, 1.5, 2.0)
DISSOLVE_BLEND = 0.35     # blend error below this (0 = a perfect mix of the two ends)
DISSOLVE_CHANGE = 0.30    # colour histogram distance between the two ends above this
DISSOLVE_MIN_DIFF = 12.0  # mean |difference| of the two end frames above this (0-255)
DISSOLVE_STILL_CELL = 6.0  # an 8x8 cell changing less than this (0-255) is "untouched"...
DISSOLVE_STILL_SHARE = 0.08  # ...and more than this share of untouched cells means no dissolve
DISSOLVE_RATE = 1.4       # the picture changes this many times faster inside than just around it


def _signals(frames: Frames):
    luma_mean, luma_std, hists, smalls = [], [], [], []
    for frame in frames.data:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma_mean.append(float(gray.mean()))
        luma_std.append(float(gray.std()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel()
        hists.append(h / max(h.sum(), 1))
        smalls.append(cv2.resize(gray, (80, max(8, int(80 * gray.shape[0] / gray.shape[1]))),
                                 interpolation=cv2.INTER_AREA).astype(np.float32))
    return np.array(luma_mean), np.array(luma_std), np.array(hists), smalls


def _fades(mean: np.ndarray, std: np.ndarray, fps: float) -> list[Transition]:
    n = len(std)
    flat = (std < FADE_FLAT) & ((mean < FADE_DARK) | (mean > 255 - FADE_DARK))
    out = []
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        a = i
        while i < n and flat[i]:
            i += 1
        b = i - 1                                   # black (or white) frames a..b
        if a == 0 or b == n - 1:
            continue                                # fade in from / out to the video's own start or end
        s = a
        while s > 0 and a - s < int(2.5 * fps) and std[s - 1] >= std[s] + FADE_STEP:
            s -= 1                                  # contrast was falling into the black
        e = b
        while e < n - 1 and e - b < int(2.5 * fps) and std[e + 1] >= std[e] + FADE_STEP:
            e += 1                                  # and rises out of it
        if a - s >= FADE_MIN_FRAMES or e - b >= FADE_MIN_FRAMES:
            out.append(Transition("fade", s / fps, (e + 1) / fps))
    return out


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(p * q)))))


def _dissolves(hists, smalls, fps: float, cuts: set[int], busy: list[tuple[int, int]]) -> list[Transition]:
    n = len(smalls)
    found: list[tuple[float, int, int]] = []          # (blend error, a, b)
    for half in sorted({max(2, int(round(L * fps / 2))) for L in DISSOLVE_LENGTHS_S}):
        for c in range(half, n - half, max(1, half // 3)):
            a, b = c - half, c + half
            if any(a <= k <= b for k in cuts) or any(a <= y and x <= b for x, y in busy):
                continue
            diff = float(np.abs(smalls[a] - smalls[b]).mean())
            if diff < DISSOLVE_MIN_DIFF or _bhattacharyya(hists[a], hists[b]) < DISSOLVE_CHANGE:
                continue
            blend = float(np.abs(smalls[c] - 0.5 * (smalls[a] + smalls[b])).mean()) / diff
            # The quarter points must sit on the same mix too (not just the middle).
            q1, q3 = (a + c) // 2, (c + b) // 2
            blend = max(blend,
                        float(np.abs(smalls[q1] - (0.75 * smalls[a] + 0.25 * smalls[b])).mean()) / diff,
                        float(np.abs(smalls[q3] - (0.25 * smalls[a] + 0.75 * smalls[b])).mean()) / diff)
            if blend >= DISSOLVE_BLEND:
                continue
            # A dissolve changes the picture faster than either shot does on
            # its own (a slowly animating picture changes evenly throughout).
            inside = np.mean([np.abs(smalls[k] - smalls[k - 1]).mean() for k in range(a + 1, b + 1)])
            around = [np.abs(smalls[k] - smalls[k - 1]).mean() for k in
                      list(range(max(1, a - half), a)) + list(range(b + 1, min(n, b + half + 1)))]
            if around and inside < DISSOLVE_RATE * float(np.mean(around)):
                continue
            # A dissolve changes the whole picture; something passing in front
            # of the lens (a hand) leaves part of the scene untouched.
            d = np.abs(smalls[a] - smalls[b])
            gh, gw = d.shape[0] // 8, d.shape[1] // 8
            cells = d[:gh * 8, :gw * 8].reshape(gh, 8, gw, 8).mean(axis=(1, 3))
            if float((cells < DISSOLVE_STILL_CELL).mean()) > DISSOLVE_STILL_SHARE:
                continue
            found.append((blend, a, b))
    picked: list[tuple[float, int, int]] = []
    for blend, a, b in sorted(found):               # best first; overlapping candidates are the same dissolve
        if any(not (b < pa or a > pb) for _, pa, pb in picked):
            continue
        picked.append((blend, a, b))
    # Pieces that touch are one dissolve (short windows fit a linear mix as well as the whole).
    merged: list[list] = []
    for blend, a, b in sorted(picked, key=lambda x: x[1]):
        if merged and a - merged[-1][2] <= max(2, int(0.12 * fps)):
            merged[-1][2] = max(merged[-1][2], b)
            merged[-1][0] = min(merged[-1][0], blend)
        else:
            merged.append([blend, a, b])
    return [Transition("dissolve", a / fps, b / fps, round(1 - blend, 3)) for blend, a, b in merged]


def detect_transitions(frames: Frames, duration: float, min_shot: float = 0.5) -> tuple[list[Shot], list[Transition]]:
    """Shots and the transitions between them: hard cuts, fades, dissolves.
    Shots meet at a gradual transition's middle (a fade's darkest frame)."""
    fps = frames.fps
    cut_shots = detect_shots(frames, duration, min_shot=min_shot)
    cut_times = [s.start for s in cut_shots[1:]]
    mean, std, hists, smalls = _signals(frames)
    fades = _fades(mean, std, fps)
    # A hard "cut" found inside a fade (into or out of black) is that fade.
    cut_times = [t for t in cut_times if not any(f.start - 0.1 <= t <= f.end + 0.1 for f in fades)]
    cut_frames = {int(round(t * fps)) for t in cut_times}
    busy = [(int(f.start * fps), int(f.end * fps)) for f in fades]
    dissolves = _dissolves(hists, smalls, fps, cut_frames, busy)
    transitions = sorted([Transition("cut", t, t) for t in cut_times] + fades + dissolves, key=lambda t: t.start)

    times = [0.0]
    for tr in transitions:
        t = tr.middle
        if t - times[-1] >= min_shot and duration - t >= min_shot:
            times.append(t)
    times.append(duration)
    shots = [Shot(k, times[k], times[k + 1]) for k in range(len(times) - 1)]
    return shots, transitions
