"""Reframe engine: video in, segments out.

    analyze(path) → Analysis        (expensive, once per file, cacheable)
    plan(analysis, "1:1") → result  (cheap, per aspect ratio / style)

Output coordinates are normalized to the *full source frame* (0..1), so a
renderer only has to crop the source; picture-area handling is internal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from .decode import decode
from .faces import FaceDetector
from .layout import Segment, plan_shot
from .probe import Rect, VideoInfo, find_picture_area, probe
from .shots import Shot, detect_shots
from .tracks import Track, track_shot

ENGINE_VERSION = "2.0.0-dev"
ANALYSIS_FPS = 5.0
MAX_UPSCALE = 2.5          # never magnify the source more than this
OUTPUT_HEIGHT = 1080


@dataclass
class Analysis:
    info: VideoInfo
    area: Rect
    shots: list[Shot]
    tracks: dict[int, list[Track]]
    timings: dict[str, float]


def analyze(path: str) -> Analysis:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    info = probe(path)
    area = find_picture_area(info)
    timings["probe"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cut_frames = decode(info, area, fps=info.fps, max_width=320)
    shots = detect_shots(cut_frames, info.duration)
    timings["shots"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    frames = decode(info, area, fps=ANALYSIS_FPS, max_width=640, upscale_to=640)
    faces = FaceDetector().detect(frames)
    timings["faces"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracks = {s.index: track_shot(s, faces, frames.fps) for s in shots}
    timings["tracks"] = time.perf_counter() - t0
    return Analysis(info, area, shots, tracks, timings)


def parse_ratio(ratio: str) -> float:
    a, b = ratio.split(":")
    return float(a) / float(b)


def _to_frame(seg: Segment, info: VideoInfo, area: Rect) -> dict:
    zones = []
    for z in seg.zones:
        kfs = [{
            "t": round(k.t, 3),
            "x": round((area.x + k.x * area.w) / info.width, 5),
            "y": round((area.y + k.y * area.h) / info.height, 5),
            "w": round(k.w * area.w / info.width, 5),
            "h": round(k.h * area.h / info.height, 5),
        } for k in z.keyframes]
        zones.append({"canvas": dict(zip(("left", "top", "width", "height"), z.canvas)), "subject": z.subject, "keyframes": kfs})
    return {"start": round(seg.start, 3), "end": round(seg.end, 3), "layout": seg.layout, "reason": seg.reason, "zones": zones}


def plan(analysis: Analysis, ratio: str) -> dict:
    target = parse_ratio(ratio)
    area = analysis.area
    pic_aspect = area.w / area.h
    # Resolution limit: don't zoom so far that the crop is magnified past MAX_UPSCALE.
    min_crop_h = min(1.0, max(0.35, OUTPUT_HEIGHT / (area.h * MAX_UPSCALE)))
    segments = [
        plan_shot(s, analysis.tracks[s.index], target, pic_aspect, ANALYSIS_FPS, min_crop_h)
        for s in analysis.shots
    ]
    return {
        "engine": ENGINE_VERSION,
        "source": {"width": analysis.info.width, "height": analysis.info.height, "duration": analysis.info.duration,
                   "fps": analysis.info.fps, "picture": asdict(area)},
        "target": ratio,
        "notes": (["Low-resolution source: no zooming in, to keep it sharp."] if min_crop_h >= 1.0 else []),
        "segments": [_to_frame(s, analysis.info, area) for s in segments],
    }
