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
from .bodies import BodyDetector
from .faces import FaceDetector
from .content import ContentBox, content_boxes
from .layout import Segment, plan_content, plan_shot
from .probe import Rect, VideoInfo, find_picture_area, probe
from .shots import Shot, detect_shots
from .options import QUALITY, Options
from .overlays import TextBand, detect_text_bands
from .speakers import People, Turn, active_person, describe_tracks, group_people, load_turns, map_speakers
from .tracks import Track, classify, mark_checked, track_shot

ENGINE_VERSION = "2.0.0-dev"


@dataclass
class Analysis:
    info: VideoInfo
    area: Rect
    shots: list[Shot]
    tracks: dict[int, list[Track]]
    turns: list[Turn]
    people: People
    text: list[TextBand]
    content: list[ContentBox | None]
    quality: str
    fps: float
    timings: dict[str, float]


def analyze(path: str, words: list[dict] | None = None, quality: str = "standard") -> Analysis:
    """`words`: the transcript's words with speaker labels (ElevenLabs shape),
    when there is one; without it, layouts are decided from the picture only."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    info = probe(path)
    area = find_picture_area(info)
    text = detect_text_bands(info, area)
    timings["probe"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cut_frames = decode(info, area, fps=info.fps, max_width=320)
    shots = detect_shots(cut_frames, info.duration)
    timings["shots"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    q = QUALITY[quality]
    frames = decode(info, area, fps=q["fps"], max_width=q["width"], upscale_to=q["width"])
    faces = FaceDetector().detect(frames)
    timings["faces"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracks = {s.index: track_shot(s, faces, frames.fps) for s in shots}
    timings["tracks"] = time.perf_counter() - t0

    # People check: a body for each face (about once a second), and whether
    # the face image changes over time.
    t0 = time.perf_counter()
    every = max(1, int(round(frames.fps)))
    bodies = BodyDetector().detect(frames, every=every)
    mark_checked(bodies, set(range(0, len(frames.data), every)))
    for shot_tracks in tracks.values():
        classify(shot_tracks, frames, bodies)
    timings["people"] = time.perf_counter() - t0

    # Where the content is, for shots without people.
    t0 = time.perf_counter()
    content = content_boxes(frames)
    timings["content"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    describe_tracks(tracks, frames)
    person_of = group_people(tracks)
    turns = load_turns(words or [])
    people = map_speakers(turns, shots, tracks, person_of)
    timings["speakers"] = time.perf_counter() - t0
    return Analysis(info, area, shots, tracks, turns, people, text, content, quality, q["fps"], timings)


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


# Where kept text goes in the new frame (share of output height): titles near
# the top, captions in the lower third but above the area TikTok/Reels/Shorts
# cover with their buttons and descriptions.
TEXT_PLACEMENT = {"title": {"center": 0.09, "max_h": 0.12}, "captions": {"center": 0.72, "max_h": 0.28}}
TEXT_MAX_W = 0.9


def _place_text(band: TextBand, info: VideoInfo, target: float) -> dict:
    spec = TEXT_PLACEMENT[band.kind]
    aspect = band.rect.w / band.rect.h                      # source pixels
    h = spec["max_h"]
    w = h * aspect / target                                 # output-normalized (w/h are fractions of each axis)
    if w > TEXT_MAX_W:
        w = TEXT_MAX_W
        h = w * target / aspect
    return {
        "id": band.id,
        "source": {"x": round(band.rect.x / info.width, 5), "y": round(band.rect.y / info.height, 5),
                   "w": round(band.rect.w / info.width, 5), "h": round(band.rect.h / info.height, 5)},
        "placement": {"left": round((1 - w) / 2, 4), "top": round(spec["center"] - h / 2, 4),
                      "width": round(w, 4), "height": round(h, 4)},
        "key": "dark",   # make the dark background transparent
    }


def plan(analysis: Analysis, ratio: str, keep_text: list[str] | None = None,
         options: Options | None = None) -> dict:
    """`keep_text`: ids of detected burned-in text (from result["text"]) the
    user chose to keep; everything else outside the picture is dropped."""
    target = parse_ratio(ratio)
    area = analysis.area
    pic_aspect = area.w / area.h
    options = options or Options(quality=analysis.quality)
    # How far the camera may zoom in: plan output resolution + the user's zoom choice.
    min_crop_h = options.min_crop_height(area.h)
    segments = []
    for s in analysis.shots:
        shot_tracks = analysis.tracks[s.index]

        def speaking(t: float, s=s, shot_tracks=shot_tracks):
            pid = active_person(analysis.turns, analysis.people, t)
            if pid is None:
                return None
            return next((tr for tr in shot_tracks if analysis.people.person_of.get((s.index, tr.track_id)) == pid), None)

        has_people = any(t.is_person for t in shot_tracks)
        if not has_people and abs(target / pic_aspect - 1) >= 0.03:
            segments.extend(plan_content(s, analysis.content, analysis.fps, target, pic_aspect))
        else:
            segments.extend(plan_shot(s, shot_tracks, target, pic_aspect, analysis.fps, min_crop_h, speaking))
    return {
        "engine": ENGINE_VERSION,
        "source": {"width": analysis.info.width, "height": analysis.info.height, "duration": analysis.info.duration,
                   "fps": analysis.info.fps, "picture": asdict(area)},
        "target": ratio,
        "settings": {"quality": analysis.quality, "outputHeight": options.output_height, "zoom": options.zoom},
        "notes": ([f"Low-resolution source for {options.output_height}p output: no zooming in, to keep it sharp. "
                   "Choose tighter framing to zoom in anyway."] if min_crop_h >= 1.0 else []),
        "speakers": {spk: {"person": pid, "confidence": round(analysis.people.confidence.get(spk, 0), 2)}
                     for spk, pid in analysis.people.speaker_person.items()},
        "segments": [_to_frame(s, analysis.info, area) for s in segments],
        # Burned-in text found outside the picture: the editor asks whether to
        # keep it (default: drop; suggest Shotline captions when not keyable).
        "text": [{"id": t.id, "kind": t.kind, "region": t.region, "description": t.describe(),
                  "onScreen": t.active, "canKeep": t.keyable} for t in analysis.text],
        "overlays": [_place_text(t, analysis.info, target) for t in analysis.text
                     if keep_text and t.id in keep_text and t.keyable],
    }
