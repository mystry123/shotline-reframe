"""Reframe engine: video in, segments out.

    analyze(path) → Analysis        (expensive, once per file, cacheable)
    plan(analysis, "1:1") → result  (cheap, per aspect ratio / style)

Output coordinates are normalized to the *full source frame* (0..1), so a
renderer only has to crop the source; picture-area handling is internal.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from dataclasses import dataclass, asdict, field

from .decode import decode
from .bodies import BodyDetector
from .faces import FaceDetector
from .content import ContentBox, content_boxes
from .layout import Segment, plan_content, plan_shot, shot_alternatives
from .probe import Rect, VideoInfo, find_picture_area, probe
from .shots import Shot, detect_shots
from .options import QUALITY, Options
from .overlays import TextBand, detect_text_bands
from .speakers import People, Turn, active_person, describe_tracks, group_people, load_turns, map_speakers
from .text import TextBox, TextDetector
from .tracks import Track, classify, mark_checked, track_shot

ENGINE_VERSION = "2.1.0-dev"


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
    text_in_picture: list[list[TextBox]]
    quality: str
    fps: float
    timings: dict[str, float]
    # One still (JPEG of the picture area) from the middle of each shot, for
    # the AI director's contact sheets.
    stills: dict[int, bytes] = field(default_factory=dict)


# Rough share of analysis time per stage, for progress reporting.
STAGES = [("probe", "Reading the video", 0.05), ("shots", "Finding cuts", 0.15), ("faces", "Finding faces", 0.25),
          ("tracks", "Tracking people", 0.05), ("people", "Checking who is a person", 0.35),
          ("content", "Finding the content and text", 0.1), ("speakers", "Matching voices", 0.05)]


def analyze(path: str, words: list[dict] | None = None, quality: str = "standard",
            on_stage=None) -> Analysis:
    """`on_stage(key, label, progress)` is called as each stage starts
    (progress 0..1), for job status reporting."""
    done = 0.0

    def stage(key: str) -> None:
        nonlocal done
        for k, label, share in STAGES:
            if k == key:
                if on_stage:
                    on_stage(k, label, round(done, 3))
                done += share
                return
    """`words`: the transcript's words with speaker labels (ElevenLabs shape),
    when there is one; without it, layouts are decided from the picture only."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    stage("probe")
    info = probe(path)
    area = find_picture_area(info)
    text = detect_text_bands(info, area)
    timings["probe"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("shots")
    cut_frames = decode(info, area, fps=info.fps, max_width=320)
    shots = detect_shots(cut_frames, info.duration)
    timings["shots"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("faces")
    q = QUALITY[quality]
    frames = decode(info, area, fps=q["fps"], max_width=q["width"], upscale_to=q["width"])
    faces = FaceDetector().detect(frames)
    timings["faces"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("tracks")
    tracks = {s.index: track_shot(s, faces, frames.fps) for s in shots}
    timings["tracks"] = time.perf_counter() - t0

    # People check: a body for each face (about once a second), and whether
    # the face image changes over time.
    t0 = time.perf_counter()
    stage("people")
    every = max(1, int(round(frames.fps)))
    bodies = BodyDetector().detect(frames, every=every)
    mark_checked(bodies, set(range(0, len(frames.data), every)))
    for shot_tracks in tracks.values():
        classify(shot_tracks, frames, bodies)
    timings["people"] = time.perf_counter() - t0

    # Where the content is, for shots without people.
    t0 = time.perf_counter()
    stage("content")
    content = content_boxes(frames)
    text_in_picture = TextDetector().detect(frames, every=every)
    timings["content"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("speakers")
    describe_tracks(tracks, frames)
    person_of = group_people(tracks)
    turns = load_turns(words or [])
    people = map_speakers(turns, shots, tracks, person_of)
    timings["speakers"] = time.perf_counter() - t0

    stills = {}
    for s in shots:
        i = int(np.clip(round((s.start + s.end) / 2 * frames.fps), 0, len(frames.data) - 1))
        ok, jpg = cv2.imencode(".jpg", frames.data[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            stills[s.index] = jpg.tobytes()
    return Analysis(info, area, shots, tracks, turns, people, text, content, text_in_picture, quality, q["fps"], timings, stills)


SCREEN_MIN_BOXES = 15          # a frame with this much small text is a screen (page, app, recording)
CAPTION_MIN_H = 0.02           # text at least this tall (share of picture height) is caption/title size
CAPTION_MIN_ASPECT = 1.5      # width / height in pixels: text runs sideways
CAPTION_MIN_SHARE = 0.2        # ...and on screen in at least this share of the shot's samples


def _shot_text(analysis: "Analysis", shot: Shot) -> tuple[bool, tuple[float, float, float, float] | None]:
    """(is the shot a screen, box of captions/titles that must stay in frame)."""
    samples = [boxes for i, boxes in enumerate(analysis.text_in_picture)
               if shot.start <= i / analysis.fps < shot.end and i % max(1, int(round(analysis.fps))) == 0]
    if not samples:
        return False, None
    counts = [len(b) for b in samples]
    small = [b for frame in samples for b in frame if b.h < CAPTION_MIN_H]
    screen = float(np.median(counts)) >= SCREEN_MIN_BOXES or (float(np.median(counts)) >= 10 and len(small) > 0.6 * sum(counts))
    # Caption-size text, and shaped like text (a word or line is wider than
    # it is tall; square blobs are beards, logos and patterns).
    pic_aspect = analysis.area.w / analysis.area.h
    big = [[b for b in frame if b.h >= CAPTION_MIN_H and (b.x1 - b.x0) * pic_aspect / b.h >= CAPTION_MIN_ASPECT]
           for frame in samples]
    with_text = [b for b in big if b]
    if len(with_text) < CAPTION_MIN_SHARE * len(samples):
        return screen, None
    allb = [b for frame in with_text for b in frame]
    x0 = float(np.percentile([b.x0 for b in allb], 5)); y0 = float(np.percentile([b.y0 for b in allb], 5))
    x1 = float(np.percentile([b.x1 for b in allb], 95)); y1 = float(np.percentile([b.y1 for b in allb], 95))
    # Text is sampled about once a second, so a short shot may catch one line
    # of a two-line caption. Captions stay in one band through a video: a
    # shot with captions in the lower half reserves the whole band.
    if (y0 + y1) / 2 >= 0.5:
        band = _caption_band(analysis)
        if band:
            x0, y0, x1, y1 = min(x0, band[0]), min(y0, band[1]), max(x1, band[2]), max(y1, band[3])
    return screen, (x0, y0, x1, y1)


def _caption_band(analysis: "Analysis") -> tuple[float, float, float, float] | None:
    """Where captions appear across the whole video (lower half, caption-size
    text, frames that aren't screens)."""
    cached = getattr(analysis, "_caption_band_cache", "unset")
    if cached != "unset":
        return cached
    every = max(1, int(round(analysis.fps)))
    boxes = [b for i, frame in enumerate(analysis.text_in_picture) if i % every == 0 and len(frame) < 10
             for b in frame if b.h >= CAPTION_MIN_H and (b.y0 + b.y1) / 2 >= 0.5
             and (b.x1 - b.x0) * analysis.area.w / analysis.area.h / b.h >= CAPTION_MIN_ASPECT]
    band = None
    if len(boxes) >= 3:
        band = (float(np.percentile([b.x0 for b in boxes], 5)), float(np.percentile([b.y0 for b in boxes], 5)),
                float(np.percentile([b.x1 for b in boxes], 95)), float(np.percentile([b.y1 for b in boxes], 95)))
    object.__setattr__(analysis, "_caption_band_cache", band)
    return band


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


def shot_options(analysis: Analysis, ratio: str, options: Options | None = None) -> list[dict]:
    """Per shot: {"shot": Shot, "options": {name: [Segment]}}, where "rules" is
    the rule-based choice and the others are alternatives the AI director may
    pick instead."""
    target = parse_ratio(ratio)
    area = analysis.area
    pic_aspect = area.w / area.h
    options = options or Options(quality=analysis.quality)
    # How far the camera may zoom in: plan output resolution + the user's zoom choice.
    min_crop_h = options.min_crop_height(area.h)
    out = []
    for s in analysis.shots:
        shot_tracks = analysis.tracks[s.index]

        def speaking(t: float, s=s, shot_tracks=shot_tracks):
            pid = active_person(analysis.turns, analysis.people, t)
            if pid is None:
                return None
            return next((tr for tr in shot_tracks if analysis.people.person_of.get((s.index, tr.track_id)) == pid), None)

        screen, captions = _shot_text(analysis, s)
        has_people = any(t.is_person for t in shot_tracks) and not screen  # faces in thumbnails aren't subjects
        if not has_people and abs(target / pic_aspect - 1) >= 0.03:
            rules = plan_content(s, analysis.content, analysis.fps, target, pic_aspect)
        else:
            rules = plan_shot(s, shot_tracks, target, pic_aspect, analysis.fps, min_crop_h, speaking, captions)
        people_tracks = shot_tracks if has_people else []
        out.append({"shot": s, "options": shot_alternatives(s, people_tracks, rules, target, pic_aspect,
                                                            analysis.fps, min_crop_h, captions)})
    return out


def plan(analysis: Analysis, ratio: str, keep_text: list[str] | None = None,
         options: Options | None = None, choices: dict[int, str] | None = None,
         director: dict | None = None) -> dict:
    """`keep_text`: ids of detected burned-in text (from result["text"]) the
    user chose to keep; everything else outside the picture is dropped.
    `choices`: shot index → option name (from the AI director); shots not
    listed use the rules. `director`: summary of how choices were made."""
    target = parse_ratio(ratio)
    area = analysis.area
    options = options or Options(quality=analysis.quality)
    min_crop_h = options.min_crop_height(area.h)
    segments = []
    picked = []
    for item in shot_options(analysis, ratio, options):
        s, opts = item["shot"], item["options"]
        name = (choices or {}).get(s.index, "rules")
        if name not in opts:
            name = "rules"
        segments.extend(opts[name])
        picked.append({"shot": s.index, "start": round(s.start, 3), "end": round(s.end, 3), "choice": name,
                       "options": list(opts)})
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
        "shots": picked,
        "director": director or {"used": False},
        # Burned-in text found outside the picture: the editor asks whether to
        # keep it (default: drop; suggest Shotline captions when not keyable).
        "text": [{"id": t.id, "kind": t.kind, "region": t.region, "description": t.describe(),
                  "onScreen": t.active, "canKeep": t.keyable} for t in analysis.text],
        "overlays": [_place_text(t, analysis.info, target) for t in analysis.text
                     if keep_text and t.id in keep_text and t.keyable],
    }
