"""Reframe engine: video in, segments out.

    analyze(path) → Analysis        (expensive, once per file, cacheable)
    plan(analysis, "1:1") → result  (cheap, per aspect ratio / style)

Output coordinates are normalized to the *full source frame* (0..1), so a
renderer only has to crop the source; picture-area handling is internal.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from dataclasses import dataclass, asdict, field

from .decode import decode
from .bodies import BodyDetector
from .lips import LIP_WIDTH, speakers_in_shot
from .persons import join_bodies
from .faces import FaceDetector
from .content import ContentBox, content_boxes
from .continuity import setups, transitions, unify
from .smoothing import Context as SmoothingContext, smooth
from .layout import Segment, plan_content, plan_shot, shot_alternatives
from .motion import estimate as estimate_motion
from .probe import Rect, VideoInfo, find_picture_area, probe
from .shots import Shot, detect_transitions
from .options import QUALITY, Options
from .overlays import TextBand, detect_text_bands
from .speakers import People, Turn, active_person, describe_tracks, group_people, load_turns, map_speakers
from .text import TextBox, TextDetector
from .tracks import Track, classify, mark_checked, track_shot

ENGINE_VERSION = "2.2.0-dev"
BODY_FPS = 2.5               # body detections per second (people check and body fallback)


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
    # Camera motion of the source between consecutive analysis frames (None at
    # the first frame of each shot); see motion.py.
    motion: list = field(default_factory=list)
    # Cuts, dissolves and fades between shots (shots meet at their middles).
    transitions: list = field(default_factory=list)
    # Who is talking, from mouth motion, during speech in shots with several people (lips.py).
    lip_turns: list = field(default_factory=list)


# Rough share of analysis time per stage, for progress reporting.
STAGES = [("probe", "Reading the video", 0.05), ("shots", "Finding cuts", 0.15), ("faces", "Finding faces", 0.22),
          ("motion", "Measuring camera motion", 0.03), ("tracks", "Tracking people", 0.05), ("people", "Checking who is a person", 0.35),
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
    shots, transitions = detect_transitions(cut_frames, info.duration)
    del cut_frames
    timings["shots"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("faces")
    q = QUALITY[quality]
    frames = decode(info, area, fps=q["fps"], max_width=q["width"], upscale_to=q["width"])
    faces = FaceDetector().detect(frames)
    # Blended frames inside a dissolve or fade show ghosts of both shots: no faces there.
    for tr in transitions:
        if tr.kind != "cut":
            for i in range(int(tr.start * frames.fps), min(len(faces), int(np.ceil(tr.end * frames.fps)) + 1)):
                faces[i] = []
    timings["faces"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("motion")
    shot_starts = [int(round(s.start * frames.fps)) for s in shots]
    motion = estimate_motion(frames, faces, shot_starts)
    timings["motion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stage("tracks")
    tracks = {s.index: track_shot(s, faces, frames.fps) for s in shots}
    timings["tracks"] = time.perf_counter() - t0

    # People check: a body for each face (about once a second), and whether
    # the face image changes over time.
    t0 = time.perf_counter()
    stage("people")
    every = max(1, int(round(frames.fps)))
    body_every = max(1, int(round(frames.fps / BODY_FPS)))
    bodies = BodyDetector().detect(frames, every=body_every)
    checked = list(range(0, len(frames.data), body_every))
    mark_checked(bodies, set(checked))
    for shot_tracks in tracks.values():
        classify(shot_tracks, frames, bodies)
    # People whose face is lost (looking down, turned away, backlit, blurred)
    # are followed by their body; faceless bodies can be people too.
    tracks = {s.index: join_bodies(s, tracks[s.index], bodies, checked, frames) for s in shots}
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
    # With several people in a shot, mouth motion says who is talking (lips.py).
    lip_turns = []
    if turns:
        for s in shots:
            people_here = [t for t in tracks[s.index] if t.is_person]
            if len(people_here) >= 2:
                lip_turns += speakers_in_shot(info, area, s, people_here, turns, max(q["width"], LIP_WIDTH))
    people = map_speakers(turns, shots, tracks, person_of,
                          [(lt.speaker, lt.shot, lt.track_id, lt.end - lt.start) for lt in lip_turns if lt.track_id is not None])
    timings["speakers"] = time.perf_counter() - t0

    stills = {}
    for s in shots:
        i = int(np.clip(round((s.start + s.end) / 2 * frames.fps), 0, len(frames.data) - 1))
        ok, jpg = cv2.imencode(".jpg", frames.data[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            stills[s.index] = jpg.tobytes()
    return Analysis(info, area, shots, tracks, turns, people, text, content, text_in_picture, quality, q["fps"], timings,
                    stills, motion, transitions, lip_turns)


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
    big = [[b for b in frame if b.h >= CAPTION_MIN_H and (b.x1 - b.x0) * pic_aspect / b.h >= CAPTION_MIN_ASPECT
            and not _is_watermark(analysis, b)] for frame in samples]
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


WATERMARK_SHARE = 0.5      # text at one fixed place and size in this share of sampled frames...
WATERMARK_SHOTS = 3        # ...across at least this many shots is a logo or watermark, not captions
WATERMARK_TOL = 0.015      # box edges within this (share of the picture) count as the same place


def _watermarks(analysis: "Analysis") -> list[tuple[float, float, float, float]]:
    """Logos and watermarks: text boxes that sit at the same place with the
    same size all through the video (captions change width with every line)."""
    cached = getattr(analysis, "_watermark_cache", None)
    if cached is not None:
        return cached
    every = max(1, int(round(analysis.fps)))
    sampled = [(i, frame) for i, frame in enumerate(analysis.text_in_picture) if i % every == 0]
    clusters: list[dict] = []
    for i, frame in sampled:
        for b in frame:
            box = (b.x0, b.y0, b.x1, b.y1)
            c = next((c for c in clusters if all(abs(u - v) <= WATERMARK_TOL for u, v in zip(c["box"], box))), None)
            if c is None:
                clusters.append({"box": box, "frames": {i}})
            else:
                c["frames"].add(i)
    shot_of = lambda i: next((s.index for s in analysis.shots if s.start <= i / analysis.fps < s.end), -1)  # noqa: E731
    marks = [c["box"] for c in clusters
             if len(c["frames"]) >= WATERMARK_SHARE * max(1, len(sampled))
             and len({shot_of(i) for i in c["frames"]}) >= WATERMARK_SHOTS]
    object.__setattr__(analysis, "_watermark_cache", marks)
    return marks


def _is_watermark(analysis: "Analysis", b) -> bool:
    return any(all(abs(u - v) <= 2 * WATERMARK_TOL for u, v in zip(m, (b.x0, b.y0, b.x1, b.y1)))
               for m in _watermarks(analysis))


TEXT_MOST_OF_SHOT = 0.7     # text on screen this share of a shot: keep it for the whole shot
TEXT_PAD_S = 0.6            # keep text in frame this long before it appears and after it goes
TEXT_JOIN_S = 2.0           # text gaps shorter than this are one stretch
MIN_PIECE_S = 1.5           # no framing piece shorter than this
TEXT_MIN_RUN = 3            # text must be seen in this many samples in a row (~seconds) to count


def _text_samples(analysis: "Analysis", shot: Shot) -> list[tuple[float, bool]]:
    """(time, caption-size text on screen) about once a second through the shot."""
    every = max(1, int(round(analysis.fps)))
    pic_aspect = analysis.area.w / analysis.area.h
    out = []
    for i, frame in enumerate(analysis.text_in_picture):
        t = i / analysis.fps
        if i % every or not (shot.start <= t < shot.end):
            continue
        out.append((t, any(b.h >= CAPTION_MIN_H and (b.x1 - b.x0) * pic_aspect / b.h >= CAPTION_MIN_ASPECT
                           and not _is_watermark(analysis, b) for b in frame)))
    return out


def shot_pieces(analysis: "Analysis", shot: Shot) -> list[Shot]:
    """A shot split where on-screen text (a name caption, a banner) comes and
    goes, so the framing keeps the text only while it is there. Text on screen
    most of the shot (running captions), or none: the whole shot, one piece."""
    if not getattr(analysis, "text_in_picture", None):
        return [shot]
    samples = _text_samples(analysis, shot)
    if not samples:
        return [shot]
    # A graphic stays up for seconds; a hit or two in a row is a pattern on a
    # shirt or a hand. One missed sample inside a run doesn't break it.
    on = [o for _, o in samples]
    filled = [on[k] or (0 < k < len(on) - 1 and on[k - 1] and on[k + 1]) for k in range(len(on))]
    keep = [False] * len(on)
    k = 0
    while k < len(on):
        if not filled[k]:
            k += 1
            continue
        j = k
        while j < len(on) and filled[j]:
            j += 1
        if sum(on[k:j]) >= TEXT_MIN_RUN:
            keep[k:j] = [True] * (j - k)
        k = j
    samples = [(t, kept) for (t, _), kept in zip(samples, keep)]
    share = float(np.mean([on for _, on in samples]))
    if share == 0 or share >= TEXT_MOST_OF_SHOT or shot.duration < 2 * MIN_PIECE_S:
        return [shot]
    spans: list[list[float]] = []
    for t, on in samples:
        if not on:
            continue
        a, b = max(shot.start, t - TEXT_PAD_S), min(shot.end, t + 1.0 + TEXT_PAD_S)
        if spans and a - spans[-1][1] < TEXT_JOIN_S:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    # Pieces: text spans and the stretches between them; a sliver joins the text next to it.
    edges = [shot.start]
    for a, b in spans:
        edges += [a, b]
    edges.append(shot.end)
    edges = sorted(set(round(e, 3) for e in edges))
    kept = [edges[0]]
    for e in edges[1:-1]:
        if e - kept[-1] >= MIN_PIECE_S and shot.end - e >= MIN_PIECE_S:
            kept.append(e)
    kept.append(edges[-1])
    if len(kept) <= 2:
        return [shot]
    return [Shot(shot.index, kept[k], kept[k + 1]) for k in range(len(kept) - 1)]


def _caption_band(analysis: "Analysis") -> tuple[float, float, float, float] | None:
    """Where captions appear across the whole video (lower half, caption-size
    text, frames that aren't screens)."""
    cached = getattr(analysis, "_caption_band_cache", "unset")
    if cached != "unset":
        return cached
    every = max(1, int(round(analysis.fps)))
    # Only shots showing a person: titles on slides and graphics sit elsewhere
    # and would stretch the band over the person's face.
    person_frames = set()
    for shot in analysis.shots:
        if any(t.is_person for t in analysis.tracks[shot.index]):
            person_frames.update(range(int(shot.start * analysis.fps), int(shot.end * analysis.fps) + 1))
    boxes = [b for i, frame in enumerate(analysis.text_in_picture)
             if i % every == 0 and len(frame) < 10 and i in person_frames
             for b in frame if b.h >= CAPTION_MIN_H and (b.y0 + b.y1) / 2 >= 0.5
             and (b.x1 - b.x0) * analysis.area.w / analysis.area.h / b.h >= CAPTION_MIN_ASPECT
             and not _is_watermark(analysis, b)]
    band = None
    if len(boxes) >= 3:
        band = (float(np.percentile([b.x0 for b in boxes], 5)), float(np.percentile([b.y0 for b in boxes], 5)),
                float(np.percentile([b.x1 for b in boxes], 95)), float(np.percentile([b.y1 for b in boxes], 95)))
    object.__setattr__(analysis, "_caption_band_cache", band)
    return band


def _with_transitions(segments: list[dict], cut_times: list[float], between: list | None = None) -> list[dict]:
    transitions(segments, cut_times)
    # A dissolve in the source: the two framings cross-fade with it, centred on its middle.
    for tr in between or []:
        if tr.kind == "dissolve":
            seg = next((x for x in segments if abs(x["start"] - tr.middle) < 1e-3), None)
            if seg is not None:
                seg["transition"] = {"type": "dissolve", "duration": round(tr.end - tr.start, 3)}
    return segments


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


CAPTION_MODES = ("keep", "replace")


def picture_captions(analysis: Analysis, shot: Shot):
    """Burned-in captions in the lower part of this shot's picture (x0, y0, x1, y1), or None."""
    _, captions = _shot_text(analysis, shot)
    if captions and (captions[1] + captions[3]) / 2 >= 0.5:
        return captions
    return None


def shot_options(analysis: Analysis, ratio: str, options: Options | None = None,
                 captions_mode: str = "keep") -> list[dict]:
    """Per shot: {"shot": Shot, "options": {name: [Segment]}, "captions": ...},
    where "rules" is the rule-based choice and the others are alternatives the
    AI director may pick instead.

    captions_mode "replace": in shots with burned-in captions in the lower part
    of the picture, every framing leaves them out entirely (Shotline adds new
    captions there); "captions" is then "replaced", or "kept" where no framing
    can leave them out (e.g. captions over the face)."""
    target = parse_ratio(ratio)
    area = analysis.area
    pic_aspect = area.w / area.h
    options = options or Options(quality=analysis.quality)
    # How far the camera may zoom in: plan output resolution + the user's zoom choice.
    min_crop_h = options.min_crop_height(area.h)
    out = []
    for s in (piece for shot in analysis.shots for piece in shot_pieces(analysis, shot)):
        shot_tracks = analysis.tracks[s.index]

        def speaking(t: float, s=s, shot_tracks=shot_tracks):
            # Talking over each other: two people's turns at once, both on screen.
            talking = {turn.speaker for turn in analysis.turns if turn.start <= t <= turn.end}
            if len(talking) >= 2:
                pids = {analysis.people.speaker_person[spk] for spk in talking if spk in analysis.people.speaker_person}
                on_screen = [tr for tr in shot_tracks if analysis.people.person_of.get((s.index, tr.track_id)) in pids]
                if len(on_screen) >= 2:
                    return tuple(on_screen[:2])
            # Seen talking (mouth motion) beats heard talking (voice-to-person mapping).
            for lt in getattr(analysis, "lip_turns", []):
                if lt.shot == s.index and lt.track_id is not None and lt.start <= t <= lt.end:
                    return next((tr for tr in shot_tracks if tr.track_id == lt.track_id), None)
            pid = active_person(analysis.turns, analysis.people, t)
            if pid is None:
                return None
            return next((tr for tr in shot_tracks if analysis.people.person_of.get((s.index, tr.track_id)) == pid), None)

        screen, captions = _shot_text(analysis, s)
        has_people = any(t.is_person for t in shot_tracks) and not screen  # faces in thumbnails aren't subjects
        strict = options.zoom != "tight"   # tight: the user's opt-in to close-ups leaving out part of the person
        people_tracks = shot_tracks if has_people else []
        band = picture_captions(analysis, s) if captions_mode == "replace" and has_people else None
        if band is not None:
            rules = plan_shot(s, shot_tracks, target, pic_aspect, analysis.fps, min_crop_h, speaking, None)
            opts = shot_alternatives(s, people_tracks, rules, target, pic_aspect, analysis.fps, min_crop_h, None,
                                     strict=strict, avoid=band)
            if opts is not None:
                out.append({"shot": s, "options": opts, "captions": "replaced"})
                continue
        if not has_people and abs(target / pic_aspect - 1) >= 0.03:
            rules = plan_content(s, analysis.content, analysis.fps, target, pic_aspect)
        else:
            rules = plan_shot(s, shot_tracks, target, pic_aspect, analysis.fps, min_crop_h, speaking, captions)
        opts = shot_alternatives(s, people_tracks, rules, target, pic_aspect, analysis.fps, min_crop_h, captions,
                                 strict=strict)
        out.append({"shot": s, "options": opts,
                    "captions": "kept" if picture_captions(analysis, s) is not None else None})
    return out


def _ranges(items: list[dict], state: str) -> list[list[float]]:
    """Merged [start, end] times of the shots whose captions are `state`."""
    out: list[list[float]] = []
    for item in items:
        if item.get("captions") != state:
            continue
        s = item["shot"]
        if out and abs(out[-1][1] - s.start) < 1e-3:
            out[-1][1] = round(s.end, 3)
        else:
            out.append([round(s.start, 3), round(s.end, 3)])
    return out


def plan(analysis: Analysis, ratio: str, keep_text: list[str] | None = None,
         options: Options | None = None, choices: dict[int, str] | None = None,
         director: dict | None = None, captions_mode: str = "keep") -> dict:
    """`keep_text`: ids of detected burned-in text (from result["text"]) the
    user chose to keep; everything else outside the picture is dropped.
    `choices`: shot index → option name (from the AI director); shots not
    listed use the rules. `director`: summary of how choices were made.
    `captions_mode`: "keep" or "replace" burned-in captions inside the picture
    (see shot_options)."""
    target = parse_ratio(ratio)
    area = analysis.area
    options = options or Options(quality=analysis.quality)
    min_crop_h = options.min_crop_height(area.h)
    items = shot_options(analysis, ratio, options, captions_mode)
    groups = setups(analysis, items)
    setup_of = {pos: g for g in groups for pos in g}
    picked_segs: list[list[Segment]] = []
    picked = []
    for pos, item in enumerate(items):
        s, opts = item["shot"], item["options"]
        # The director picks per setup (on one shot of it): apply that pick to every shot of the setup.
        chosen = [(p, (choices or {}).get(items[p]["shot"].index)) for p in setup_of[pos]]
        rep, name = next(((p, n) for p, n in chosen if n), (pos, "rules"))
        if name not in opts:
            # This shot lacks that option (near-duplicates are merged per shot):
            # take its option of the same layout and closest size.
            want = items[rep]["options"].get(name, [None])[0]
            same = [(n, segs) for n, segs in opts.items()
                    if want is not None and segs[0].layout == want.layout and len(segs) == 1]
            name = min(same, key=lambda ns: abs(ns[1][0].info.get("crop_h", 1) - want.info.get("crop_h", 1)))[0] \
                if same else "rules"
        picked_segs.append(opts[name])
        picked.append({"shot": s.index, "start": round(s.start, 3), "end": round(s.end, 3), "choice": name,
                       "options": list(opts), "setup": items[setup_of[pos][0]]["shot"].index})
    # One framing per setup: no resizing at the jump cuts of a talking head.
    unify(groups, picked_segs, target, area.w / area.h)
    # No blink of another layout between cuts (a short shot over a blurred
    # background between two crops, or the other way round): see smoothing.
    if os.environ.get("REFRAME_SMOOTHING", "on") != "off":   # "off": compare without it (eval)
        smooth(SmoothingContext(analysis, target, area.w / area.h, lambda s: _shot_text(analysis, s),
                                lambda s: picture_captions(analysis, s)), items, picked_segs, picked)
    segments = [seg for segs in picked_segs for seg in segs]
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
        "segments": _with_transitions([_to_frame(s, analysis.info, area) for s in segments],
                                      [s.start for s in analysis.shots[1:]], getattr(analysis, "transitions", [])),
        "transitions": [{"type": t.kind, "start": round(t.start, 3), "end": round(t.end, 3)}
                        for t in getattr(analysis, "transitions", []) if t.kind != "cut"],
        "shots": picked,
        # Burned-in captions inside the picture: whether there are any, and with
        # "replace", where they were left out (the backend adds Shotline captions
        # for exactly those times) and where they had to stay.
        "pictureCaptions": any(picture_captions(analysis, s) is not None for s in analysis.shots),
        "captions": {"mode": captions_mode, "replaced": _ranges(items, "replaced"), "kept": _ranges(items, "kept")},
        "director": director or {"used": False},
        # Burned-in text found outside the picture: the editor asks whether to
        # keep it (default: drop; suggest Shotline captions when not keyable).
        "text": [{"id": t.id, "kind": t.kind, "region": t.region, "description": t.describe(),
                  "onScreen": t.active, "canKeep": t.keyable} for t in analysis.text],
        "overlays": [_place_text(t, analysis.info, target) for t in analysis.text
                     if keep_text and t.id in keep_text and t.keyable],
    }
