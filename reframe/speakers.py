"""Who is speaking, and which on-screen person that is.

1. Speech turns from the transcript's word-level speaker labels (ElevenLabs
   diarization). Backchannels ("yeah", "mm"), under 1 s or 3 words or fewer,
   are ignored: they never move the camera.
2. People across shots. The same person appears in many shots (cuts between
   cameras), each time as a new track. Tracks are grouped into people by
   appearance below and around the face (clothing colour), which is stable
   within an interview and needs no face-recognition model.
3. Voice → person. Every second a speaker talks while exactly one person is
   on screen is a vote for that pairing; the best one-to-one assignment over
   the whole video wins (Hungarian). Each pairing gets a confidence; low
   confidence means "unknown", and layouts then avoid guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .decode import Frames
from .shots import Shot
from .tracks import Track

MIN_TURN_S = 1.0
MIN_TURN_WORDS = 4
MERGE_GAP_S = 0.5
SAME_PERSON = 0.5           # Bhattacharyya distance below which two tracks are one person
ATTACH_MAX = 0.7            # a leftover track joins its nearest person below this...
ATTACH_MARGIN = 0.08        # ...if that person is clearly nearer than the next one
MIN_CONFIDENCE = 0.6


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    words: int


@dataclass
class People:
    """person id per (shot, track id), and speaker → person mapping."""
    person_of: dict[tuple[int, int], int] = field(default_factory=dict)
    speaker_person: dict[str, int] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)


def load_turns(words: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    for w in words:
        if w.get("type", "word") != "word" or not w.get("speaker_id"):
            continue
        s = w["speaker_id"]
        if turns and turns[-1].speaker == s and w["start"] - turns[-1].end <= MERGE_GAP_S:
            turns[-1].end = w["end"]
            turns[-1].words += 1
        else:
            turns.append(Turn(s, w["start"], w["end"], 1))
    return [t for t in turns if t.end - t.start >= MIN_TURN_S and t.words >= MIN_TURN_WORDS]


def describe_tracks(tracks: dict[int, list[Track]], frames: Frames) -> None:
    """Appearance descriptor per real-person track: HSV colour histogram of the
    head-and-torso region below the face, averaged over the track."""
    for shot_tracks in tracks.values():
        for t in shot_tracks:
            if not t.is_person:
                continue
            hists = []
            for d in t.dets[:: max(1, len(t.dets) // 8)]:
                img = frames.data[d.frame]
                H, W = img.shape[:2]
                x0 = int(max(0, (d.cx - d.w) * W))
                x1 = int(min(W, (d.cx + d.w) * W))
                y0 = int(min(H - 1, (d.y + d.h * 1.2) * H))
                y1 = int(min(H, (d.y + d.h * 3.0) * H))
                if y1 - y0 < 4 or x1 - x0 < 4:
                    continue
                hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
                hists.append(cv2.normalize(h, None).flatten())
            t.appearance = np.mean(hists, axis=0).astype(np.float32) if hists else None  # type: ignore[attr-defined]


def group_people(tracks: dict[int, list[Track]]) -> dict[tuple[int, int], int]:
    """Greedy agglomerative grouping of person tracks across shots by appearance.
    Two tracks in the same shot are never the same person."""
    items = [(s, t) for s, ts in tracks.items() for t in ts if t.is_person and getattr(t, "appearance", None) is not None]
    people: list[list[tuple[int, Track]]] = []
    for shot, t in sorted(items, key=lambda it: -len(it[1].dets)):
        best, best_d = None, SAME_PERSON
        for p in people:
            if any(s == shot for s, _ in p):
                continue
            d = np.mean([cv2.compareHist(t.appearance, o.appearance, cv2.HISTCMP_BHATTACHARYYA) for _, o in p])  # type: ignore[attr-defined]
            if d < best_d:
                best, best_d = p, d
        (best if best is not None else people.append([]) or people[-1]).append((shot, t))  # type: ignore[union-attr]

    # Second pass: a person seen once from an unusual angle or light (e.g. from
    # behind in a wide shot) ends up alone; attach it to the clearly nearest
    # person it can belong to.
    for lone in [p for p in people if len(p) == 1]:
        shot, t = lone[0]
        dists = []
        for p in people:
            if p is lone or not p or any(s == shot for s, _ in p):
                continue
            dists.append((np.mean([cv2.compareHist(t.appearance, o.appearance, cv2.HISTCMP_BHATTACHARYYA) for _, o in p]), id(p), p))  # type: ignore[attr-defined]
        dists.sort(key=lambda d: d[0])
        if dists and dists[0][0] < ATTACH_MAX and (len(dists) == 1 or dists[1][0] - dists[0][0] >= ATTACH_MARGIN):
            dists[0][2].append(lone.pop())
    people = [p for p in people if p]
    return {(s, t.track_id): i for i, p in enumerate(people) for s, t in p}


def map_speakers(turns: list[Turn], shots: list[Shot], tracks: dict[int, list[Track]],
                 person_of: dict[tuple[int, int], int]) -> People:
    speakers = sorted({t.speaker for t in turns})
    persons = sorted(set(person_of.values()))
    out = People(person_of=person_of)
    if not speakers or not persons:
        return out
    votes = np.zeros((len(speakers), len(persons)))
    for shot in shots:
        people = [t for t in tracks[shot.index] if t.is_person]
        if len(people) != 1:
            continue  # only unambiguous shots vote
        pid = person_of.get((shot.index, people[0].track_id))
        if pid is None:
            continue
        for turn in turns:
            overlap = min(turn.end, shot.end) - max(turn.start, shot.start)
            if overlap > 0:
                votes[speakers.index(turn.speaker), persons.index(pid)] += overlap
    rows, cols = linear_sum_assignment(-votes)
    for r, c in zip(rows, cols):
        total = votes[r].sum()
        if total <= 0:
            continue
        share = votes[r, c] / total
        out.confidence[speakers[r]] = float(share)
        if share >= MIN_CONFIDENCE:
            out.speaker_person[speakers[r]] = persons[c]
    return out


def active_person(turns: list[Turn], people: People, t: float) -> int | None:
    """Mapped person speaking at time t (None when nobody mapped is)."""
    for turn in turns:
        if turn.start <= t <= turn.end and turn.speaker in people.speaker_person:
            return people.speaker_person[turn.speaker]
    return None
