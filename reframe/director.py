"""AI director: a vision model picks the framing of each shot.

The rules give every shot a safe framing ("rules") plus alternatives (closer,
as close as the shape allows, the person shown whole, the whole picture). For
shots with more than one option, each option is drawn on a still from the
shot, laid out as a labelled contact sheet, and Claude picks one per shot with
a short reason.

Any failure (no access, refusal, unparsable reply, low confidence) keeps the
rules' choice for that shot, so the director can only change framings, never
break a plan.

Environment:
    REFRAME_DIRECTOR         off (default) | bedrock | anthropic
    REFRAME_DIRECTOR_MODEL   model id without the Bedrock prefix (default claude-opus-5-5)
    REFRAME_DIRECTOR_REGION  AWS region for Bedrock (default us-east-1)
"""

from __future__ import annotations

import base64
import json
import os
import re
import time

import cv2
import numpy as np

from .layout import Segment

PROMPT_VERSION = "1"
MIN_CONFIDENCE = 0.5
SHOTS_PER_SHEET = 5
TILE_H = 240
MAX_SIDE = 1568            # larger images are scaled down by the API anyway
LETTERS = "ABCDEFG"

SYSTEM = """You are the editor reframing a video for another aspect ratio (for example a vertical phone video turned into a horizontal one). For each shot you see the original frame and up to five candidate framings, labelled A, B, C… A is the automatic choice.

Pick the candidate that will look best to a viewer, judging from the still:
- Never cut through a person's head or face, and don't leave the main person awkwardly cut at the edge.
- Text that is part of the picture (subtitles burned into the video, slide text, screen content, product names) must stay whole and readable. Cutting words off is a serious fault.
- Among candidates that keep everything important, prefer the one where the subject is biggest. A small person in a mostly blurred frame is worse than a slightly tighter frame that still keeps their head, face and any captions.
- Screens, slides and graphics should be shown whole when they carry information.
- Keep it consistent: shots from the same setup should get the same kind of framing unless their content differs.
Keep A unless another candidate is clearly better.

Reply with JSON only, no other text:
{"shots": [{"shot": <shot number>, "choice": "<letter>", "confidence": <0..1>, "reason": "<one short sentence>"}]}"""


# ---------------------------------------------------------------------------
# Drawing candidates
# ---------------------------------------------------------------------------

def _segment_at(segs: list[Segment], t: float) -> Segment:
    return next((s for s in segs if s.start <= t < s.end), segs[len(segs) // 2])


def _interp(kfs, t: float) -> tuple[float, float, float, float]:
    if t <= kfs[0].t:
        k = kfs[0]
        return k.x, k.y, k.w, k.h
    for a, b in zip(kfs, kfs[1:]):
        if t <= b.t:
            r = (t - a.t) / max(b.t - a.t, 1e-9)
            return a.x + r * (b.x - a.x), a.y + r * (b.y - a.y), a.w + r * (b.w - a.w), a.h + r * (b.h - a.h)
    k = kfs[-1]
    return k.x, k.y, k.w, k.h


def _crop(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0 = max(0, int(round(x * W))), max(0, int(round(y * H)))
    x1, y1 = min(W, int(round((x + w) * W))), min(H, int(round((y + h) * H)))
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]


def _blurred(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    sh, sw = img.shape[:2]
    cover = max(out_w / sw, out_h / sh)
    bg = cv2.resize(img, (int(sw * cover) + 1, int(sh * cover) + 1), interpolation=cv2.INTER_AREA)
    oy, ox = (bg.shape[0] - out_h) // 2, (bg.shape[1] - out_w) // 2
    bg = cv2.GaussianBlur(bg[oy:oy + out_h, ox:ox + out_w], (0, 0), sigmaX=max(out_w, out_h) / 40)
    return (bg * 0.6).astype(np.uint8)


def draw_option(still: np.ndarray, segs: list[Segment], t: float, target_aspect: float, out_h: int) -> np.ndarray:
    """The output frame an option produces at time t (still = the picture area)."""
    out_w = max(1, int(round(out_h * target_aspect)))
    seg = _segment_at(segs, t)
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    if seg.layout in ("fit", "framed"):
        canvas[:] = _blurred(still, out_w, out_h)
    for i, z in enumerate(seg.zones):
        left, top, cw, ch = z.canvas
        zx, zy, zw, zh = int(round(left * out_w)), int(round(top * out_h)), max(1, int(round(cw * out_w))), max(1, int(round(ch * out_h)))
        if seg.layout == "fit":
            sh, sw = still.shape[:2]
            fit = min(zw / sw, zh / sh)
            fg = cv2.resize(still, (max(1, int(sw * fit)), max(1, int(sh * fit))), interpolation=cv2.INTER_AREA)
            y, x = zy + (zh - fg.shape[0]) // 2, zx + (zw - fg.shape[1]) // 2
            canvas[y:y + fg.shape[0], x:x + fg.shape[1]] = fg
        else:
            crop = _crop(still, *_interp(z.keyframes, t))
            canvas[zy:zy + zh, zx:zx + zw] = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_AREA)
        if i == 1:  # divider between split zones
            canvas[zy - 1:zy + 1] = 0
    return canvas


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, tile])


def contact_sheet(rows: list[dict], target_aspect: float) -> bytes:
    """rows: {"number", "still", "options": [(letter, segs)], "t"} → JPEG bytes."""
    drawn_rows = []
    for row in rows:
        still = row["still"]
        sh, sw = still.shape[:2]
        original = cv2.resize(still, (max(1, int(TILE_H * sw / sh)), TILE_H), interpolation=cv2.INTER_AREA)
        tiles = [_label(original, f"Shot {row['number']}: original")]
        for letter, segs in row["options"]:
            tile = draw_option(still, segs, row["t"], target_aspect, TILE_H)
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 255, 255), 1)
            tiles.append(_label(tile, letter))
        gap = np.full((TILE_H + 28, 12, 3), 40, np.uint8)
        parts = []
        for tile in tiles:
            parts += [tile, gap]
        drawn_rows.append(np.hstack(parts[:-1]))
    width = max(r.shape[1] for r in drawn_rows)
    sheet = []
    for r in drawn_rows:
        pad = np.full((r.shape[0], width - r.shape[1], 3), 40, np.uint8)
        sheet += [np.hstack([r, pad]), np.full((16, width, 3), 40, np.uint8)]
    img = np.vstack(sheet[:-1])
    scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


# ---------------------------------------------------------------------------
# The model call
# ---------------------------------------------------------------------------

def configured() -> bool:
    return os.environ.get("REFRAME_DIRECTOR", "off") in ("bedrock", "anthropic")


def model_id() -> str:
    model = os.environ.get("REFRAME_DIRECTOR_MODEL", "claude-opus-5-5")
    return ("anthropic." + model) if os.environ.get("REFRAME_DIRECTOR") == "bedrock" else model


def _client():
    if os.environ.get("REFRAME_DIRECTOR") == "bedrock":
        from anthropic import AnthropicBedrockMantle
        return AnthropicBedrockMantle(aws_region=os.environ.get("REFRAME_DIRECTOR_REGION", "us-east-1"))
    from anthropic import Anthropic
    return Anthropic()


def _ask(sheets: list[bytes], shot_list: str) -> str:
    import anthropic

    content: list[dict] = []
    for i, jpg in enumerate(sheets):
        content.append({"type": "text", "text": f"Contact sheet {i + 1} of {len(sheets)}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.standard_b64encode(jpg).decode()}})
    content.append({"type": "text", "text": f"Shots to decide (letters available per shot):\n{shot_list}"})
    response = _client().messages.create(
        model=model_id(),
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the reply was cut off")
    return "".join(b.text for b in response.content if b.type == "text")


def _parse(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON in the reply")
    data = json.loads(match.group(0))
    shots = data.get("shots")
    if not isinstance(shots, list):
        raise ValueError("reply has no shots list")
    return shots


def direct(analysis, shot_options: list[dict], target_aspect: float) -> dict:
    """Choices for the shots that have alternatives.

    Returns {"used", "model", "choices": {shot index: option name},
    "notes": [{shot, choice, reason, confidence}], "seconds", "error"?}."""
    started = time.time()
    summary: dict = {"used": False, "model": model_id(), "choices": {}, "notes": []}
    rows, letters_of = [], {}
    for item in shot_options:
        shot, opts = item["shot"], item["options"]
        still_jpg = getattr(analysis, "stills", {}).get(shot.index)
        if len(opts) < 2 or still_jpg is None:
            continue
        still = cv2.imdecode(np.frombuffer(still_jpg, np.uint8), cv2.IMREAD_COLOR)
        names = list(opts)[: len(LETTERS)]          # "rules" first, so it is always A
        letters_of[shot.index] = dict(zip(LETTERS, names))
        rows.append({"number": shot.index + 1, "index": shot.index, "still": still, "t": (shot.start + shot.end) / 2,
                     "options": [(LETTERS[i], opts[n]) for i, n in enumerate(names)]})
    if not rows:
        summary["seconds"] = 0.0
        return summary
    try:
        sheets = [contact_sheet(rows[i:i + SHOTS_PER_SHEET], target_aspect) for i in range(0, len(rows), SHOTS_PER_SHEET)]
        shot_list = "\n".join(f"Shot {r['number']}: {', '.join(l for l, _ in r['options'])}" for r in rows)
        replies = _parse(_ask(sheets, shot_list))
        by_number = {r["number"]: r for r in rows}
        for reply in replies:
            row = by_number.get(int(reply.get("shot", -1)))
            if row is None:
                continue
            letter = str(reply.get("choice", "A")).strip().upper()[:1]
            name = letters_of[row["index"]].get(letter)
            confidence = float(reply.get("confidence", 0))
            note = {"shot": row["index"], "choice": name or "rules", "confidence": round(confidence, 2),
                    "reason": str(reply.get("reason", ""))[:200]}
            if name and name != "rules" and confidence >= MIN_CONFIDENCE:
                summary["choices"][row["index"]] = name
            elif name != "rules":
                note["choice"] = "rules"
            summary["notes"].append(note)
        summary["used"] = True
    except Exception as error:  # noqa: BLE001 - the rules stand in
        summary["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    summary["seconds"] = round(time.time() - started, 2)
    return summary
