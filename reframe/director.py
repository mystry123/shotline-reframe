"""AI director: a vision model picks the framing of each shot.

The rules give every shot a safe framing ("rules") plus alternatives (closer,
as close as the shape allows, the person shown whole, the whole picture). For
shots with more than one option, each option is drawn on a still from the
shot, laid out as a labelled contact sheet, and a vision model picks one per
shot with a short reason.

Any failure (no access, timeout, refusal, invalid reply, low confidence) keeps
the rules' choice for that shot and is logged, so the director can only change
framings, never break a plan. Rate limits (429) and server errors (5xx) are
retried with exponential backoff first.

Environment (the older REFRAME_DIRECTOR* names are still read as fallbacks):
    DIRECTOR_PROVIDER        off (default)
                             gemini     Gemini on Vertex AI (google-genai, Application
                                        Default Credentials; no API keys). Also reads
                                        GOOGLE_GENAI_USE_VERTEXAI=True,
                                        GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
                             converse   any vision model on Amazon Bedrock (Converse API)
                             bedrock    Claude on Amazon Bedrock
                             anthropic  Claude on the Claude API
    DIRECTOR_MODEL           model id. Defaults: gemini → gemini-3.8-flash;
                             converse → qwen.qwen3-vl-235b-a22b; bedrock/anthropic →
                             claude-opus-5-5 (Bedrock's "anthropic." prefix is added)
    DIRECTOR_REGION          AWS region for Bedrock (default ap-south-1)
    DIRECTOR_THINKING        Gemini thinking level (default low: a whole video's
                             sheets in seconds; medium/high can exceed the deadline)
    DIRECTOR_TIMEOUT_S       per-attempt timeout (default 60)
    DIRECTOR_RETRIES         extra attempts after a 429/5xx (default 3)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import time

import cv2
import numpy as np

from .layout import Segment

log = logging.getLogger("reframe.director")

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

def _env(name: str, default: str = "") -> str:
    """DIRECTOR_<name>, or the older REFRAME_DIRECTOR_<name> / REFRAME_DIRECTOR."""
    legacy = "REFRAME_DIRECTOR" if name == "PROVIDER" else f"REFRAME_DIRECTOR_{name}"
    return os.environ.get(f"DIRECTOR_{name}") or os.environ.get(legacy) or default


def _provider() -> str:
    return _env("PROVIDER", "off")


def _region() -> str:
    return _env("REGION", "ap-south-1")


def configured() -> bool:
    return _provider() in ("gemini", "converse", "bedrock", "anthropic")


def model_id() -> str:
    if _provider() == "gemini":
        return _env("MODEL", "gemini-3.8-flash")
    if _provider() == "converse":
        return _env("MODEL", "qwen.qwen3-vl-235b-a22b")
    model = _env("MODEL", "claude-opus-5-5")
    return ("anthropic." + model) if _provider() == "bedrock" else model


def _client():
    if _provider() == "bedrock":
        from anthropic import AnthropicBedrockMantle
        return AnthropicBedrockMantle(aws_region=_region())
    from anthropic import Anthropic
    return Anthropic()


class Retryable(Exception):
    """A rate limit or server error worth trying again."""


def _with_retries(call):
    """Run `call`, retrying Retryable errors with exponential backoff and jitter."""
    retries = int(_env("RETRIES", "3"))
    for attempt in range(retries + 1):
        try:
            return call()
        except Retryable as error:
            if attempt == retries:
                raise
            delay = min(20.0, 1.5 * 2 ** attempt) * (0.75 + 0.5 * random.random())
            log.warning("director: %s; retrying in %.1fs (attempt %d of %d)", error, delay, attempt + 2, retries + 1)
            time.sleep(delay)


def _ask(sheets: list[bytes], shot_list: str) -> str:
    if _provider() == "gemini":
        return _with_retries(lambda: _ask_gemini(sheets, shot_list))
    if _provider() == "converse":
        return _ask_converse(sheets, shot_list)
    return _ask_claude(sheets, shot_list)


# The reply shape, enforced by Gemini's structured output.
REPLY_SCHEMA = {"type": "object", "required": ["shots"], "properties": {"shots": {"type": "array", "items": {
    "type": "object", "required": ["shot", "choice", "confidence", "reason"], "properties": {
        "shot": {"type": "integer"}, "choice": {"type": "string"}, "confidence": {"type": "number"},
        "reason": {"type": "string"}}}}}}


_gemini = None


def _gemini_client():
    """One Vertex AI client per process (a client closes itself when garbage
    collected, so it must be kept). Configured from the environment
    (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) and
    Application Default Credentials: gcloud ADC on a laptop, the attached
    service account on GCP."""
    global _gemini
    if _gemini is None:
        _gemini = _new_gemini_client()
    return _gemini


def _new_gemini_client():
    """Vertex AI client from the environment (GOOGLE_GENAI_USE_VERTEXAI,
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) and Application Default
    Credentials: gcloud ADC on a laptop, the attached service account on GCP."""
    from google import genai
    from google.genai import types

    timeout_ms = int(float(_env("TIMEOUT_S", "60")) * 1000)
    # Retries are ours (_with_retries), so the SDK makes a single attempt.
    return genai.Client(http_options=types.HttpOptions(timeout=timeout_ms,
                                                      retry_options=types.HttpRetryOptions(attempts=1)))


def _ask_gemini(sheets: list[bytes], shot_list: str) -> str:
    from google.genai import errors, types

    contents: list = []
    for i, jpg in enumerate(sheets):
        contents.append(f"Contact sheet {i + 1} of {len(sheets)}:")
        contents.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
    contents.append(f"Shots to decide (letters available per shot):\n{shot_list}")
    try:
        response = _gemini_client().models.generate_content(
            model=model_id(),
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                               response_json_schema=REPLY_SCHEMA,
                                               thinking_config=types.ThinkingConfig(thinking_level=_env("THINKING", "low"))),
        )
    except errors.APIError as error:
        if error.code == 429 or (error.code or 0) >= 500:
            raise Retryable(f"Gemini HTTP {error.code}") from error
        raise
    candidate = (response.candidates or [None])[0]
    reason = getattr(candidate, "finish_reason", None)
    if reason is not None and getattr(reason, "name", str(reason)) != "STOP":
        raise RuntimeError(f"Gemini stopped: {getattr(reason, 'name', reason)}")
    if not response.text:
        raise ValueError("empty reply")
    return response.text


def _ask_converse(sheets: list[bytes], shot_list: str) -> str:
    """Any Bedrock model that takes images, through the Converse API."""
    import boto3

    content: list[dict] = []
    for i, jpg in enumerate(sheets):
        content.append({"text": f"Contact sheet {i + 1} of {len(sheets)}:"})
        content.append({"image": {"format": "jpeg", "source": {"bytes": jpg}}})
    content.append({"text": f"Shots to decide (letters available per shot):\n{shot_list}"})
    response = boto3.client("bedrock-runtime", region_name=_region()).converse(
        modelId=model_id(),
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 4000},
    )
    if response.get("stopReason") in ("content_filtered", "guardrail_intervened"):
        raise RuntimeError("the model declined")
    if response.get("stopReason") == "max_tokens":
        raise RuntimeError("the reply was cut off")
    return "".join(block.get("text", "") for block in response["output"]["message"]["content"])


def _ask_claude(sheets: list[bytes], shot_list: str) -> str:
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
        log.warning("director: %s failed, keeping the rules' framing: %s", model_id(), summary["error"])
    summary["seconds"] = round(time.time() - started, 2)
    return summary
