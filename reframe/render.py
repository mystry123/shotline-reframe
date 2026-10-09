"""Preview renderer: applies a plan to the source video with ffmpeg + numpy.

Used for visual review and the test harness. Production renders happen in
Remotion from the same segments; both interpolate keyframes the same way
(linear within a segment, a hard step at segment boundaries).
"""

from __future__ import annotations

import subprocess

import cv2
import numpy as np


def _interp(kfs: list[dict], t: float) -> tuple[float, float, float, float]:
    if t <= kfs[0]["t"]:
        k = kfs[0]
        return k["x"], k["y"], k["w"], k["h"]
    for a, b in zip(kfs, kfs[1:]):
        if t <= b["t"]:
            r = (t - a["t"]) / max(b["t"] - a["t"], 1e-9)
            return tuple(a[c] + r * (b[c] - a[c]) for c in ("x", "y", "w", "h"))  # type: ignore[return-value]
    k = kfs[-1]
    return k["x"], k["y"], k["w"], k["h"]


def _crop(frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = frame.shape[:2]
    x0, y0 = int(round(x * W)), int(round(y * H))
    x1, y1 = int(round((x + w) * W)), int(round((y + h) * H))
    return frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]


def _blur_cover(src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    sh, sw = src.shape[:2]
    cover = max(out_w / sw, out_h / sh)
    bg = cv2.resize(src, (int(sw * cover) + 1, int(sh * cover) + 1), interpolation=cv2.INTER_AREA)
    oy, ox = (bg.shape[0] - out_h) // 2, (bg.shape[1] - out_w) // 2
    bg = cv2.GaussianBlur(bg[oy:oy + out_h, ox:ox + out_w], (0, 0), sigmaX=max(out_w, out_h) / 40)
    return (bg * 0.6).astype(np.uint8)


def _overlay_text(canvas: np.ndarray, frame: np.ndarray, ov: dict) -> None:
    """Lift burned-in text off its dark background and lay it over the canvas,
    keeping its colours."""
    H, W = frame.shape[:2]
    oh, ow = canvas.shape[:2]
    s, p = ov["source"], ov["placement"]
    src = frame[int(s["y"] * H):int((s["y"] + s["h"]) * H), int(s["x"] * W):int((s["x"] + s["w"]) * W)]
    tw, th = max(1, int(p["width"] * ow)), max(1, int(p["height"] * oh))
    tx, ty = int(p["left"] * ow), int(p["top"] * oh)
    text = cv2.resize(src, (tw, th), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # Alpha from brightness: the dark background goes transparent, soft edges stay soft.
    luma = text.max(axis=2)  # max channel keeps saturated colours (red, teal) opaque
    alpha = np.clip((luma - 30.0) / 50.0, 0, 1)
    # A soft shadow keeps light text readable over light parts of the video
    # (on the original black bars it didn't need one).
    radius = max(2, th // 40)
    shadow = cv2.GaussianBlur(cv2.dilate(alpha, np.ones((radius * 2 + 1,) * 2, np.uint8)), (0, 0), radius)
    region = canvas[ty:ty + th, tx:tx + tw].astype(np.float32)
    region *= (1 - 0.65 * shadow)[..., None]
    a = alpha[..., None]
    canvas[ty:ty + th, tx:tx + tw] = (a * text + (1 - a) * region).astype(np.uint8)


def _view(seg: dict, t: float, pic: dict, W: int, H: int, frame_aspect: float):
    """((left, top, width, height) canvas, (x, y, w, h) crop, blur 0..1) of a
    one-zone segment at time t, or None for split screens."""
    if seg["layout"] == "fit":
        pa = pic["w"] / pic["h"]
        w, h = (1.0, frame_aspect / pa) if pa > frame_aspect else (pa / frame_aspect, 1.0)
        return ((1 - w) / 2, (1 - h) / 2, w, h), (pic["x"] / W, pic["y"] / H, pic["w"] / W, pic["h"] / H), 1.0
    if len(seg["zones"]) != 1:
        return None
    z = seg["zones"][0]
    c = z["canvas"]
    return ((c["left"], c["top"], c["width"], c["height"]), _interp(z["keyframes"], t),
            1.0 if seg["layout"] == "framed" else 0.0)


def _ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 2 * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 2 / 2


def _blend(a, b, p: float):
    mix = lambda x, y: tuple(xi + (yi - xi) * p for xi, yi in zip(x, y))  # noqa: E731
    return mix(a[0], b[0]), mix(a[1], b[1]), a[2] + (b[2] - a[2]) * p


def render(source: str, result: dict, out_path: str, out_h: int = 720) -> None:
    src = result["source"]
    W, H, fps = src["width"], src["height"], src["fps"]
    a, b = (float(v) for v in result["target"].split(":"))
    out_w = int(round(out_h * a / b / 2) * 2)
    pic = src["picture"]
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", source, "-an", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
         "-i", "pipe:0", "-i", source, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path],
        stdin=subprocess.PIPE,
    )
    frame_bytes = W * H * 3
    segments = result["segments"]
    i = 0
    while True:
        buf = reader.stdout.read(frame_bytes)  # type: ignore[union-attr]
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        t = i / fps
        seg = next((s for s in segments if s["start"] <= t < s["end"]), segments[-1])
        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        picture = frame[pic["y"]:pic["y"] + pic["h"], pic["x"]:pic["x"] + pic["w"]]
        view = _view(seg, t, pic, W, H, out_w / out_h)
        if view is not None:
            # One-zone layouts share one model (crop shown in a canvas, maybe over
            # a blurred copy); an eased start blends from the previous framing.
            k = segments.index(seg)
            ease = (seg.get("transition") or {}).get("duration", 0) if (seg.get("transition") or {}).get("type") == "ease" else 0
            if ease and k > 0 and t < seg["start"] + ease:
                prev = segments[k - 1]
                before = _view(prev, min(t, prev["end"] - 1e-3), pic, W, H, out_w / out_h)
                if before is not None:
                    view = _blend(before, view, _ease((t - seg["start"]) / ease))
            (cl, ct, cw, ch), crop, blur = view
            if blur > 0:
                canvas[:] = (_blur_cover(picture, out_w, out_h) * blur).astype(np.uint8)
            zx, zy = int(round(cl * out_w)), int(round(ct * out_h))
            zw, zh = max(1, int(round(cw * out_w))), max(1, int(round(ch * out_h)))
            canvas[zy:zy + zh, zx:zx + zw] = cv2.resize(_crop(frame, *crop), (zw, zh), interpolation=cv2.INTER_CUBIC)
        else:
            for z in seg["zones"]:
                c = z["canvas"]
                zx, zy = int(round(c["left"] * out_w)), int(round(c["top"] * out_h))
                zw, zh = int(round(c["width"] * out_w)), int(round(c["height"] * out_h))
                crop = _crop(frame, *_interp(z["keyframes"], t))
                canvas[zy:zy + zh, zx:zx + zw] = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_CUBIC)
        if len(seg["zones"]) == 2:  # thin divider between split zones
            canvas[out_h // 2 - 1:out_h // 2 + 1] = 0
        for ov in result.get("overlays", []):
            _overlay_text(canvas, frame, ov)
        writer.stdin.write(canvas.tobytes())  # type: ignore[union-attr]
        i += 1
    writer.stdin.close()  # type: ignore[union-attr]
    writer.wait()
    reader.wait()
