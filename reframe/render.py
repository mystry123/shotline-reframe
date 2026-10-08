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


def _fit_over_blur(src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    sh, sw = src.shape[:2]
    cover = max(out_w / sw, out_h / sh)
    bg = cv2.resize(src, (int(sw * cover) + 1, int(sh * cover) + 1), interpolation=cv2.INTER_AREA)
    oy, ox = (bg.shape[0] - out_h) // 2, (bg.shape[1] - out_w) // 2
    bg = cv2.GaussianBlur(bg[oy:oy + out_h, ox:ox + out_w], (0, 0), sigmaX=max(out_w, out_h) / 40)
    bg = (bg * 0.6).astype(np.uint8)
    fit = min(out_w / sw, out_h / sh)
    fg = cv2.resize(src, (int(sw * fit), int(sh * fit)), interpolation=cv2.INTER_CUBIC)
    y, x = (out_h - fg.shape[0]) // 2, (out_w - fg.shape[1]) // 2
    bg[y:y + fg.shape[0], x:x + fg.shape[1]] = fg
    return bg


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
        for z in seg["zones"]:
            c = z["canvas"]
            zx, zy = int(round(c["left"] * out_w)), int(round(c["top"] * out_h))
            zw, zh = int(round(c["width"] * out_w)), int(round(c["height"] * out_h))
            if seg["layout"] == "fit":
                picture = frame[pic["y"]:pic["y"] + pic["h"], pic["x"]:pic["x"] + pic["w"]]
                canvas[zy:zy + zh, zx:zx + zw] = _fit_over_blur(picture, zw, zh)
            elif seg["layout"] == "framed":
                # Blurred copy of the picture behind, the framed region on top.
                picture = frame[pic["y"]:pic["y"] + pic["h"], pic["x"]:pic["x"] + pic["w"]]
                canvas[:] = _fit_over_blur(picture, out_w, out_h) * 0 + _blur_cover(picture, out_w, out_h)
                crop = _crop(frame, *_interp(z["keyframes"], t))
                canvas[zy:zy + zh, zx:zx + zw] = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_CUBIC)
            else:
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
