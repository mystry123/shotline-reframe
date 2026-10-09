"""Synthetic clips with exact answers, made with ffmpeg from its built-in test
patterns (so they are ours, small, and run in CI).

Each clip has a truth record:
    transitions   [{type: cut|dissolve|fade, start, end}] (a cut: start == end)
    picture       {x, y, w, h} in pixels: the real picture inside any bars
    camera        {"pan_px_per_s": dx} or {"zoom": [from, to], "over_s": s} (source camera motion)

Generated into eval/.cache/synthetic (git-ignored); regenerated when missing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache" / "synthetic"
FPS = 25

# Distinct-looking, cheap sources (lavfi), each 1280x720 unless stated.
A = "testsrc2=s={s}:r=25:d={d}"
B = "smptehdbars=s={s}:r=25:d={d}"
C = "gradients=s={s}:r=25:d={d}:speed=0.03:c0=0x203060:c1=0xe0a020:c2=0x30a060:c3=0xa03050:nb_colors=4"


def _src(kind: str, s: str = "1280x720", d: float = 3.0) -> list[str]:
    spec = {"a": A, "b": B, "c": C}[kind].format(s=s, d=d)
    return ["-f", "lavfi", "-t", str(d), "-i", spec]


CLIPS: dict[str, dict] = {
    "cuts": {
        "inputs": _src("a", d=2) + _src("b", d=2) + _src("c", d=2),
        "filter": "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "truth": {"transitions": [{"type": "cut", "start": 2.0, "end": 2.0}, {"type": "cut", "start": 4.0, "end": 4.0}],
                  "picture": {"x": 0, "y": 0, "w": 1280, "h": 720}},
    },
    "dissolve": {
        "inputs": _src("a", d=3) + _src("b", d=3),
        "filter": "[0:v][1:v]xfade=transition=fade:duration=1:offset=2,format=yuv420p[v]",
        "truth": {"transitions": [{"type": "dissolve", "start": 2.0, "end": 3.0}],
                  "picture": {"x": 0, "y": 0, "w": 1280, "h": 720}},
    },
    "fade": {
        "inputs": _src("a", d=3) + _src("c", d=3),
        "filter": "[0:v][1:v]xfade=transition=fadeblack:duration=1:offset=2,format=yuv420p[v]",
        "truth": {"transitions": [{"type": "fade", "start": 2.0, "end": 3.0}],
                  "picture": {"x": 0, "y": 0, "w": 1280, "h": 720}},
    },
    "black_bars": {   # a 9:16 picture centred in 16:9 on black
        "inputs": _src("a", s="720x1280", d=4),
        "filter": "[0:v]scale=608:1080,pad=1920:1080:656:0:black,format=yuv420p[v]",
        "truth": {"transitions": [], "picture": {"x": 656, "y": 0, "w": 608, "h": 1080}},
    },
    "blurred_bars": {   # the same over a blurred, zoomed copy of itself (CapCut style)
        "inputs": _src("a", s="720x1280", d=4),
        "filter": ("[0:v]split[f][b];[b]scale=1920:-2,crop=1920:1080,boxblur=24:2[bg];"
                   "[f]scale=608:1080[fg];[bg][fg]overlay=656:0,format=yuv420p[v]"),
        "truth": {"transitions": [], "picture": {"x": 656, "y": 0, "w": 608, "h": 1080}},
    },
    "blurred_bars_tall": {   # a 16:9 picture inside 9:16 over a blurred copy (top and bottom bars)
        "inputs": _src("a", s="1280x720", d=4),
        "filter": ("[0:v]split[f][b];[b]scale=-2:1920,crop=1080:1920,boxblur=24:2[bg];"
                   "[f]scale=1080:608[fg];[bg][fg]overlay=0:656,format=yuv420p[v]"),
        "truth": {"transitions": [], "picture": {"x": 0, "y": 656, "w": 1080, "h": 608}},
    },
    "pan": {   # the camera pans right across a wide, detailed still at 160 px/s (on 1280 px)
        "inputs": ["-loop", "1", "-t", "5", "-framerate", "25", "-i", "{texture}"],
        "filter": "[0:v]crop=1280:720:x='min(t*160,1280)':y=0,format=yuv420p[v]",
        "truth": {"transitions": [], "picture": {"x": 0, "y": 0, "w": 1280, "h": 720},
                  "camera": {"pan_px_per_s": 160}},
    },
    "zoom": {   # the camera zooms in 1.0x -> 1.5x over 4 s on the centre of a detailed still
        "inputs": ["-loop", "1", "-t", "4", "-framerate", "25", "-i", "{texture}"],
        "filter": "[0:v]crop=1280:720:640:0,scale=2560:1440,zoompan=z='1+0.5*on/100':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s=1280x720:fps=25,format=yuv420p[v]",
        "truth": {"transitions": [], "picture": {"x": 0, "y": 0, "w": 1280, "h": 720},
                  "camera": {"zoom": [1.0, 1.5], "over_s": 4.0}},
    },
}


# Built from the real clips in testdata/ when they are there (local runs only;
# never in CI or git): realistic footage with exact transitions and bars.
TESTDATA = Path(__file__).resolve().parent.parent / "testdata"
LOCAL: dict[str, dict] = {
    "real_dissolve": {
        "needs": ["clip006", "clip005"],
        "args": ["-ss", "2", "-t", "4", "-i", "{clip006}", "-ss", "0", "-t", "4", "-i", "{clip005}"],
        "filter": ("[0:v]scale=480:854,setsar=1,fps=30[a];[1:v]scale=480:854,setsar=1,fps=30[b];"
                   "[a][b]xfade=transition=fade:duration=0.8:offset=3,format=yuv420p[v]"),
        "truth": {"transitions": [{"type": "dissolve", "start": 3.0, "end": 3.8}], "picture": {"x": 0, "y": 0, "w": 480, "h": 854}},
    },
    "real_fade": {
        "needs": ["clip006", "clip005"],
        "args": ["-ss", "2", "-t", "4", "-i", "{clip006}", "-ss", "0", "-t", "4", "-i", "{clip005}"],
        "filter": ("[0:v]scale=480:854,setsar=1,fps=30[a];[1:v]scale=480:854,setsar=1,fps=30[b];"
                   "[a][b]xfade=transition=fadeblack:duration=0.8:offset=3,format=yuv420p[v]"),
        "truth": {"transitions": [{"type": "fade", "start": 3.0, "end": 3.8}], "picture": {"x": 0, "y": 0, "w": 480, "h": 854}},
    },
    "real_blurred_bars": {
        "needs": ["clip003"],
        "args": ["-t", "20", "-i", "{clip003}"],
        "filter": ("[0:v]split[f][b];[b]scale=1920:-2,crop=1920:1080,gblur=sigma=30[bg];"
                   "[f]scale=608:1080[fg];[bg][fg]overlay=656:0,format=yuv420p[v]"),
        "truth": {"transitions": [{"type": "cut", "start": t, "end": t} for t in (0.77, 2.17, 5.43, 6.6, 7.57, 9.23,
                                                                                    10.47, 11.2, 11.97, 13.53, 16.5, 18.2)],
                  "picture": {"x": 656, "y": 0, "w": 608, "h": 1080}},
    },
}


def local_available() -> list[str]:
    return [n for n, spec in LOCAL.items() if all((TESTDATA / f"{c}.mp4").exists() for c in spec["needs"])]


def ensure_local(name: str) -> Path:
    out = CACHE.parent / "local" / f"{name}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = LOCAL[name]
    args = [a.format(**{c: str(TESTDATA / f"{c}.mp4") for c in spec["needs"]}) for a in spec["args"]]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", spec["filter"], "-map", "[v]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", str(out)], check=True)
    return out


def texture() -> Path:
    """A detailed, non-repeating still (one Mandelbrot frame, 2560x720) for
    the camera-motion clips: flat or repeating patterns can't be tracked."""
    out = CACHE / "texture.png"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "mandelbrot=s=2560x720:start_scale=1.2:end_pts=1:maxiter=256", "-frames:v", "1", str(out)],
                       check=True)
    return out


def path(name: str) -> Path:
    return CACHE / f"{name}.mp4"


def truth(name: str) -> dict:
    return CLIPS[name]["truth"]


def ensure(name: str) -> Path:
    """The clip's file, generated if missing."""
    out = path(name)
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    spec = CLIPS[name]
    inputs = [str(texture()) if a == "{texture}" else a for a in spec["inputs"]]
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", spec["filter"], "-map", "[v]",
           "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)]
    subprocess.run(cmd, check=True)
    (CACHE / f"{name}.json").write_text(json.dumps(spec["truth"], indent=1))
    return out


if __name__ == "__main__":
    for name in CLIPS:
        print(name, ensure(name))
