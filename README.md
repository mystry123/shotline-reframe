# shotline-reframe

Shotline's AI reframe engine: turns a video into another shape (9:16, 1:1, 2:3, 16:9, …) by finding the real picture, the shots, the people and who is speaking, then choosing a layout and a steady camera path per shot.

Plan and quality targets: "Reframe Engine v2" (Shotline plan page, 8 Oct 2026).

## Pipeline (`reframe/`)

| Stage | Module | What it does |
|---|---|---|
| Probe | `probe.py` | Size, rotation, and the real picture area inside black bars (even when the bars hold titles or captions) |
| Decode | `decode.py` | ffmpeg crops, resamples and scales; Python only sees analysed frames |
| Shots | `shots.py` | Cut detection on the picture area only (adaptive, flash-frame tolerant) |
| Faces | `faces.py` | YuNet face detection (MIT), normalized coordinates |
| Tracks | `tracks.py` | ByteTrack per shot; rejects paintings, photos, objects and hands (presence, liveness, size) |
| Layout | `layout.py` | Per-shot layout (full / single / split / fit) and camera path (hold or smoothed pan, fixed zoom) |
| Engine | `engine.py` | `analyze(path)` once per file, `plan(analysis, "1:1")` per shape; segments in full-frame coordinates |
| Render | `render.py` | Preview renderer for review and tests (production renders in Remotion from the same segments) |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python - <<'PY'
from reframe.engine import analyze, plan
from reframe.render import render
a = analyze("input.mp4")
r = plan(a, "1:1")
render("input.mp4", r, "out_1x1.mp4")
PY
```

CPU only for now; every model runs through OpenCV DNN / ONNX Runtime so moving to GPU is a configuration change.
