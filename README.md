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
| Bodies | `bodies.py` | YOLOX-S person detection (Apache-2.0): confirms faces belong to people, fallback for turned-away subjects |
| Tracks | `tracks.py` | ByteTrack per shot; a face is a person when a body claims it as its head and its image changes over time (rejects ornaments, paintings, photos, hands) |
| Speakers | `speakers.py` | Transcript speaker turns, people across shots by appearance, voice → person mapping |
| Text | `overlays.py` | Burned-in titles/captions in the bars, for the keep-or-drop question |
| Options | `options.py` | Analysis quality (plan), output resolution (plan), zoom tightness (user) |
| Layout | `layout.py` | Per-shot layout (full / single / split / fit) and camera path (hold or smoothed pan, fixed zoom) |
| Engine | `engine.py` | `analyze(path)` once per file, `plan(analysis, "1:1")` per shape; segments in full-frame coordinates |
| Director | `director.py` | Optional AI director: per-shot alternatives drawn on contact sheets, a vision model picks one; any failure keeps the rules |
| Render | `render.py` | Preview renderer for review and tests (production renders in Remotion from the same segments) |

## Service

```bash
REFRAME_SERVICE_TOKEN=… REFRAME_ALLOWED_HOSTS=cdn.example.com,bucket.s3.amazonaws.com \
  .venv/bin/uvicorn reframe.service:app --port 8090
```

`POST /analyses` → `GET /analyses/{id}` (stage, progress, detected text) → `POST /analyses/{id}/plan` (ratio, zoom, keepText, outputHeight). See `reframe/service.py` for details. For local development without a token: `REFRAME_DEV=1`.

### AI director (optional)

Gemini on Vertex AI with Application Default Credentials (no API keys): `gcloud auth application-default login` on a laptop, the VM's attached service account in production.

```bash
DIRECTOR_PROVIDER=gemini DIRECTOR_MODEL=gemini-3.8-flash \
GOOGLE_GENAI_USE_VERTEXAI=True GOOGLE_CLOUD_PROJECT=shotline-511017 GOOGLE_CLOUD_LOCATION=global
```

Other providers: `bedrock` (Claude on Bedrock), `converse` (any Bedrock vision model), `anthropic`. Retries 429/5xx with backoff; on failure, timeout or invalid output the plan keeps the rules' framing and the reason is logged. Settings: `DIRECTOR_THINKING` (default `low`), `DIRECTOR_TIMEOUT_S` (60), `DIRECTOR_RETRIES` (3). See `reframe/director.py`.

## Run the engine directly

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

Models: YuNet (MIT) and YOLOX-S (Apache-2.0) from the OpenCV model zoo, in `models/`.

CPU only for now; every model runs through OpenCV DNN / ONNX Runtime so moving to GPU is a configuration change.
