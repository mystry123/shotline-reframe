# shotline-reframe

The reframing engine behind [Shotline](https://shotline.in), a browser video editor. It turns a video into another shape (9:16, 1:1, 2:3, 16:9, and others) the way an editor would. It finds the real picture, the shots, the people and who is speaking, then picks a layout and a steady camera path for each shot.

The engine runs in production as an HTTP service called by the Shotline backend. It doesn't render video. It returns a **plan**: per-shot segments with crop keyframes in source-frame coordinates. Shotline renders that plan in Remotion, and this repo's preview renderer replays it the same way for review and tests.

**Stack:** Python 3.12 · FastAPI · OpenCV DNN / ONNX Runtime · ffmpeg · ByteTrack · SciPy · Gemini on Vertex AI (optional) · GitHub Actions · GCP

## The problem

A naive center crop or a "follow the biggest face" crop breaks on real uploads:

- **Re-exports with bars.** The real picture often sits inside black or blurred bars, and those bars carry titles and animated captions. Black-bar detection gets fooled by that text.
- **False faces.** Face detectors fire on paintings, photos, ornaments and hands. A turned-away or backlit person loses their face box.
- **Jump cuts.** A talking-head video has dozens of cuts from one camera position. Framing each shot on its own makes the person resize at every cut.
- **Several people.** When two or three people are on screen, the camera has to follow whoever is talking, without flipping on every "yeah".
- **Jitter.** Smoothing detector output with a moving average still drifts and wobbles. A real camera operator holds, pans at a steady speed, or eases between the two.
- **Burned-in captions.** A crop that slices the caption band in half looks broken.

The engine handles each of these with a measured, testable rule. An optional vision model then chooses among the framings those rules have already marked safe, and never invents one.

## How it works

```
analyze(path) → Analysis          once per file; expensive; cached
plan(analysis, "9:16") → plan     per shape and zoom; milliseconds
```

| Stage | Module | What it does |
|---|---|---|
| Probe | `probe.py` | Size, rotation, and the real picture area inside bars. Rows and columns are scored by how picture-like they are across many frames, so caption text in the bars doesn't fool it. |
| Blurred bars | `borders.py` | Finds a picture placed over a blurred, enlarged copy of itself, using sharpness profiles. |
| Decode | `decode.py` | ffmpeg rotates, crops, resamples and scales in C; Python only sees the frames it analyses. Long videos stay within a memory budget. |
| Shots | `shots.py` | Adaptive cut detection on the picture area only, plus fades and dissolves. It tolerates flash frames and strobes. |
| Camera motion | `motion.py` | Source camera pans, tilts and zooms (Lucas-Kanade + RANSAC, with people masked out). |
| Faces | `faces.py` | YuNet face detection (MIT licence). |
| Bodies | `bodies.py` | YOLOX-S person detection (Apache-2.0). Confirms a face belongs to a body, and keeps following people when the face is lost. |
| Tracks | `tracks.py`, `persons.py` | ByteTrack per shot. A face counts as a person only if a body claims it as its head and its image changes over time, which rejects paintings, photos and ornaments. |
| Content | `content.py` | For shots with no people (slides, graphics, screen recordings, products): the box holding what stands out. |
| Text | `text.py`, `overlays.py` | PP-OCRv3 text detection, used to keep captions in frame, recognise screen recordings and slides, and ask whether to keep titles from the bars. |
| Speakers | `speakers.py`, `lips.py` | Speech turns from word-level diarization, people matched across shots by appearance, and mouth motion on aligned faces to tell who is talking. Backchannels never move the camera. |
| Layout | `layout.py` | Per-shot layout: `full`, `single` (follow one person), `split` (two stacked people) or `fit` (picture over a blurred background). |
| Camera paths | `paths.py` | L1-optimal paths (Grundmann et al., CVPR 2011, the approach behind YouTube's stabiliser): holds, constant-speed pans and eased moves, never drift. |
| Continuity | `continuity.py`, `smoothing.py` | Shots from one camera setup share one framing. Framing changes within a shot ease, and a short shot between two cuts can't blink a different layout. |
| Director | `director.py` | Optional AI director. Each shot's safe alternatives are drawn on a labelled contact sheet and a vision model picks one with a reason. Any failure, timeout or low-confidence reply keeps the rules' choice. |
| Render | `render.py` | Preview renderer (ffmpeg + numpy) that interpolates keyframes exactly as production does. |

All models are small, permissively licensed ONNX files from the OpenCV model zoo (`models/`), running on CPU through OpenCV DNN or ONNX Runtime. Moving them to a GPU is a configuration change.

## Quality: an evaluation harness with a CI gate

Reframing quality is easy to regress without noticing, so every change goes through `eval/`:

- **Synthetic clips with exact answers.** `eval/synthetic.py` uses ffmpeg's test patterns to build clips with known cuts, dissolves, fades, bars and camera pans. They're small, generated on the fly, and run in CI.
- **Labelled real clips.** `eval/labels/*.json` holds the labels; the videos stay out of git.
- **Metrics** (`eval/metrics.py`). Each plan is replayed exactly as the renderer draws it and measured:
  - people and captions: face cut rate, caption-band cut rate, share of the upper body kept
  - motion: unmotivated jumps, 95th-percentile jerk, direction reversals per minute, zoom
  - against labels: transition F1 and picture-area IoU
- **Baseline.** Results are compared with the committed `eval/baseline.json`. CI fails when quality regresses.

```bash
make eval            # synthetic + real clips present in testdata/
make eval-ci         # what CI runs
make eval-baseline   # accept the current numbers as the new baseline
```

Unit and integration tests (`tests/`, 86 tests) cover host allowlisting, borders, continuity, the director's fallbacks, framing guards, lip activity, camera paths, smoothing, speaker switching, transitions and the HTTP service. Tests that need real clips skip in CI.

## Service

```
POST /analyses             {videoUrl, words?, quality?}        → 202 {id, status}
GET  /analyses/{id}        → {status, stage, progress, error?, text?}
POST /analyses/{id}/plan   {ratio, zoom?, keepText?, outputHeight?, captions?} → plan
GET  /health
```

- **Background analyses.** Analyses run in the background, so no request waits on one.
- **Deduplication.** The analysis id is derived from the inputs, so resubmitting the same video returns the existing analysis.
- **Caching.** Results are cached on disk and survive restarts. Planning a shape from a finished analysis takes milliseconds.
- **Security.** Every request except `/health` needs `X-Internal-Token`. Videos are downloaded only from an allowlist of hosts, optionally narrowed to a path prefix such as one Cloud Storage bucket (`REFRAME_ALLOWED_HOSTS`), without following redirects, and under a size cap.

```bash
REFRAME_SERVICE_TOKEN=… REFRAME_ALLOWED_HOSTS=storage.googleapis.com/<bucket> \
  .venv/bin/uvicorn reframe.service:app --port 8090
```

For local development without a token, set `REFRAME_DEV=1`. All settings are documented at the top of `reframe/service.py`.

### AI director (optional)

The director uses Gemini on Vertex AI with Application Default Credentials, so there are no API keys: run `gcloud auth application-default login` on a laptop, and production uses the VM's attached service account.

```bash
DIRECTOR_PROVIDER=gemini DIRECTOR_MODEL=gemini-3.8-flash \
GOOGLE_GENAI_USE_VERTEXAI=True GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global
```

Other providers are `bedrock` (Claude on Bedrock), `converse` (any Bedrock vision model) and `anthropic`. Calls retry 429 and 5xx responses with backoff. Settings: `DIRECTOR_THINKING`, `DIRECTOR_TIMEOUT_S`, `DIRECTOR_RETRIES`.

## Run it locally

Requires Python 3.12 and ffmpeg.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
```

Reframe a clip and render a preview:

```bash
.venv/bin/python - <<'PY'
from reframe.engine import analyze, plan
from reframe.render import render
a = analyze("input.mp4")
r = plan(a, "9:16")
render("input.mp4", r, "out_9x16.mp4")
PY
```

## Deployment

- **CI** (`.github/workflows/ci.yml`): ruff, pytest and the synthetic evaluation on every push.
- **Deploy** (`.github/workflows/deploy.yml`): after CI passes on `main`, GitHub Actions signs in to GCP through **Workload Identity Federation**, so no long-lived keys are stored. The federation is restricted to this repository's `main` branch. The workflow reaches the VM over an **IAP** tunnel and runs `deploy/shotline-deploy`, the only command the deployer may run through sudo. The deploy runs as its own systemd job, so a dropped connection doesn't kill it. It health-checks the new version and rolls back automatically if that fails, and distinct exit codes let CI tell a failed deploy from an unreachable VM. Manual dispatch redeploys, or rolls back to any commit on `main`.
- **Runtime:** a systemd service bound to localhost, behind Caddy for HTTPS.

## Licences

Models: YuNet (MIT), YOLOX-S (Apache-2.0) and PP-OCRv3 (Apache-2.0), from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo).
