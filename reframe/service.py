"""HTTP service for the reframe engine (used by the Shotline backend).

    POST /analyses             {videoUrl, words?, quality?}  → 202 {id, status}
    GET  /analyses/{id}        → {id, status, stage, stageLabel, progress, error?, text?}
    POST /analyses/{id}/plan   {ratio, zoom?, keepText?, outputHeight?} → plan (segments …)
    GET  /health

Analyses run in the background (no request waits for one). The id is derived
from the inputs, so submitting the same video again returns the same analysis
instead of redoing it; results are cached on disk and survive restarts.
Planning a shape from a finished analysis takes milliseconds.

Security: every request (except /health) needs X-Internal-Token; videos are
downloaded only from REFRAME_ALLOWED_HOSTS, without following redirects.

Environment:
    REFRAME_SERVICE_TOKEN   shared secret (required unless REFRAME_DEV=1)
    REFRAME_ALLOWED_HOSTS   comma-separated hostnames videos may come from
    REFRAME_CACHE_DIR       where analyses are kept (default ./cache)
    REFRAME_WORKERS         analyses run at the same time (default 1)
    REFRAME_MAX_BYTES       largest video accepted (default 4 GB)
    REFRAME_CACHE_DAYS      how long analyses are kept (default 14)
    REFRAME_DIRECTOR …      AI director settings (see reframe/director.py)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import director
from .engine import ENGINE_VERSION, Analysis, analyze, parse_ratio, plan, shot_options
from .options import OUTPUT_HEIGHTS, QUALITY, ZOOM, Options

TOKEN = os.environ.get("REFRAME_SERVICE_TOKEN", "")
DEV = os.environ.get("REFRAME_DEV") == "1"
ALLOWED_HOSTS = {h.strip().lower() for h in os.environ.get("REFRAME_ALLOWED_HOSTS", "").split(",") if h.strip()}
CACHE_DIR = Path(os.environ.get("REFRAME_CACHE_DIR", "cache")).resolve()
WORKERS = int(os.environ.get("REFRAME_WORKERS", "1"))
MAX_BYTES = int(os.environ.get("REFRAME_MAX_BYTES", str(4 * 1024**3)))
CACHE_DAYS = float(os.environ.get("REFRAME_CACHE_DAYS", "14"))

if not TOKEN and not DEV:
    raise RuntimeError("REFRAME_SERVICE_TOKEN is not set (set REFRAME_DEV=1 for local development only)")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="shotline-reframe", version=ENGINE_VERSION)
executor = ThreadPoolExecutor(max_workers=WORKERS)
_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}          # id -> status (in memory; finished ones are also on disk)
_loaded: dict[str, Analysis] = {}              # small in-memory cache of finished analyses


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    videoUrl: str
    words: list[dict] | None = None
    quality: str = Field(default="standard")


class PlanRequest(BaseModel):
    ratio: str
    zoom: str = "balanced"
    keepText: list[str] = Field(default_factory=list)
    outputHeight: int = 1080
    director: bool = True       # let the AI director pick framings (when configured)


def _auth(token: str | None) -> None:
    if DEV and not TOKEN:
        return
    if not token or not hmac.compare_digest(token, TOKEN):
        raise HTTPException(401, "Missing or wrong X-Internal-Token")


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise HTTPException(400, "videoUrl must be an http(s) URL")
    if parsed.scheme == "http" and not DEV:
        raise HTTPException(400, "videoUrl must use https")
    if ALLOWED_HOSTS and parsed.hostname.lower() not in ALLOWED_HOSTS:
        raise HTTPException(400, f"videoUrl host {parsed.hostname} is not allowed")


def _analysis_id(req: AnalysisRequest) -> str:
    words_digest = hashlib.sha256(json.dumps(req.words or [], sort_keys=True).encode()).hexdigest()
    key = f"{ENGINE_VERSION}|{req.videoUrl}|{req.quality}|{words_digest}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _path(analysis_id: str) -> Path:
    return CACHE_DIR / f"{analysis_id}.pkl"


# ---------------------------------------------------------------------------
# Background work
# ---------------------------------------------------------------------------

def _set(analysis_id: str, **fields: Any) -> None:
    with _lock:
        _jobs.setdefault(analysis_id, {"id": analysis_id}).update(fields)


def _download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=False, timeout=httpx.Timeout(30.0, read=120.0)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Couldn't download the video (HTTP {r.status_code})")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise RuntimeError("The video is larger than the reframe limit")
                f.write(chunk)


def _run(analysis_id: str, req: AnalysisRequest) -> None:
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="reframe-") as tmp:
            _set(analysis_id, status="running", stage="download", stageLabel="Downloading the video", progress=0.0)
            video = Path(tmp) / "video"
            _download(req.videoUrl, video)

            def on_stage(key: str, label: str, progress: float) -> None:
                # Download counts as the first 5%.
                _set(analysis_id, stage=key, stageLabel=label, progress=round(0.05 + 0.95 * progress, 3))

            result = analyze(str(video), req.words, quality=req.quality, on_stage=on_stage)
        result.info = type(result.info)(**{**result.info.__dict__, "path": req.videoUrl})
        tmp_path = _path(analysis_id).with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(result, f)
        tmp_path.replace(_path(analysis_id))
        with _lock:
            _loaded[analysis_id] = result
            while len(_loaded) > 16:
                _loaded.pop(next(iter(_loaded)))
        _set(analysis_id, status="done", stage="done", stageLabel="Done", progress=1.0,
             seconds=round(time.time() - started, 2), timings={k: round(v, 2) for k, v in result.timings.items()})
    except Exception as error:  # noqa: BLE001 - reported to the caller
        traceback.print_exc()
        message = str(error) if isinstance(error, RuntimeError) else "The video couldn't be analysed."
        _set(analysis_id, status="failed", error=message)


def _load(analysis_id: str) -> Analysis | None:
    with _lock:
        if analysis_id in _loaded:
            return _loaded[analysis_id]
    path = _path(analysis_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        result = pickle.load(f)
    with _lock:
        _loaded[analysis_id] = result
    return result


def _directed(analysis_id: str, result: Analysis, ratio: str, options: Options) -> dict:
    """The AI director's picks for this analysis and shape, cached on disk
    (only successful runs are cached; failures are retried next time)."""
    key = hashlib.sha256(f"{analysis_id}|{ratio}|{options.zoom}|{options.output_height}|"
                         f"{director.model_id()}|{director.PROMPT_VERSION}".encode()).hexdigest()[:32]
    path = CACHE_DIR / "director" / f"{key}.json"
    if path.exists():
        summary = json.loads(path.read_text())
        summary["choices"] = {int(k): v for k, v in summary["choices"].items()}
        summary["cached"] = True
        return summary
    summary = director.direct(result, shot_options(result, ratio, options), parse_ratio(ratio), options.output_height)
    if summary.get("used"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(summary))
    return summary


def _cleanup() -> None:
    cutoff = time.time() - CACHE_DAYS * 86400
    for p in [*CACHE_DIR.glob("*.pkl"), *CACHE_DIR.glob("director/*.json")]:
        if p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


_cleanup()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    with _lock:
        running = sum(1 for j in _jobs.values() if j.get("status") in ("queued", "running"))
    return {"status": "ok", "engine": ENGINE_VERSION, "running": running, "workers": WORKERS,
            "director": director.model_id() if director.configured() else None}


@app.post("/analyses", status_code=202)
def create_analysis(req: AnalysisRequest, x_internal_token: str | None = Header(default=None)) -> dict:
    _auth(x_internal_token)
    if req.quality not in QUALITY:
        raise HTTPException(400, f"quality must be one of {sorted(QUALITY)}")
    _check_url(req.videoUrl)
    analysis_id = _analysis_id(req)
    with _lock:
        existing = _jobs.get(analysis_id)
    if existing and existing.get("status") in ("queued", "running", "done"):
        return {"id": analysis_id, "status": existing["status"]}
    if _path(analysis_id).exists():
        _set(analysis_id, status="done", stage="done", stageLabel="Done", progress=1.0)
        return {"id": analysis_id, "status": "done"}
    _set(analysis_id, status="queued", stage="queued", stageLabel="Waiting to start", progress=0.0, error=None)
    executor.submit(_run, analysis_id, req)
    return {"id": analysis_id, "status": "queued"}


@app.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, x_internal_token: str | None = Header(default=None)) -> dict:
    _auth(x_internal_token)
    with _lock:
        job = dict(_jobs.get(analysis_id, {}))
    if not job:
        if _path(analysis_id).exists():
            job = {"id": analysis_id, "status": "done", "stage": "done", "stageLabel": "Done", "progress": 1.0}
        else:
            raise HTTPException(404, "Unknown analysis (it may have expired; submit it again)")
    if job.get("status") == "done":
        result = _load(analysis_id)
        if result is not None:
            job["text"] = [{"id": t.id, "kind": t.kind, "region": t.region, "description": t.describe(),
                            "onScreen": t.active, "canKeep": t.keyable} for t in result.text]
            job["duration"] = result.info.duration
    return job


@app.post("/analyses/{analysis_id}/plan")
def plan_analysis(analysis_id: str, req: PlanRequest, x_internal_token: str | None = Header(default=None)) -> dict:
    _auth(x_internal_token)
    if req.zoom not in ZOOM:
        raise HTTPException(400, f"zoom must be one of {sorted(ZOOM)}")
    if req.outputHeight not in OUTPUT_HEIGHTS:
        raise HTTPException(400, f"outputHeight must be one of {list(OUTPUT_HEIGHTS)}")
    try:
        a, b = (float(v) for v in req.ratio.split(":"))
        if a <= 0 or b <= 0:
            raise ValueError
    except ValueError:
        raise HTTPException(400, "ratio must look like 9:16") from None
    result = _load(analysis_id)
    if result is None:
        raise HTTPException(409, "The analysis isn't finished (or has expired)")
    options = Options(quality=result.quality, output_height=req.outputHeight, zoom=req.zoom)
    summary = _directed(analysis_id, result, req.ratio, options) if req.director and director.configured() else None
    return plan(result, req.ratio, keep_text=req.keepText, options=options,
                choices=summary["choices"] if summary else None,
                director={k: v for k, v in summary.items() if k != "choices"} if summary else None)
