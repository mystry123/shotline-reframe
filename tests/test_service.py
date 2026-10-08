"""The HTTP service: auth, allowed hosts, async analysis with progress, caching,
planning. Serves clip 001 from a local web server."""

import functools
import http.server
import importlib
import os
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "testdata" / "clip001.mp4"
pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="clip001.mp4 not available")


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(CLIP.parent))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ.update({"REFRAME_DEV": "1", "REFRAME_SERVICE_TOKEN": "test-token",
                       "REFRAME_ALLOWED_HOSTS": "127.0.0.1", "REFRAME_CACHE_DIR": str(tmp_path_factory.mktemp("cache"))})
    from fastapi.testclient import TestClient
    import reframe.service as service
    importlib.reload(service)
    return TestClient(service.app)


H = {"X-Internal-Token": "test-token"}


def wait_done(client, analysis_id, timeout=120):
    seen = set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/analyses/{analysis_id}", headers=H).json()
        seen.add(job.get("stage"))
        if job["status"] in ("done", "failed"):
            return job, seen
        time.sleep(0.2)
    raise AssertionError("analysis didn't finish")


def test_requires_the_token_and_an_allowed_host(client, server):
    assert client.post("/analyses", json={"videoUrl": f"{server}/clip001.mp4"}).status_code == 401
    bad = client.post("/analyses", json={"videoUrl": "http://evil.example/x.mp4"}, headers=H)
    assert bad.status_code == 400 and "not allowed" in bad.json()["detail"]


def test_analysis_runs_in_the_background_and_plans_any_shape(client, server):
    url = f"{server}/clip001.mp4"
    created = client.post("/analyses", json={"videoUrl": url}, headers=H)
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    job, stages = wait_done(client, analysis_id)
    assert job["status"] == "done", job
    assert {"faces", "people"} & stages or job["progress"] == 1.0
    assert [t["id"] for t in job["text"]] == ["top-title", "bottom-captions"]

    # Same input again: same analysis, no rework.
    again = client.post("/analyses", json={"videoUrl": url}, headers=H).json()
    assert again == {"id": analysis_id, "status": "done"}

    for ratio in ("16:9", "1:1", "2:3"):
        t = time.time()
        r = client.post(f"/analyses/{analysis_id}/plan", json={"ratio": ratio, "keepText": ["bottom-captions"]}, headers=H)
        assert r.status_code == 200, r.text
        assert time.time() - t < 1.0
        body = r.json()
        assert body["target"] == ratio and body["segments"]
        assert [o["id"] for o in body["overlays"]] == ["bottom-captions"]


def test_plan_validates_its_inputs(client):
    assert client.post("/analyses/nope/plan", json={"ratio": "1:1"}, headers=H).status_code == 409
    assert client.post("/analyses/nope/plan", json={"ratio": "wide"}, headers=H).status_code == 400
    assert client.post("/analyses/nope/plan", json={"ratio": "1:1", "zoom": "huge"}, headers=H).status_code == 400
