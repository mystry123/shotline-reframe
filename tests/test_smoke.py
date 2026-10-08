"""Checks that need no test video (they run in CI, where the clips aren't
available): the service starts, guards its routes and validates input."""

import importlib
import os

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ.update({"REFRAME_SERVICE_TOKEN": "smoke-token", "REFRAME_ALLOWED_HOSTS": "media.example.com",
                       "REFRAME_CACHE_DIR": str(tmp_path_factory.mktemp("cache"))})
    os.environ.pop("REFRAME_DEV", None)
    from fastapi.testclient import TestClient
    import reframe.service as service
    importlib.reload(service)
    return TestClient(service.app)


H = {"X-Internal-Token": "smoke-token"}


def test_health_needs_no_token(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["engine"]


def test_routes_need_the_token(client):
    assert client.get("/analyses/abc").status_code == 401
    assert client.get("/analyses/abc", headers={"X-Internal-Token": "wrong"}).status_code == 401


@pytest.mark.parametrize("url,code", [
    ("http://media.example.com/v.mp4", 400),      # https only outside dev
    ("https://evil.example.org/v.mp4", 400),      # host not allowed
    ("file:///etc/passwd", 400),
])
def test_video_urls_are_checked(client, url, code):
    assert client.post("/analyses", json={"videoUrl": url}, headers=H).status_code == code


def test_unknown_analysis_and_bad_plan_input(client):
    assert client.get("/analyses/" + "0" * 32, headers=H).status_code == 404
    assert client.post("/analyses/x/plan", json={"ratio": "wide"}, headers=H).status_code == 400
    assert client.post("/analyses/x/plan", json={"ratio": "9:16", "zoom": "huge"}, headers=H).status_code == 400
    assert client.post("/analyses/x/plan", json={"ratio": "9:16", "captions": "erase"}, headers=H).status_code == 400
