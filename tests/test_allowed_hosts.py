"""REFRAME_ALLOWED_HOSTS: plain hosts, and host/prefix entries for shared hosts (GCS)."""
import os

import pytest


@pytest.fixture(scope="module")
def service():
    # The module refuses to import without a token; other tests reload it with their own env.
    os.environ.setdefault("REFRAME_SERVICE_TOKEN", "hosts-test-token")
    import reframe.service as service
    return service


@pytest.fixture(scope="module")
def allowed(service):
    return service._parse_allowed("media.example.com, storage.googleapis.com/shotline-media-prod/ ,")


@pytest.fixture
def ok(service, allowed):
    return lambda host, path: service._url_allowed(host, path, allowed)


def test_parses_plain_and_prefixed_entries(allowed):
    assert allowed == {("media.example.com", None), ("storage.googleapis.com", "shotline-media-prod")}


def test_plain_host_allows_any_path(ok):
    assert ok("media.example.com", "/a/b.mp4")
    assert ok("MEDIA.example.com", "/x.mp4")


def test_prefixed_host_allows_only_that_bucket(ok):
    assert ok("storage.googleapis.com", "/shotline-media-prod/uploads/u1/v.mp4")
    assert not ok("storage.googleapis.com", "/other-bucket/v.mp4")
    assert not ok("storage.googleapis.com", "/shotline-media-prod-evil/v.mp4")
    assert not ok("storage.googleapis.com", "/shotline-media-prod")
    assert not ok("evil.example.com", "/shotline-media-prod/v.mp4")


def test_no_escaping_the_prefix(ok):
    assert not ok("storage.googleapis.com", "/shotline-media-prod/../other/v.mp4")
    assert not ok("storage.googleapis.com", "/shotline-media-prod/%2e%2e/other/v.mp4")
    assert not ok("storage.googleapis.com", "/shotline-media-prod%2F..%2Fother/v.mp4")
