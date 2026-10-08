"""AI director: choices, fallbacks and contact sheets, with the model call
replaced by canned replies (no network)."""

import json
import os
from pathlib import Path

import pytest

from reframe import director
from reframe.engine import analyze, parse_ratio, plan, shot_options

ROOT = Path(__file__).resolve().parent.parent
VIDEO = os.environ.get("CLIP001", str(ROOT / "testdata" / "clip001.mp4"))
WORDS = json.loads((ROOT / "testdata" / "clip001.transcript.json").read_text())["words"]

pytestmark = pytest.mark.skipif(not Path(VIDEO).exists(), reason="clip001.mp4 not available")


@pytest.fixture(scope="module")
def analysis():
    return analyze(VIDEO, WORDS)


@pytest.fixture
def options(analysis):
    return shot_options(analysis, "2:3")


def _reply(monkeypatch, text=None, error=None):
    seen = {}

    def fake(sheets, shot_list):
        seen["sheets"], seen["list"] = sheets, shot_list
        if error:
            raise error
        return text

    monkeypatch.setattr(director, "_ask", fake)
    return seen


def test_every_shot_has_the_rules_choice_first(options):
    assert all(list(o["options"])[0] == "rules" for o in options)
    assert any(len(o["options"]) > 1 for o in options)


def test_a_confident_pick_changes_that_shot_only(analysis, options, monkeypatch):
    shot = next(o for o in options if len(o["options"]) > 1)
    seen = _reply(monkeypatch, 'Sure: {"shots": [{"shot": %d, "choice": "B", "confidence": 0.9, "reason": "bigger"}]}'
                  % (shot["shot"].index + 1))
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert summary["used"] and summary["choices"] == {shot["shot"].index: list(shot["options"])[1]}
    assert seen["sheets"] and all(s[:2] == b"\xff\xd8" for s in seen["sheets"])   # JPEGs

    rules = plan(analysis, "2:3")
    directed = plan(analysis, "2:3", choices=summary["choices"], director=summary)
    changed = [s for s in directed["shots"] if s["choice"] != "rules"]
    assert [s["shot"] for s in changed] == [shot["shot"].index]
    assert directed["director"]["used"] and rules["director"] == {"used": False}


def test_low_confidence_and_unknown_letters_keep_the_rules(analysis, options, monkeypatch):
    multi = [o["shot"].index + 1 for o in options if len(o["options"]) > 1][:2]
    _reply(monkeypatch, json.dumps({"shots": [
        {"shot": multi[0], "choice": "B", "confidence": 0.3, "reason": "unsure"},
        {"shot": multi[-1], "choice": "Z", "confidence": 0.9, "reason": "no such option"},
    ]}))
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert summary["used"] and summary["choices"] == {}
    assert {n["choice"] for n in summary["notes"]} == {"rules"}


@pytest.mark.parametrize("text,error", [("no json here", None), (None, RuntimeError("the model declined"))])
def test_failures_fall_back_to_the_rules(analysis, options, monkeypatch, text, error):
    _reply(monkeypatch, text, error)
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert not summary["used"] and summary["choices"] == {} and "error" in summary


def test_converse_request_shape(analysis, options, monkeypatch):
    """The Bedrock Converse path sends every sheet as an image and reads the text back."""
    import boto3

    sent = {}

    class FakeRuntime:
        def converse(self, **kwargs):
            sent.update(kwargs)
            return {"stopReason": "end_turn", "output": {"message": {"content": [{"text": '{"shots": []}'}]}}}

    monkeypatch.setenv("REFRAME_DIRECTOR", "converse")
    monkeypatch.setenv("REFRAME_DIRECTOR_MODEL", "amazon.nova-pro-v1:0")
    monkeypatch.setattr(boto3, "client", lambda *a, **k: FakeRuntime())
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert summary["used"] and summary["model"] == "amazon.nova-pro-v1:0"
    images = [c for c in sent["messages"][0]["content"] if "image" in c]
    assert images and all(c["image"]["source"]["bytes"][:2] == b"\xff\xd8" for c in images)


def test_gemini_request_shape(analysis, options, monkeypatch):
    """The Gemini path sends the sheets inline with a reply schema, and reports HTTP errors."""
    import httpx

    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, body=json)
        reply = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"shots": []}'}]}}]}
        return httpx.Response(200, json=reply)

    monkeypatch.setenv("REFRAME_DIRECTOR", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "post", fake_post)
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert summary["used"] and sent["url"].endswith(f"/models/{director.model_id()}:generateContent")
    assert sent["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert any("inline_data" in p for p in sent["body"]["contents"][0]["parts"])

    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(402, json={"error": {"message": "credits depleted"}}))
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert not summary["used"] and "402" in summary["error"]
