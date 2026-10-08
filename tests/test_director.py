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
    monkeypatch.setattr(director, "SHOTS_PER_SHEET", 100)   # one request: deterministic

    def fake(sheets, shot_list, deadline=None):
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
    setup = next(s["setup"] for s in directed["shots"] if s["shot"] == shot["shot"].index)
    # The pick applies to the whole setup (same person and camera position), and only to it.
    assert {s["shot"] for s in changed} == {s["shot"] for s in directed["shots"] if s["setup"] == setup}
    assert shot["shot"].index in {s["shot"] for s in changed}
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


class _FakeModels:
    """Stands in for client.models: replies (or raises) in order."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        from google.genai import types
        return types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason="STOP", content=types.Content(role="model", parts=[types.Part(text=outcome)]))])


def _gemini(monkeypatch, outcomes):
    models = _FakeModels(outcomes)
    monkeypatch.setenv("DIRECTOR_PROVIDER", "gemini")
    monkeypatch.setenv("DIRECTOR_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(director, "_gemini_client", lambda: type("C", (), {"models": models})())
    monkeypatch.setattr(director.time, "sleep", lambda s: None)
    monkeypatch.setattr(director, "SHOTS_PER_SHEET", 100)   # one request: deterministic
    return models


def _api_error(code):
    from google.genai import errors
    return (errors.ServerError if code >= 500 else errors.ClientError)(code, {"error": {"code": code, "message": "x"}})


def test_gemini_sends_images_with_a_reply_schema(analysis, options, monkeypatch):
    models = _gemini(monkeypatch, ['{"shots": []}'])
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert summary["used"] and summary["model"] == "gemini-3.8-flash"
    call = models.calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].response_mime_type == "application/json" and call["config"].response_json_schema
    images = [c for c in call["contents"] if not isinstance(c, str)]
    assert images and all(c.inline_data.mime_type == "image/jpeg" for c in images)


def test_gemini_retries_rate_limits_and_server_errors(analysis, options, monkeypatch):
    models = _gemini(monkeypatch, [_api_error(429), _api_error(503), '{"shots": []}'])
    assert director.direct(analysis, options, parse_ratio("2:3"))["used"]
    assert len(models.calls) == 3


@pytest.mark.parametrize("outcomes,expect", [
    ([_api_error(400)], "ClientError"),                                   # not retried
    ([_api_error(429)] * 4, "Retryable"),                                 # retries run out
    (["not json at all"], "ValueError"),                                  # invalid output
    (['{"shots": [{"shot": "x"}]}'], "ValueError"),                       # wrong shape
])
def test_gemini_failures_keep_the_rules_and_are_logged(analysis, options, monkeypatch, caplog, outcomes, expect):
    models = _gemini(monkeypatch, outcomes)
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert not summary["used"] and summary["choices"] == {} and expect in summary["error"]
    assert "keeping the rules' framing" in caplog.text
    assert len(models.calls) == (4 if expect == "Retryable" else 1)


def test_gemini_timeout_keeps_the_rules(analysis, options, monkeypatch):
    import httpx
    _gemini(monkeypatch, [httpx.ReadTimeout("timed out")])
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert not summary["used"] and "ReadTimeout" in summary["error"]


def test_a_failed_sheet_keeps_the_rules_for_its_shots_only(analysis, options, monkeypatch):
    """Sheets are separate requests: one failing doesn't lose the others' picks."""
    multi = [o["shot"].index + 1 for o in options if len(o["options"]) > 1]
    monkeypatch.setattr(director, "SHOTS_PER_SHEET", 2)

    def fake(sheets, shot_list, deadline=None):
        first = int(shot_list.split(":")[0].split()[-1])
        if first == multi[0]:
            raise RuntimeError("Gemini HTTP 504")
        return json.dumps({"shots": [{"shot": n, "choice": "B", "confidence": 0.9, "reason": "x"}
                                     for n in multi[2:]]})

    monkeypatch.setattr(director, "_ask", fake)
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert summary["used"] and "504" in summary["error"]
    assert not {multi[0] - 1, multi[1] - 1} & set(summary["choices"])       # first sheet: rules
    assert summary["choices"]                                               # the others: picked


def test_retries_stop_at_the_budget(analysis, options, monkeypatch):
    models = _gemini(monkeypatch, [_api_error(503)] * 10)
    monkeypatch.setenv("DIRECTOR_BUDGET_S", "6")      # backoff (>=1.1 s) + 5 s margin won't fit
    summary = director.direct(analysis, options, parse_ratio("2:3"))
    assert not summary["used"] and len(models.calls) == 1
