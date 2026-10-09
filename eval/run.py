"""Reframe evaluation: analyse each clip, plan every target shape, measure,
compare with the committed baseline.

    python -m eval.run                    # synthetic clips + the real clips present in testdata/
    python -m eval.run --synthetic-only   # what CI runs (no video in the repo)
    python -m eval.run --update-baseline  # accept the current numbers as the new baseline

Real clips are testdata/<name>.mp4 with labels in eval/labels/<name>.json
(the videos stay out of git). The AI director is off: the evaluation measures
the engine itself, deterministically.

Exit code 1 when a gate fails (see GATES).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from reframe.engine import analyze, parse_ratio, plan
from reframe.options import Options

from . import synthetic
from .metrics import camera_metrics, picture_iou, plan_metrics, transition_f1

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BASELINE = ROOT / "baseline.json"
RESULTS = ROOT / ".cache" / "results.json"
RATIOS = ["16:9", "9:16", "1:1", "2:3"]

# metric: (direction that is worse, tolerance, kind). "abs": an absolute change;
# "rel": a relative change, ignored below an absolute floor (noise).
GATES = {
    "face_cut_rate": ("up", 0.005, "abs"),
    "caption_cut_rate": ("up", 0.005, "abs"),
    "person_kept": ("down", 0.02, "abs"),
    "jumps": ("up", 0, "abs"),
    "jerk_p95": ("up", 0.20, "rel"),
    "reversals_per_min": ("up", 0.20, "rel"),
    "transition_f1": ("down", 0.02, "abs"),
    "picture_iou": ("down", 0.02, "abs"),
    "camera_error": ("up", 0.05, "abs"),
}
REL_FLOOR = {"jerk_p95": 0.5, "reversals_per_min": 1.0}
ZOOM_WARN = 2.0


def _clips(synthetic_only: bool) -> list[tuple[str, Path, dict, list | None]]:
    """(name, video, truth, transcript words)."""
    out = [(f"syn/{n}", synthetic.ensure(n), synthetic.truth(n), None) for n in synthetic.CLIPS]
    if synthetic_only:
        return out
    out += [(f"local/{n}", synthetic.ensure_local(n), synthetic.LOCAL[n]["truth"], synthetic.local_words(n))
            for n in synthetic.local_available()]
    for label in sorted((ROOT / "labels").glob("*.json")):
        video = REPO / "testdata" / f"{label.stem}.mp4"
        if not video.exists():
            continue
        words_file = REPO / "testdata" / f"{label.stem}.transcript.json"
        words = json.loads(words_file.read_text())["words"] if words_file.exists() else None
        out.append((label.stem, video, json.loads(label.read_text()), words))
    return out


def evaluate(synthetic_only: bool = False) -> dict:
    results: dict = {"clips": {}}
    for name, video, truth, words in _clips(synthetic_only):
        started = time.perf_counter()
        a = analyze(str(video), words)
        seconds = time.perf_counter() - started
        entry: dict = {"analysis": {"seconds": round(seconds, 1), "duration": round(a.info.duration, 2),
                                    "picture_iou": picture_iou(a, truth["picture"])}, "plans": {}}
        if truth.get("transitions") is not None:
            entry["analysis"].update(transition_f1(a, truth["transitions"]))
        entry["analysis"].update(camera_metrics(a, truth.get("camera", {})))
        pic_aspect = a.area.w / a.area.h
        for ratio in RATIOS:
            if abs(parse_ratio(ratio) / pic_aspect - 1) < 0.03:
                continue
            entry["plans"][ratio] = plan_metrics(a, plan(a, ratio, options=Options(quality=a.quality)))
        results["clips"][name] = entry
        print(f"  {name}: analysed {a.info.duration:.0f}s of video in {seconds:.1f}s", file=sys.stderr)
    return results


def _flat(results: dict) -> dict[str, float]:
    flat = {}
    for clip, entry in results["clips"].items():
        for k, v in entry["analysis"].items():
            if k in GATES and v is not None:
                flat[f"{clip}|{k}"] = v
        for ratio, metrics in entry["plans"].items():
            for k, v in metrics.items():
                if k in GATES and v is not None:
                    flat[f"{clip}|{ratio}|{k}"] = v
    return flat


def compare(current: dict, baseline: dict) -> list[str]:
    """Gate failures, as lines."""
    now, before = _flat(current), _flat(baseline)
    failures = []
    for key, value in now.items():
        if key not in before:
            continue
        metric = key.rsplit("|", 1)[1]
        worse_dir, tol, kind = GATES[metric]
        old = before[key]
        delta = value - old if worse_dir == "up" else old - value
        if kind == "rel":
            if max(value, old) < REL_FLOOR.get(metric, 0):
                continue
            delta = delta / max(abs(old), REL_FLOOR.get(metric, 1e-6))
        if delta > tol + 1e-9:
            failures.append(f"{key}: {old} -> {value}")
    return failures


def table(results: dict) -> str:
    cols = ["face_cut_rate", "caption_cut_rate", "person_kept", "jumps", "jerk_p95", "reversals_per_min", "zoom_p95"]
    lines = [f"{'clip':22} {'shape':6} " + " ".join(f"{c[:12]:>12}" for c in cols)]
    for clip, entry in results["clips"].items():
        a = entry["analysis"]
        extra = f"picture_iou={a['picture_iou']}" + (f" transition_f1={a['transition_f1']}" if "transition_f1" in a else "") \
            + (f" camera_error={a['camera_error']}" if "camera_error" in a else "")
        lines.append(f"{clip:22} {'':6} {extra}")
        for ratio, m in entry["plans"].items():
            cells = []
            for c in cols:
                v = m.get(c)
                warn = c == "zoom_p95" and v is not None and v > ZOOM_WARN
                cells.append(f"{'-' if v is None else v:>11}{'!' if warn else ' '}")
            lines.append(f"{'':22} {ratio:6} " + " ".join(cells))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    results = evaluate(args.synthetic_only)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    print(table(results))

    if args.update_baseline:
        merged = json.loads(BASELINE.read_text()) if BASELINE.exists() else {"clips": {}}
        merged["clips"].update(results["clips"])
        BASELINE.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
        print(f"\nBaseline updated ({len(results['clips'])} clips).")
        return 0
    if not BASELINE.exists():
        print("\nNo baseline yet: run with --update-baseline.")
        return 0
    failures = compare(results, json.loads(BASELINE.read_text()))
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll gates pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
