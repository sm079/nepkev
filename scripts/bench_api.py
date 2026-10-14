"""Measure POST /route latency against a running API, using the held-out test messages of the browser demo.

    uv run python scripts/bench_api.py --url http://localhost:8000 --rounds 3

Requests are sent one at a time (no concurrency), so the numbers are per-request latency on an otherwise idle server.
Reports p50/p95 of the model time the API reports (latency_ms) and of the round trip seen by the client, and how many
predictions match the test labels (hard-labelled messages only).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

PRESETS = Path(__file__).resolve().parent.parent / "demo" / "data" / "presets.json"


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def post(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(f"{url}/route", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.load(resp)
    return out, (time.perf_counter() - started) * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rounds", type=int, default=3, help="passes over the messages (the first one also warms up)")
    args = ap.parse_args()

    presets = json.loads(PRESETS.read_text(encoding="utf-8"))
    post(args.url, {"text": presets[0]["message"], "channel": presets[0]["domain"]})   # warm-up, not counted
    model, total, hits, scored = [], [], 0, 0
    n = len(presets) * args.rounds
    for i in range(n):
        p = presets[i % len(presets)]
        out, ms = post(args.url, {"text": p["message"], "channel": p["domain"]})
        model.append(out["latency_ms"])
        total.append(ms)
        if i < len(presets) and p["issue"] is not None:
            scored += 1
            hits += out["intent"] == p["issue"]
        print(f"{i + 1}/{n}", flush=True)
    print(json.dumps({
        "requests": n, "model_version": out["model_version"],
        "model_ms": {"p50": round(percentile(model, 0.5)), "p95": round(percentile(model, 0.95)),
                     "mean": round(statistics.mean(model))},
        "round_trip_ms": {"p50": round(percentile(total, 0.5)), "p95": round(percentile(total, 0.95))},
        "issue_matches": f"{hits}/{scored}",
    }, indent=2))


if __name__ == "__main__":
    main()
