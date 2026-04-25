"""Verify the live-training SSE stream works end-to-end against the deployed
HuggingFace Space. Starts a 200-episode session, subscribes to the stream,
and asserts that we get at least the baselines + final-done event with a
positive lift over the random baseline.

Why this matters
----------------
``smoke_production_space.py`` exercises the synchronous step/reset endpoints
but not the SSE training stream — which is the most visible feature for a
judge clicking around. This script catches deployment regressions where the
backend route exists but the SSE plumbing is broken (CORS, X-Accel-Buffering,
asyncio task gc, etc.) — every one of which would show up here as a hang or
a partial event sequence rather than a 500.

Run: ``python smoke_production_training.py``

Expected output (15s on a fresh Space):
    /api/training/start (task1) -> session=...
      baselines random=0.27 menu=58 eps=200
      ...metric stream...
      DONE   final=0.567  lift=+0.296
    PASS  live training is healthy in production
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx

BASE = "https://auenchanters-postmortemenv.hf.space"
TASK = "task1_recent_deploy"
EPISODES = 500          # UI default; ~15s on the Space at the local 30ms cadence


def _parse_sse(buf: bytes) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for raw_event in buf.split(b"\n\n"):
        block = raw_event.decode("utf-8", "replace")
        if not block.strip():
            continue
        ev_type = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                ev_type = line[7:].strip()
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        if not data_lines:
            continue
        try:
            obj = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        out.append((ev_type, obj))
    return out


def main() -> int:
    print(f"--- production live training smoke: {BASE}")

    with httpx.Client(timeout=30.0) as c:
        r = c.post(
            f"{BASE}/api/training/start",
            json={"task_id": TASK, "n_episodes": EPISODES},
        )
        if r.status_code != 200:
            print(f"  /api/training/start -> {r.status_code} {r.text[:200]}")
            return 1
        sess = r.json()
        sid = sess["session_id"]
        print(f"  /api/training/start ({TASK}) -> session={sid}  eps={sess['n_episodes']}")

    # Subscribe to the SSE stream — chunked transfer, no buffering.
    deadline = time.time() + 60.0
    saw_baselines = False
    saw_done = False
    final_lift: float | None = None
    metric_count = 0
    last_metric_ep = 0
    buf = b""

    with httpx.Client(timeout=60.0) as c:
        with c.stream("GET", f"{BASE}/api/training/stream/{sid}") as resp:
            if resp.status_code != 200:
                print(f"  /api/training/stream -> {resp.status_code}")
                return 1
            for chunk in resp.iter_bytes():
                buf += chunk
                # SSE events are delimited by a blank line — split greedily.
                while b"\n\n" in buf:
                    head, _, buf = buf.partition(b"\n\n")
                    head += b"\n\n"
                    for ev_type, obj in _parse_sse(head):
                        if ev_type == "baselines":
                            saw_baselines = True
                            print(
                                f"  baselines random={obj['random']:.3f} "
                                f"menu={obj['action_menu_size']} "
                                f"eps={obj['n_episodes']}"
                            )
                        elif ev_type == "metric":
                            metric_count += 1
                            last_metric_ep = obj["episode"]
                            if metric_count % 50 == 0:
                                print(
                                    f"    ep {last_metric_ep:>3}/{EPISODES}  "
                                    f"rolling={obj['rolling_mean']:.3f}  "
                                    f"lift={obj['lift_over_random']:+.3f}"
                                )
                        elif ev_type == "done":
                            saw_done = True
                            final_lift = obj["lift_over_random"]
                            print(
                                f"  DONE   final={obj['final_rolling_mean']:.3f}  "
                                f"lift={final_lift:+.3f}"
                            )
                        elif ev_type == "error":
                            print(f"  STREAM ERROR: {obj.get('message')}")
                            return 1
                        elif ev_type == "_eof":
                            break
                if saw_done or time.time() > deadline:
                    break

    if not saw_baselines:
        print("FAIL  never received baselines event")
        return 1
    if not saw_done:
        print(f"FAIL  stream ended without 'done' event ({metric_count} metrics, last ep {last_metric_ep})")
        return 1
    if final_lift is None or final_lift < 0.05:
        # Lift threshold — random baseline alone shouldn't beat itself by 0.05+.
        # If we land below, the policy isn't actually learning in production.
        print(f"FAIL  final lift {final_lift!r} below +0.05 threshold")
        return 1
    print("PASS  live training is healthy in production")
    return 0


if __name__ == "__main__":
    sys.exit(main())
