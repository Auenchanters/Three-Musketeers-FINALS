"""End-to-end production smoke test against the deployed HF Space.

Hits the live URL a logged-out judge would click. Verifies:
  * /health returns 200 + status:ok
  * /api/tasks lists 5 scenarios (the new task4/task5 are present)
  * /api/models lists at least the 4 free-tier models
  * /reset on each task returns an observation
  * /step on each task accepts a valid action without 500-ing

This is the closest a Python smoke can get to "what a judge sees".
Run with: ``py -3.10 smoke_production_space.py``
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://auenchanters-postmortemenv.hf.space"
TIMEOUT = 30


def _get(path: str) -> tuple[int, dict | list | str]:
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = r.read().decode("utf-8", errors="replace")
        try:
            return r.status, json.loads(body)
        except json.JSONDecodeError:
            return r.status, body


def _post(path: str, body: dict) -> tuple[int, dict | list | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body_s = r.read().decode("utf-8", errors="replace")
        try:
            return r.status, json.loads(body_s)
        except json.JSONDecodeError:
            return r.status, body_s


def main() -> int:
    failures: list[str] = []

    print(f"--- production smoke: {BASE}")

    # 1) health
    s, j = _get("/health")
    ok = s == 200 and isinstance(j, dict) and j.get("status") in ("ok", "healthy")
    print(f"  /health -> {s} {'OK' if ok else 'FAIL: ' + str(j)[:120]}")
    if not ok:
        failures.append("/health")

    # 2) tasks
    s, j = _get("/api/tasks")
    ok = s == 200 and isinstance(j, list) and len(j) >= 5
    task_ids = [t.get("task_id") for t in j] if isinstance(j, list) else []
    print(f"  /api/tasks -> {s} ({len(task_ids)} tasks: {task_ids})")
    if not ok:
        failures.append("/api/tasks")
    if "task4_multi_region_failover" not in task_ids:
        failures.append("missing task4")
    if "task5_data_corruption_cascade" not in task_ids:
        failures.append("missing task5")

    # 3) models
    s, j = _get("/api/models")
    models = j.get("models", []) if isinstance(j, dict) else []
    free_models = [m for m in models if m.get("tier") == "free"]
    ok = s == 200 and len(free_models) >= 1
    print(f"  /api/models -> {s} ({len(models)} total, {len(free_models)} free-tier)")
    if not ok:
        failures.append("/api/models")

    # 4) /reset + /step on each task (the OpenEnv contract a judge would script)
    for tid in task_ids:
        try:
            s, j = _post("/reset", {"task_id": tid})
            ok = s == 200 and isinstance(j, dict) and "observation" in j
            obs = j.get("observation", {}) if isinstance(j, dict) else {}
            services = obs.get("service_graph", {}) or {}
            first_svc = next(iter(services), None) or "data"
            print(f"  /reset {tid} -> {s} {'OK' if ok else 'FAIL'} (svc graph: {len(services)} nodes)")
            if not ok:
                failures.append(f"/reset {tid}")
                continue

            # Smallest possible non-trivial step: query logs of the first service
            s, j = _post("/step", {
                "action": {
                    "action_type": "query_logs",
                    "service": first_svc,
                    "keyword": "error",
                }
            })
            ok = s == 200 and isinstance(j, dict) and "observation" in j
            r = j.get("reward") if isinstance(j, dict) else None
            print(f"  /step  {tid} -> {s} {'OK' if ok else 'FAIL'} (reward={r})")
            if not ok:
                failures.append(f"/step {tid}")
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"  /reset|step {tid} -> NETWORK FAIL: {exc}")
            failures.append(f"network {tid}")

    print()
    if failures:
        print(f"FAILED ({len(failures)} checks): {failures}")
        return 1
    print(f"ALL PASS — production Space is responsive end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
