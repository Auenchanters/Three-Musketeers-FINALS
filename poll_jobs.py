"""Poll all jobs listed in active_jobs.txt with elapsed time."""
import datetime as dt
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()

JOBS = []
p = Path("active_jobs.txt")
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            label, jid = line.strip().split("\t", 1)
            JOBS.append((label, jid))

now = dt.datetime.now(dt.timezone.utc)
total_cost = 0.0
RATE_PER_HR = 1.25  # rough A10G/A100 SXM cost
for label, jid in JOBS:
    try:
        j = api.inspect_job(job_id=jid, namespace="jacklachan")
    except Exception as exc:
        print(f"{label:14s} {jid[:10]} INSPECT_FAIL {exc}")
        continue
    started = getattr(j, "created_at", None) or now
    if isinstance(started, str):
        started = dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
    elapsed = (now - started).total_seconds()
    cost = (elapsed / 3600.0) * RATE_PER_HR
    if j.status.stage == "RUNNING":
        total_cost += cost
    msg = getattr(j.status, "message", None) or ""
    print(f"{label:14s} {jid[:10]} {j.status.stage:9s}  {int(elapsed//60):3d}m{int(elapsed%60):02d}s  ~${cost:.2f}  {msg}")
print(f"running-cost (running jobs only): ~${total_cost:.2f}")
