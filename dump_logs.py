"""Dump current log buffer for each job in parallel with a per-job timeout."""
import threading
import time
from pathlib import Path
from huggingface_hub import HfApi

TIMEOUT_S = 25

JOBS = []
p = Path("active_jobs.txt")
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            label, jid = line.strip().split("\t", 1)
            JOBS.append((label, jid))


def dump(label: str, jid: str) -> None:
    api = HfApi()
    path = f"{label}_logs.txt"
    f = open(path, "w", encoding="utf-8")
    deadline = time.time() + TIMEOUT_S
    try:
        for ln in api.fetch_job_logs(job_id=jid, namespace="jacklachan"):
            f.write(ln)
            if time.time() > deadline:
                break
    except Exception as exc:
        f.write(f"\n[dump_logs] error: {exc}\n")
    finally:
        f.close()
    print(f"{label} -> {path}")


threads = [threading.Thread(target=dump, args=(l, j), daemon=True) for l, j in JOBS]
for t in threads:
    t.start()
for t in threads:
    t.join(TIMEOUT_S + 5)
print("done")
