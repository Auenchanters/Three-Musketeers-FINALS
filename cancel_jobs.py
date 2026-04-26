"""Cancel all currently-running training jobs to recover budget."""
from huggingface_hub import HfApi

api = HfApi()
JOBS = {
    "7B":      "69ecc279d2c8bd8662bcdb1d",
    "Coder7B": "69ecc4ead2c8bd8662bcdb4c",
    "14B":     "69ecc4ecd2c8bd8662bcdb4e",
}
for label, jid in JOBS.items():
    try:
        api.cancel_job(job_id=jid, namespace="jacklachan")
        print(f"cancelled {label} ({jid})")
    except Exception as exc:
        print(f"{label} cancel failed: {exc}")
