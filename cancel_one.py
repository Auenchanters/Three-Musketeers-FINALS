import sys
from huggingface_hub import HfApi
api = HfApi()
for jid in sys.argv[1:]:
    try:
        api.cancel_job(job_id=jid, namespace="jacklachan")
        print(f"cancelled {jid}")
    except Exception as exc:
        print(f"{jid}: {exc}")
