"""Probe HF account state: token role, jobs history, billing/usage info."""
from __future__ import annotations

import os
from pathlib import Path
import requests
from huggingface_hub import HfApi


def _token() -> str:
    if t := os.environ.get("HF_TOKEN"):
        return t.strip()
    cache = Path.home() / ".cache" / "huggingface" / "token"
    return cache.read_text().strip()


def main() -> None:
    api = HfApi()
    tok = _token()

    print("--- token role ---")
    try:
        from huggingface_hub.utils import get_token_permission
        print("role:", get_token_permission())
    except Exception as exc:
        print("role probe failed:", exc)

    print("\n--- jobs listing (recent) ---")
    try:
        jobs = list(api.list_jobs())[:10]
        if not jobs:
            print("(no prior jobs)")
        for j in jobs:
            print(j)
    except Exception as exc:
        print("list_jobs error:", exc)

    print("\n--- whoami-v2 (full) ---")
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.get("https://huggingface.co/api/whoami-v2", headers=h, timeout=15)
    print("status:", r.status_code)
    if r.status_code == 200:
        j = r.json()
        for k in ("name", "type", "plan", "canPay", "periodEnd", "usage", "isPro"):
            if k in j:
                print(f"  {k}: {j[k]}")

    print("\n--- billing usage (best-effort) ---")
    for endpoint in [
        "https://huggingface.co/api/users/Auenchanters/billing",
        "https://huggingface.co/api/billing/usage",
        "https://huggingface.co/api/jobs/quota",
    ]:
        try:
            r = requests.get(endpoint, headers=h, timeout=15)
            print(f"  {endpoint} -> {r.status_code}")
            if r.status_code == 200:
                print("   body:", r.text[:600])
        except Exception as exc:
            print(f"  {endpoint} -> ERROR {exc}")


if __name__ == "__main__":
    main()
