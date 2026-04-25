"""Sync the GitHub main branch to the HF Space at Auenchanters/postmortemenv.

Uses the HF Hub Python API (which auto-handles Xet/LFS) instead of `git push`
because the Space repo rejects in-band binary files via the pre-receive hook.

Strategy
--------
* Upload the entire repo as a single commit via ``upload_folder``.
* Skip ephemeral / generated artefacts via ``ignore_patterns``.
* Skip the previously-deleted ``sft_output/final/*`` weights (deleted in commit
  ``cbce426 Cleanup unused files and make structure neat``).
* Auto-track binaries (``*.png``, ``*.safetensors``) via Xet on the Space side.
* Print the commit URL so the deployment can be verified manually.

Run with: ``py -3.10 sync_to_hf_space.py``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ID = "Auenchanters/postmortemenv"
REPO_TYPE = "space"

# Files / globs that should never go to the Space repo
IGNORE = [
    # VCS + tooling
    ".git/**",
    ".github/**",
    ".gitignore",
    ".cursor/**",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    # Local-only utility scripts (now in main but not deployed code)
    "dump_logs.py",
    "extract_eval.py",
    "extract_progress.py",
    "launch_fast.py",
    "poll_jobs.py",
    "sync_to_hf_space.py",
    "smoke_train_stream.py",
    "smoke_full_pipeline.py",
    # Per-run logs (huge, ephemeral)
    "training_data/runs/**",
    # Local backups
    "*.local-backup",
    # Notebook checkpoints
    "**/.ipynb_checkpoints/**",
    # Anything in sft_output already deleted in main
    "sft_output/**",
    # Old generated scenario data deleted upstream
    "data/generated/**",
    # Local hf token cache shouldn't be uploaded (defensive)
    "**/token",
]

# Documentation files that the README references — keep ALL md/txt for context
ALLOW_BIG = {".png", ".jpg", ".jpeg", ".safetensors"}


def repo_size_estimate(root: Path) -> tuple[int, int, int]:
    """(file_count, total_bytes, big_file_count) for human-friendly logging."""
    files = total = big = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel.startswith(".git/") or "__pycache__" in rel or ".cursor/" in rel:
            continue
        files += 1
        sz = p.stat().st_size
        total += sz
        if p.suffix.lower() in ALLOW_BIG:
            big += 1
    return files, total, big


def main() -> int:
    api = HfApi()
    me = whoami()
    print(f"[hf] authenticated as: {me['name']}")

    root = Path(__file__).resolve().parent
    files, total, big = repo_size_estimate(root)
    mb = total / 1024 / 1024
    print(f"[hf] uploading {files} files ({mb:.1f} MB, {big} binary tracked via Xet)")
    print(f"[hf] target: https://huggingface.co/spaces/{REPO_ID}")
    print(f"[hf] ignore patterns ({len(IGNORE)}): {IGNORE}")

    t0 = time.time()
    commit = api.upload_folder(
        folder_path=str(root),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=(
            "Sync GitHub main: hackathon finalization, training pipeline, "
            "research-lab UI, live training chart, 2 new tasks, 95% CI eval"
        ),
        ignore_patterns=IGNORE,
    )
    dt = time.time() - t0
    print(f"[hf] commit done in {dt:.1f}s")
    print(f"[hf] commit url: {commit.commit_url}")
    print(f"[hf] space url:  https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
