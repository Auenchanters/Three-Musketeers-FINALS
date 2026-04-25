"""Launch fast verification + production runs on the new pluggable script.

Each job uses the new fast-eval defaults:
  EVAL_SEEDS=4         -> 12 episodes total
  EVAL_MAX_NEW_TOKENS=80
  EVAL_MAX_STEPS=20    -> ~3-5 min eval (vs the 40+ min before)
"""
import sys
from huggingface_hub import HfApi, get_token

TOKEN = get_token()  # use cached login from `hf auth login`
assert TOKEN, "no cached HF token; run `py -m huggingface_hub.cli.hf auth login` first"
api = HfApi(token=TOKEN)

LAUNCHES = {
    "Qwen7B": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "upload_repo": "jacklachan/PostmortemEnv-SFT-Qwen7B",
        "epochs": "6",
        "max_seq": "1536",
        "branch": "tanush/phase1-3-finalization",
    },
    "QwenCoder7B": {
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "upload_repo": "jacklachan/PostmortemEnv-SFT-QwenCoder7B",
        "epochs": "6",
        "max_seq": "1536",
        "branch": "tanush/phase1-3-finalization",
    },
    "Qwen14B": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "upload_repo": "jacklachan/PostmortemEnv-SFT-Qwen14B",
        "epochs": "4",
        "max_seq": "1536",
        "branch": "tanush/phase1-3-finalization",
    },
}


def launch(label: str) -> str:
    cfg = LAUNCHES[label]
    env = {
        "BASE_MODEL": cfg["base_model"],
        "UPLOAD_REPO": cfg["upload_repo"],
        "EPOCHS": cfg["epochs"],
        "MAX_SEQ": cfg["max_seq"],
        "BRANCH": cfg["branch"],
        "EVAL_SEEDS": "4",
        "EVAL_MAX_NEW_TOKENS": "80",
        "EVAL_MAX_STEPS": "20",
        "EVAL_BASE_TOO": "0",
    }
    job = api.run_uv_job(
        script="scripts/hf_job_train.py",
        flavor="a10g-small",
        env=env,
        secrets={"HF_TOKEN": TOKEN},
        namespace="jacklachan",
        token=TOKEN,
    )
    return job.id


if __name__ == "__main__":
    targets = sys.argv[1:] or list(LAUNCHES)
    out = []
    for t in targets:
        if t not in LAUNCHES:
            print(f"unknown: {t}; choices = {list(LAUNCHES)}")
            continue
        jid = launch(t)
        print(f"{t:14s} -> {jid}")
        out.append((t, jid))
    with open("active_jobs.txt", "w", encoding="utf-8") as f:
        for lbl, jid in out:
            f.write(f"{lbl}\t{jid}\n")
