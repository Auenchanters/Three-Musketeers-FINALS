"""Scan each *_logs.txt for key milestones and print one-line summary per job."""
from pathlib import Path

FILES = ["7B_logs.txt", "Coder7B_logs.txt", "14B_logs.txt"]
MARKERS = [
    ("install_done", "torch=2.11.0+cu130  cuda=True"),
    ("baselines_done", "REWARD GAP"),
    ("sft_done", "SFT model saved to sft_output"),
    ("eval_phase", "[4/4] evaluate trained"),
    ("eval_load_base", "loading tokenizer + base model"),
    ("eval_apply_lora", "applying LoRA adapter"),
    ("eval_done", "[eval] trained mean"),
    ("upload", "[hf-job] uploading"),
    ("done", "[hf-job] DONE in"),
]
ERROR_KEYWORDS = ["Traceback", "OutOfMemory", "CUDA out of memory", "RuntimeError",
                  "ImportError", "ModuleNotFoundError", "FileNotFoundError",
                  "AssertionError", "ValueError"]

for f in FILES:
    p = Path(f)
    if not p.exists():
        print(f"{f:18s} (missing)")
        continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    hits = [name for name, mk in MARKERS if mk in text]
    errs = [k for k in ERROR_KEYWORDS if k in text]
    print(f"{f:18s} size={len(text)}  milestones={hits}  errs={errs}")
    if "[eval] trained mean" in text:
        idx = text.find("[eval] trained mean")
        print("   ->", text[idx:idx+250].replace("\n", " "))
    elif "applying LoRA adapter" in text:
        idx = text.rfind("applying LoRA adapter")
        tail = text[idx:idx+500].replace("\n", " ")
        print("   eval_tail:", tail[:300])
