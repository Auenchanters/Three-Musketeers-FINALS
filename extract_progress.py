"""Show key milestones from a long single-line log."""
import sys
import re

path = sys.argv[1] if len(sys.argv) > 1 else "Qwen7B_logs.txt"
text = open(path, encoding="utf-8", errors="ignore").read()
size = len(text)

MARKERS = [
    ("install done",  r"installing requirements-train|installed forgiving JSON parser"),
    ("device",        r"device=NVIDIA[^[]*"),
    ("collect",       r"\[hf-job\] \[1/4\][^[]*"),
    ("baselines",     r"REWARD GAP[^[]*"),
    ("sft start",     r"\[hf-job\] \[3/4\][^[]*"),
    ("sft step",      r"(\d+%)\|[^|]+\| (\d+/90)"),
    ("sft saved",     r"SFT model saved[^[]*"),
    ("adapter push",  r"pushed sft_output[^[]*"),
    ("eval start",    r"\[hf-job\] \[4/4\][^[]*"),
    ("eval ep",       r"\[eval\] ep [^]]*?\([^)]+\)"),
    ("eval mean",     r"\[eval\] (?:trained|base) mean[^[]*"),
    ("eval by diff",  r"\[eval\] (?:trained|base) by difficulty[^[]*"),
    ("eval wrote",    r"\[eval\] wrote[^[]*"),
    ("done",          r"\[hf-job\] DONE in[^[]*"),
    ("error",         r"(?:Traceback|Error:|WARN[^:]*:)[^[]{0,200}"),
]
print(f"log size: {size:,} chars")
for label, pat in MARKERS:
    matches = list(re.finditer(pat, text))
    if not matches:
        continue
    if label in ("eval ep", "sft step"):
        for m in matches:
            print(f"  {label}: {m.group(0).strip()}")
    else:
        last = matches[-1]
        print(f"  {label}: {last.group(0).strip()[:200]}")
