"""Pull only the meaningful eval lines out of a single-line log file."""
import sys
import re

path = sys.argv[1] if len(sys.argv) > 1 else "Qwen7B_logs.txt"
text = open(path, encoding="utf-8", errors="ignore").read()

PATTERNS = [
    r"\[eval\] ep [^]]*?\([^)]+\)",
    r"\[eval\] trained mean[^[]+",
    r"\[eval\] base mean[^[]+",
    r"\[eval\] (?:trained|base) by difficulty[^[]+",
    r"\[hf-job\] pushed [^[]+",
    r"\[hf-job\] DONE in[^[]+",
    r"WARN[^[]+",
    r"Traceback[^[]+",
]
for p in PATTERNS:
    for m in re.finditer(p, text):
        print(m.group(0).strip())
