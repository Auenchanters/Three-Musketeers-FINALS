"""Launch an HF Jobs training run for PostmortemEnv.

Uses the HuggingFace Inference API / Jobs API to run the training
script on a GPU instance. Requires `huggingface_hub` and a valid token.
"""
import subprocess
import sys

# The exact command the README documents:
# hf jobs uv run --flavor l4x1 --secrets HF_TOKEN ...
# Since we don't have `hf` CLI on Windows, we'll use `huggingface-cli`
# or fall back to a direct API call.

def main():
    # Try to find the huggingface-cli or hf binary
    import shutil
    
    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    
    if not hf_bin:
        # Try the Python Scripts directory
        import os
        scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
        for name in ["hf.exe", "huggingface-cli.exe"]:
            candidate = os.path.join(scripts_dir, name)
            if os.path.exists(candidate):
                hf_bin = candidate
                break
    
    if not hf_bin:
        print("ERROR: Cannot find 'hf' or 'huggingface-cli' binary.")
        print()
        print("To install:")
        print("  pip install huggingface_hub[hf_xfer]")
        print("  # or")  
        print("  pipx install huggingface_hub")
        print()
        print("Alternatively, run this command manually from a terminal with `hf` installed:")
        print()
        print_manual_command()
        return 1
    
    print(f"Found HF CLI at: {hf_bin}")
    
    cmd = [
        hf_bin, "jobs", "uv", "run",
        "--flavor", "l4x1",
        "--secrets", "HF_TOKEN",
        "-e", "BASE_MODEL=Qwen/Qwen2.5-7B-Instruct",
        "-e", "COLLECT_SEEDS=50",
        "-e", "EPOCHS=5",
        "-e", "MAX_SEQ=4096",
        "-e", "EVAL_SEEDS=10",
        "-e", "EVAL_BASE_TOO=1",
        "-e", "EVAL_MAX_NEW_TOKENS=192",
        "-e", "EVAL_MAX_STEPS=30",
        "-e", "SMOKE=0",
        "-e", "UPLOAD_REPO=Auenchanters/postmortemenv-qwen7b-improved",
        "https://raw.githubusercontent.com/Auenchanters/Three-Musketeers-FINALS/main/scripts/hf_job_train.py",
    ]
    
    print("Launching:")
    print("  " + " ".join(cmd))
    print()
    
    result = subprocess.run(cmd)
    return result.returncode


def print_manual_command():
    print("""hf jobs uv run ^
  --flavor l4x1 ^
  --secrets HF_TOKEN ^
  -e BASE_MODEL=Qwen/Qwen2.5-7B-Instruct ^
  -e COLLECT_SEEDS=50 ^
  -e EPOCHS=5 ^
  -e MAX_SEQ=4096 ^
  -e EVAL_SEEDS=10 ^
  -e EVAL_BASE_TOO=1 ^
  -e EVAL_MAX_NEW_TOKENS=192 ^
  -e EVAL_MAX_STEPS=30 ^
  -e SMOKE=0 ^
  -e UPLOAD_REPO=Auenchanters/postmortemenv-qwen7b-improved ^
  https://raw.githubusercontent.com/Auenchanters/Three-Musketeers-FINALS/main/scripts/hf_job_train.py""")


if __name__ == "__main__":
    sys.exit(main())
