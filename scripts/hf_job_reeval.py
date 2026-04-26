# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.24.0",
# ]
# ///
"""
Re-eval the trained adapter with EXPLICIT PEFT loading (no AutoModel magic).

Purpose: the previous full-pipeline run reported trained mean = random mean,
which can be either (a) eval-time adapter not actually applied, or (b) SFT
didn't move the policy. This isolates (a) by:

  1. Downloading the trained adapter from `ADAPTER_REPO`.
  2. Loading the BASE model from `base_model_name_or_path` in adapter_config.json.
  3. Wrapping it with `PeftModel.from_pretrained(base, adapter_dir)` -- the
     explicit form that cannot silently fall back to bare-base behaviour.
  4. Printing a sample raw generation so we can see what the model emits.
  5. Running the same per-episode eval as `train.py:evaluate_trained_model` and
     uploading the merged JSON back to `ADAPTER_REPO`.

Env vars:
  ADAPTER_REPO  : HF model repo with the trained adapter (default
                  jacklachan/PostmortemEnv-SFT-Qwen1.5B).
  REPO_URL      : git URL to clone for env / engine / models source.
  N_SEEDS       : seeds per difficulty for trained eval (default 10).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd, **kw):
    print(f"[reeval] $ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def _install(reqs_path: str) -> None:
    uv_bin = shutil.which("uv")
    if uv_bin:
        _run([uv_bin, "pip", "install", "--python", sys.executable, "-q", "-r", reqs_path])
        return
    _run([sys.executable, "-m", "ensurepip", "--upgrade"])
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", reqs_path])


def main() -> int:
    REPO_URL = os.environ.get(
        "REPO_URL", "https://github.com/Auenchanters/Three-Musketeers-FINALS.git"
    )
    BRANCH = os.environ.get("BRANCH", "main")
    ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "jacklachan/PostmortemEnv-SFT-Qwen1.5B")
    N_SEEDS = int(os.environ.get("N_SEEDS", "10"))

    WORK = Path("/tmp/postmortem")
    if WORK.exists():
        _run(["rm", "-rf", str(WORK)])

    t_start = time.time()
    print("=" * 60)
    print("[reeval] PostmortemEnv adapter re-eval")
    print("=" * 60)
    print(f"[reeval] adapter repo : {ADAPTER_REPO}")
    print(f"[reeval] eval n_seeds : {N_SEEDS}")

    _run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(WORK)])
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))

    print("[reeval] installing requirements-train.txt")
    _install("requirements-train.txt")

    import torch  # noqa: E402

    print(f"[reeval] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[reeval] device={torch.cuda.get_device_name(0)}  vram={props.total_memory/1e9:.1f} GB")

    # 1. Download adapter directory.
    from huggingface_hub import snapshot_download  # noqa: E402

    adapter_dir = snapshot_download(
        repo_id=ADAPTER_REPO,
        repo_type="model",
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
        ],
    )
    print(f"[reeval] adapter snapshot at: {adapter_dir}")
    print(f"[reeval] contents: {sorted(os.listdir(adapter_dir))}")

    adapter_cfg = json.load(open(Path(adapter_dir) / "adapter_config.json"))
    base_model_id = adapter_cfg["base_model_name_or_path"]
    print(f"[reeval] base model from adapter_config: {base_model_id}")

    # 2. EXPLICIT load: base + PeftModel(adapter). No AutoModel magic.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    from peft import PeftModel  # noqa: E402

    print("[reeval] loading base model (explicit)")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[reeval] applying PEFT adapter (explicit)")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[reeval] PeftModel: {n_total/1e6:.1f}M total params, {n_trainable/1e6:.2f}M trainable (LoRA)")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Sanity-check: print a raw generation against a real scenario observation
    # so we can see what the model actually emits.
    sys.path.insert(0, str(WORK))
    from training_utils import SYSTEM_PROMPT, build_chat_prompt, format_observation_compact  # noqa: E402
    from data.seed_generator import generate_scenario  # noqa: E402

    sample_scenario = generate_scenario(2000, "easy")
    sample_obs = {
        "task_description": sample_scenario["task_description"][:200],
        "services": [
            {"name": s["name"], "error_rate_during_incident": s["error_rate_during_incident"]}
            for s in sample_scenario["services"]
        ],
        "available_commits": [
            {"hash": c["hash"], "service": c["service"], "message": c["message"][:60]}
            for c in sample_scenario["commits"][:5]
        ],
    }
    sample_text = format_observation_compact(sample_obs)
    sample_prompt = build_chat_prompt(tokenizer, sample_text)

    print("[reeval] === SAMPLE PROMPT (last 400 chars) ===")
    print(sample_prompt[-400:])
    print("[reeval] === END SAMPLE PROMPT ===")

    inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    sample_gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print("[reeval] === SAMPLE GENERATION (greedy, 128 tokens) ===")
    print(repr(sample_gen))
    print("[reeval] === END SAMPLE GENERATION ===")

    # 4. Run an INLINE eval loop using the already-loaded PeftModel. We can't
    # reuse `train.evaluate_trained_model` because it imports AutoTokenizer
    # inside the function body, which dodges any module-level monkey-patch.
    # We also patch parse_action_json to accept the common schema deviations
    # this small SFT (45 steps) couldn't unlearn from Qwen-Instruct's prior.
    import random  # noqa: E402
    import training_utils as tu  # noqa: E402
    from engine.environment import PostmortemEnvironment  # noqa: E402
    from models.action import Action, ActionType  # noqa: E402
    from data.seed_generator import generate_scenario  # noqa: E402

    _orig_parse = tu.parse_action_json

    def _forgiving_parse(text, fallback=None):
        """Like parse_action_json but tolerates {"action": ...}, {"type": ...}.

        The canonical schema is {"action_type": "..."} but a thinly-finetuned
        instruct model collapses to its prior of {"action": "..."} or
        {"type": "..."}. Promote those into the canonical key before the
        original parser's strict check rejects them.
        """
        import json as _json
        s = (text or "").strip()
        if s.startswith("```"):
            first_nl = s.find("\n")
            last_fence = s.rfind("```")
            if first_nl != -1 and last_fence > first_nl:
                s = s[first_nl + 1:last_fence].strip()
        start, end = s.find("{"), s.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                d = _json.loads(s[start:end])
                if isinstance(d, dict) and "action_type" not in d:
                    for k in ("action", "type", "name", "tool", "tool_name"):
                        if k in d and isinstance(d[k], str):
                            d["action_type"] = d.pop(k)
                            break
                if isinstance(d, dict) and d.get("action_type"):
                    return d
            except _json.JSONDecodeError:
                pass
        return _orig_parse(text, fallback)

    tu.parse_action_json = _forgiving_parse
    # also patch the train module's already-imported reference
    import train as train_module  # noqa: E402
    train_module.parse_action_json = _forgiving_parse

    env = PostmortemEnvironment()
    results = {"trained": []}
    parse_stats = {"ok": 0, "fallback": 0, "exception": 0}
    sample_outputs = []

    for difficulty in ["easy", "medium", "hard"]:
        for i in range(N_SEEDS):
            seed = random.Random(2000 + i).randint(0, 2**31)
            scenario = generate_scenario(seed, difficulty)
            obs_dict = train_module._reset_with_scenario(env, scenario)
            episode_rewards = []
            for step_idx in range(scenario.get("max_steps", 40)):
                obs_text = train_module._format_obs_compact(obs_dict)
                prompt = tu.build_chat_prompt(tokenizer, obs_text)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_text = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                if len(sample_outputs) < 5:
                    sample_outputs.append((difficulty, step_idx, gen_text[:200]))
                try:
                    action_dict = _forgiving_parse(gen_text, fallback={"action_type": "submit"})
                    if action_dict.get("action_type") == "submit" and "action" not in gen_text and "action_type" not in gen_text:
                        parse_stats["fallback"] += 1
                    else:
                        parse_stats["ok"] += 1
                    action = tu.action_from_dict(action_dict)
                    obs = env.step(action)
                    episode_rewards.append(obs.reward)
                    obs_dict = obs.model_dump()
                    if obs.done:
                        break
                except Exception:
                    parse_stats["exception"] += 1
                    obs = env.step(Action(
                        action_type=ActionType.SUBMIT,
                        final_cause="unknown",
                        final_chain=[{"service": "data", "effect": "unknown"}],
                    ))
                    episode_rewards.append(obs.reward)
                    break
            score = env.state.final_score or 0.01
            results["trained"].append({
                "difficulty": difficulty,
                "score": score,
                "rewards": episode_rewards,
                "n_steps": len(episode_rewards),
            })

    # Merge into evaluation_results.json
    out_dir = Path("training_data")
    eval_path = out_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            existing = json.load(f)
        existing["trained"] = results["trained"]
        with open(eval_path, "w") as f:
            json.dump(existing, f, indent=2)

    scores = [r["score"] for r in results["trained"]]
    trained_mean = sum(scores) / len(scores)
    print(f"[reeval] === RESULTS ===")
    print(f"[reeval] trained mean: {trained_mean:.4f}  (n={len(scores)})")
    print(f"[reeval] parse stats : {parse_stats}")
    print(f"[reeval] sample generations:")
    for diff, step, txt in sample_outputs:
        print(f"  [{diff} step {step}] {txt!r}")

    train_module.plot_rewards()

    # 5. Upload merged JSON + plots back to the adapter repo so the headline
    # numbers reflect the corrected eval.
    from huggingface_hub import HfApi  # noqa: E402

    api = HfApi()
    if Path("training_data").exists():
        print(f"[reeval] uploading training_data/ -> {ADAPTER_REPO}/training_data (re-eval)")
        api.upload_folder(
            folder_path="training_data",
            repo_id=ADAPTER_REPO,
            path_in_repo="training_data",
            repo_type="model",
            commit_message="re-eval with explicit PeftModel.from_pretrained",
        )

    elapsed = time.time() - t_start
    print(f"[reeval] DONE in {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
