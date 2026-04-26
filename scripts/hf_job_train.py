# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "hf_transfer>=0.1.6",
# ]
# ///
"""
HF Jobs entrypoint: clone PostmortemEnv, run `train.py full`, upload artifacts.

Pluggable: any HF causal LM with a chat template works as ``BASE_MODEL`` (Qwen,
Llama, Mistral, Phi, Gemma, ...). The eval helper falls back to a Llama-style
template if a model exposes none.

Env vars
--------
REPO_URL              git URL to clone (default: the canonical GitHub repo)
BRANCH                git branch (default: main)
BASE_MODEL            HF model id for SFT base (default: Qwen/Qwen2.5-7B-Instruct).
                      Any chat-tuned causal LM works; use full HF id like
                      ``meta-llama/Llama-3.1-8B-Instruct`` etc.
EPOCHS                SFT epochs (default: 5 in FULL, 1 in SMOKE)
MAX_SEQ               max seq length for SFT (default: 4096)
COLLECT_SEEDS         demos per difficulty (default: 20 FULL / 2 SMOKE)
EVAL_SEEDS            eval episodes per difficulty (default: 4 FULL / 2 SMOKE)
EVAL_MAX_NEW_TOKENS   tokens generated per LM call during eval (default: 192)
EVAL_MAX_STEPS        cap on env steps per eval episode (default: 30)
EVAL_BASE_TOO         "1" -> also evaluate the bare BASE before LoRA (delta study).
EVAL_ONLY             "1" -> skip SFT, just benchmark BASE_MODEL (+optional
                      ADAPTER_REPO). Lets you plug in any externally-trained
                      RL policy (DPO/PPO/GRPO/SFT/...) without retraining.
ADAPTER_REPO          (only with EVAL_ONLY=1) HF repo containing a PEFT/LoRA
                      adapter to apply on top of BASE_MODEL.
ADAPTER_SUBFOLDER     (optional) subfolder inside ADAPTER_REPO for the adapter.
SMOKE                 "1" -> tiny pipeline (~5 min). "0" -> full pipeline.
UPLOAD_REPO           if set, push sft_output + training_data to this model repo.
                      The adapter is pushed RIGHT AFTER SFT (before eval) so a
                      slow eval phase can never lose the trained artefact.
GRPO                  "1" -> after SFT, also run train.py grpo (stretch).
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Force unbuffered stdout/stderr inside the container. HF Jobs streams the
# job's stdout to the dashboard, but Python defaults to block-buffered when
# stdout is not a TTY (which it never is inside a job container) — so a
# silent SFT phase looks "hung" for 20+ min when it's actually working.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

# Switch HF Hub to the high-throughput Rust transfer client. Cuts a fresh
# 7B model download (~15 GB) from ~12 min to ~2-3 min on an L4 container.
# DEFAULT: OFF. We've seen the Rust binary panic silently inside a uv-built
# Python 3.12 container ("Set HF_DEBUG=1 ..." with no further output and the
# job effectively hanging at the first download). The standard Python
# downloader is slower (~12 min for 7B) but observable. Set
# HF_HUB_ENABLE_HF_TRANSFER=1 in the job env to opt back in.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# Make sure transformers / accelerate / TRL print their own logs at INFO so
# the SFT phase is visible while it runs (dataset prep, token counts, ...).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_VERBOSITY", "info")
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "INFO")
# NOTE: do NOT set HF_DEBUG=1 globally here. Empirically that flag *itself*
# triggers a silent hang in `huggingface_hub` 0.36.x inside the uv-built
# python:3.12 container — the curl-style HEAD/GET trace prints fine, then the
# next chunk-streamed download appears to deadlock. Letting the library run in
# its default (quiet) mode is the path that actually completes (verified by the
# Qwen2.5-3B-Instruct full job, ID 69ed0982d2c8bd8662bce31b → COMPLETED in 24m).
# If you need to debug a future download, set HF_DEBUG=1 in the *job* env (not
# baked into the script) so the choice is per-run rather than implicit.


def _heartbeat(label: str, interval: float = 30.0) -> threading.Event:
    """Spawn a background thread that prints a heartbeat every ``interval``s.

    Returns the stop event — call ``event.set()`` to terminate the thread.
    Used to surface progress through silent phases (model download, dataset
    tokenization, first SFT optimizer step) so a stalled job is never
    confused with a slow but healthy one.
    """
    stop = threading.Event()
    t0 = time.time()

    def _beat() -> None:
        n = 0
        while not stop.wait(interval):
            n += 1
            print(
                f"[hf-job] heartbeat {label} t+{int(time.time() - t0)}s (#{n} alive)",
                flush=True,
            )

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return stop


def _run(cmd, **kw):
    """Run a subprocess, stream output, raise on non-zero."""
    print(f"[hf-job] $ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def _install(reqs_path: str) -> None:
    """Install a requirements file into THIS interpreter's environment.

    `uv run` puts us inside a script-scoped venv that ships no `pip`, so
    `python -m pip install ...` fails with "No module named pip". Use the
    `uv` binary itself with `--python sys.executable` to target the same
    interpreter; fall back to bootstrapping pip via `ensurepip` if `uv`
    is unavailable for any reason.
    """
    uv_bin = shutil.which("uv")
    if uv_bin:
        _run([uv_bin, "pip", "install", "--python", sys.executable, "-q", "-r", reqs_path])
        return
    _run([sys.executable, "-m", "ensurepip", "--upgrade"])
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", reqs_path])


def _eval_trained_with_peft(
    base_model_id: str,
    adapter_dir: str,
    n_seeds: int,
    parse_fn,
    max_new_tokens: int = 80,
    max_steps_per_ep: int = 20,
    is_base: bool = False,
) -> dict:
    """Evaluate a trained LoRA adapter (or the bare base) end-to-end.

    Designed for speed and to accept ANY HF causal LM:
      * uses `apply_chat_template` if available, else a generic Qwen-style fallback
      * stops generation as soon as the model emits a closing brace ``}`` so we
        never burn 128 tokens on a 30-token JSON action
      * caps each episode at ``max_steps_per_ep`` (default 20) so a model that
        refuses to SUBMIT can't drag eval out for 80 minutes
      * logs `[eval] ep i/N (diff=...) steps=... score=...` per episode so we
        can see progress live instead of staring at a blank screen
    Returns a dict with per-episode rows and aggregate stats.
    """
    import json
    import random
    import time as _time
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel
    from engine.environment import PostmortemEnvironment
    from models.action import Action, ActionType
    from data.seed_generator import generate_scenario
    from training_utils import SYSTEM_PROMPT, action_from_dict
    from train import _reset_with_scenario, _format_obs_compact

    def _chat_prompt(messages):
        """Render a chat history -> prompt string, with a Llama fallback."""
        apply = getattr(tok, "apply_chat_template", None)
        if apply is not None and getattr(tok, "chat_template", None):
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except (ValueError, TypeError):
                pass
        body = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return f"{body}\nASSISTANT:"

    print(f"[eval] loading tokenizer + base model ({base_model_id})", flush=True)
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if is_base or not Path(adapter_dir).exists():
        print(f"[eval] running BASE model only (no adapter) from {base_model_id}", flush=True)
        model = base
    else:
        print(f"[eval] applying LoRA adapter from {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    close_brace_id = tok.encode("}", add_special_tokens=False)[-1]

    class StopOnCloseBrace(StoppingCriteria):
        """End generation once the model closes its first JSON object."""

        def __call__(self, input_ids, scores, **kwargs):
            return bool(input_ids[0, -1].item() == close_brace_id)

    stoppers = StoppingCriteriaList([StopOnCloseBrace()])

    env = PostmortemEnvironment()
    rows = []
    parse_ok = 0
    parse_bad = 0
    total_steps = 0
    ep_idx = 0
    n_total = 3 * n_seeds
    t_eval_start = _time.time()

    for difficulty in ("easy", "medium", "hard"):
        for i in range(n_seeds):
            ep_idx += 1
            t_ep = _time.time()
            seed = random.Random(2000 + i).randint(0, 2**31)
            scenario = generate_scenario(seed, difficulty)
            obs_dict = _reset_with_scenario(env, scenario)
            ep_rewards = []
            cap = min(scenario.get("max_steps", max_steps_per_ep), max_steps_per_ep)
            ep_parse_ok = 0
            ep_parse_bad = 0
            # Maintain a CHAT HISTORY across the episode -- training data was
            # multi-turn (user obs / assistant action / user obs / ... / submit)
            # so eval must also be multi-turn or the model never figures out
            # when to stop investigating.
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            done = False
            for _ in range(cap):
                obs_text = _format_obs_compact(obs_dict)
                messages.append({"role": "user", "content": obs_text})
                prompt = _chat_prompt(messages)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                        stopping_criteria=stoppers,
                    )
                text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                messages.append({"role": "assistant", "content": text})
                total_steps += 1
                try:
                    action_dict = parse_fn(text, fallback={"action_type": "submit"})
                    parse_ok += 1
                    ep_parse_ok += 1
                    action = action_from_dict(action_dict)
                    obs = env.step(action)
                    ep_rewards.append(obs.reward)
                    obs_dict = obs.model_dump()
                    if obs.done:
                        done = True
                        break
                except Exception:
                    parse_bad += 1
                    ep_parse_bad += 1
                    obs = env.step(Action(
                        action_type=ActionType.SUBMIT,
                        final_cause="unknown",
                        final_chain=[{"service": "data", "effect": "unknown"}],
                    ))
                    ep_rewards.append(obs.reward)
                    done = True
                    break
            # If the model never SUBMITted within cap, force a submit so the
            # grader can return a real (probably low) score instead of the
            # 0.01 placeholder.
            if not done:
                try:
                    obs = env.step(Action(
                        action_type=ActionType.SUBMIT,
                        final_cause="unknown",
                        final_chain=[{"service": "data", "effect": "unknown"}],
                    ))
                    ep_rewards.append(obs.reward)
                except Exception:
                    pass
            score = env.state.final_score if env.state.final_score is not None else 0.01
            rows.append({
                "difficulty": difficulty,
                "score": score,
                "rewards": ep_rewards,
                "n_steps": len(ep_rewards),
                "submitted_naturally": done,
            })
            ep_dt = _time.time() - t_ep
            print(
                f"[eval] ep {ep_idx:2d}/{n_total:2d} diff={difficulty:6s} "
                f"steps={len(ep_rewards):2d} score={score:5.3f} "
                f"parse={ep_parse_ok}/{ep_parse_ok + ep_parse_bad} "
                f"submit={'Y' if done else 'F'} "
                f"({ep_dt:5.1f}s)",
                flush=True,
            )

    out_path = Path("training_data/evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}
    key = "base" if is_base else "trained"
    existing[key] = rows
    out_path.write_text(json.dumps(existing, indent=2))

    scores = [r["score"] for r in rows]
    mean = sum(scores) / len(scores) if scores else 0.0
    by_diff = {d: [] for d in ("easy", "medium", "hard")}
    for r in rows:
        by_diff[r["difficulty"]].append(r["score"])
    diff_means = {d: (sum(v) / len(v) if v else 0.0) for d, v in by_diff.items()}
    eval_dt = _time.time() - t_eval_start
    print(
        f"[eval] {key} mean score = {mean:.3f}  n={len(scores)}  "
        f"parse_ok={parse_ok}/{total_steps}  parse_bad={parse_bad}  "
        f"({eval_dt:.1f}s total)",
        flush=True,
    )
    print(
        f"[eval] {key} by difficulty: "
        + "  ".join(f"{d}={diff_means[d]:.3f}" for d in ("easy", "medium", "hard")),
        flush=True,
    )
    print(f"[eval] wrote {out_path}", flush=True)
    return {"mean": mean, "by_difficulty": diff_means, "n": len(scores)}


def _push_adapter(
    local_dir: str,
    repo_id: str,
    commit_msg: str,
    path_in_repo: str | None = None,
) -> None:
    """Best-effort intermediate upload of an adapter or any folder to HF Hub.

    Wrapped in try/except so a transient network blip doesn't kill the job.
    """
    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()
        create_repo(repo_id, exist_ok=True, repo_type="model")
        kwargs = dict(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
        )
        if path_in_repo:
            kwargs["path_in_repo"] = path_in_repo
        api.upload_folder(**kwargs)
        print(f"[hf-job] pushed {local_dir}/ -> {repo_id}/{path_in_repo or ''} ({commit_msg!r})", flush=True)
    except Exception as exc:
        print(f"[hf-job] WARN: push of {local_dir}/ to {repo_id} failed: {exc}", flush=True)


def main() -> int:
    REPO_URL = os.environ.get(
        "REPO_URL", "https://github.com/Auenchanters/Three-Musketeers-FINALS.git"
    )
    BRANCH = os.environ.get("BRANCH", "main")
    SMOKE = os.environ.get("SMOKE", "0") == "1"
    UPLOAD_REPO = os.environ.get("UPLOAD_REPO", "").strip()
    DO_GRPO = os.environ.get("GRPO", "0") == "1"
    BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()
    MAX_SEQ = int(os.environ.get("MAX_SEQ", "4096"))
    EPOCHS_OVERRIDE = os.environ.get("EPOCHS", "").strip()
    COLLECT_OVERRIDE = os.environ.get("COLLECT_SEEDS", "").strip()
    EVAL_OVERRIDE = os.environ.get("EVAL_SEEDS", "").strip()
    EVAL_MAX_NEW_TOK = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "192"))
    EVAL_MAX_STEPS = int(os.environ.get("EVAL_MAX_STEPS", "30"))
    EVAL_BASE_TOO = os.environ.get("EVAL_BASE_TOO", "0") == "1"
    # EVAL_ONLY mode: skip SFT, just download an external adapter (any RL/SFT/
    # DPO/GRPO trained PEFT) from ADAPTER_REPO and benchmark it against the
    # PostmortemEnv. Lets you plug in any RL-trained policy.
    EVAL_ONLY = os.environ.get("EVAL_ONLY", "0") == "1"
    ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "").strip()
    ADAPTER_SUBFOLDER = os.environ.get("ADAPTER_SUBFOLDER", "").strip()

    WORK = Path("/tmp/postmortem")
    if WORK.exists():
        _run(["rm", "-rf", str(WORK)])

    t_start = time.time()
    print("=" * 60)
    print("[hf-job] PostmortemEnv training job")
    print("=" * 60)
    print(f"[hf-job] repo        : {REPO_URL}@{BRANCH}")
    print(f"[hf-job] mode        : {'SMOKE' if SMOKE else 'FULL'}")
    print(f"[hf-job] base_model  : {BASE_MODEL}")
    print(f"[hf-job] max_seq     : {MAX_SEQ}")
    print(f"[hf-job] grpo stretch: {DO_GRPO}")
    print(f"[hf-job] upload to   : {UPLOAD_REPO or '(no upload)'}")

    _run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(WORK)])
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))

    print("[hf-job] installing requirements-train.txt (~3 min)")
    install_hb = _heartbeat("pip-install", interval=30.0)
    try:
        _install("requirements-train.txt")
    finally:
        install_hb.set()

    # Sanity print: which GPU did we land on?
    import torch  # noqa: E402

    print(f"[hf-job] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"[hf-job] device={torch.cuda.get_device_name(0)}  vram={vram_gb:.1f} GB")

    # `run_full_pipeline` calls `run_sft` with the in-tree SFTConfig defaults
    # (max_seq_length=2048, batch_size=2, fp16, no gradient checkpointing) which
    # OOMs a 22 GB L4 even on Qwen-1.5B because hard-difficulty oracle demos
    # produce long sequences. Drive each phase manually here so we can pass
    # memory-friendly overrides without touching the canonical train.py.
    from train import (  # noqa: E402
        collect_demonstrations,
        evaluate_agents,
        run_sft,
        plot_rewards,
    )
    from trl import SFTConfig  # noqa: E402

    # Forgiving JSON parser: training data uses {"action_type": ...} but
    # smaller models often emit {"action": ...} or wrap the JSON in
    # markdown code fences. The strict parser in training_utils silently
    # falls back to a no-op submit on every step, making the trained agent
    # look identical to random. Patch the parser globally so both training
    # and evaluation see the same forgiving behaviour.
    import training_utils as _tu  # noqa: E402
    import json as _json
    import re as _re

    _orig_parse = _tu.parse_action_json
    _FENCE = _re.compile(r"^```(?:json)?\s*|\s*```$", _re.MULTILINE)

    def _forgiving_parse(text, fallback=None):
        if not isinstance(text, str):
            return _orig_parse(text, fallback=fallback)
        cleaned = _FENCE.sub("", text.strip()).strip()
        try:
            d = _json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    d = _json.loads(cleaned[start : end + 1])
                except Exception:
                    return _orig_parse(text, fallback=fallback)
            else:
                return _orig_parse(text, fallback=fallback)
        if isinstance(d, dict) and "action_type" not in d:
            for k in ("action", "type", "name", "tool", "tool_name"):
                if k in d and isinstance(d[k], str):
                    d["action_type"] = d.pop(k)
                    break
        if isinstance(d, dict) and d.get("action_type"):
            return d
        return _orig_parse(text, fallback=fallback)

    _tu.parse_action_json = _forgiving_parse
    # Re-export under train.py's namespace too (it imported by name)
    import train as _train_mod  # noqa: E402

    _train_mod.parse_action_json = _forgiving_parse
    print("[hf-job] installed forgiving JSON parser (handles ```...```, action vs action_type)")

    # Monkey-patch SFTConfig to enable gradient checkpointing + bf16 (Ada/L4
    # natively supports bf16 and avoids GradScaler memory of fp16). Anything
    # explicitly passed in still wins; we only set defaults.
    _orig_init = SFTConfig.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["gradient_checkpointing"] = True
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        kwargs["bf16"] = True
        kwargs["fp16"] = False  # mutually exclusive with bf16
        # Don't try to ship logs to wandb / tensorboard / aim / etc. — none of
        # those services are configured inside HF Jobs and the trainer can
        # silently block on them. Force "none" + frequent step logging so
        # progress is visible in the job's stdout instead.
        kwargs.setdefault("report_to", "none")
        kwargs.setdefault("logging_steps", 1)
        kwargs.setdefault("logging_first_step", True)
        kwargs.setdefault("disable_tqdm", False)
        _orig_init(self, *args, **kwargs)

    SFTConfig.__init__ = _patched_init

    # train.py loads the base model with torch_dtype=fp16 (legacy default).
    # Combined with our bf16 SFTConfig that creates a silent dtype mismatch
    # which TRL papers over by re-casting weights at every fwd/bwd pass —
    # ~10% slower than loading in bf16 directly. Patch from_pretrained so
    # the model lands in bf16 to match the trainer.
    from transformers import AutoModelForCausalLM as _Auto  # noqa: E402
    import torch as _torch  # noqa: E402

    _orig_fp = _Auto.from_pretrained.__func__ if hasattr(_Auto.from_pretrained, "__func__") else _Auto.from_pretrained

    def _patched_from_pretrained(cls, *args, **kwargs):
        if _torch.cuda.is_available():
            kwargs["torch_dtype"] = _torch.bfloat16
            kwargs.setdefault("attn_implementation", "sdpa")
        return _orig_fp(cls, *args, **kwargs)

    _Auto.from_pretrained = classmethod(_patched_from_pretrained)
    print("[hf-job] patched AutoModelForCausalLM.from_pretrained -> bf16 + sdpa")

    # EVAL_ONLY shortcut: skip SFT, optionally pull an external adapter,
    # and benchmark. Works for ANY HF causal LM (raw or with PEFT adapter)
    # so you can plug in policies trained externally with any RL algorithm.
    if EVAL_ONLY:
        eval_seeds = int(EVAL_OVERRIDE) if EVAL_OVERRIDE else 4
        adapter_dir = "external_adapter"
        if ADAPTER_REPO:
            from huggingface_hub import snapshot_download
            print(f"[hf-job] EVAL_ONLY: downloading adapter {ADAPTER_REPO}"
                  + (f"/{ADAPTER_SUBFOLDER}" if ADAPTER_SUBFOLDER else ""))
            local = snapshot_download(
                repo_id=ADAPTER_REPO,
                allow_patterns=[
                    f"{ADAPTER_SUBFOLDER}/*" if ADAPTER_SUBFOLDER else "*",
                ],
            )
            adapter_dir = (Path(local) / ADAPTER_SUBFOLDER) if ADAPTER_SUBFOLDER else local
            print(f"[hf-job] EVAL_ONLY: adapter at {adapter_dir}")
        else:
            print("[hf-job] EVAL_ONLY: no ADAPTER_REPO -> evaluating BASE model only")
            adapter_dir = "no_adapter_dir_will_skip"
        print(f"[hf-job] EVAL_ONLY: base={BASE_MODEL}  adapter={adapter_dir}  n_seeds={eval_seeds}")
        _eval_trained_with_peft(
            base_model_id=BASE_MODEL,
            adapter_dir=str(adapter_dir),
            n_seeds=eval_seeds,
            parse_fn=_forgiving_parse,
            max_new_tokens=EVAL_MAX_NEW_TOK,
            max_steps_per_ep=EVAL_MAX_STEPS,
            is_base=not bool(ADAPTER_REPO),
        )
        if UPLOAD_REPO and Path("training_data").exists():
            _push_adapter(
                "training_data",
                UPLOAD_REPO,
                "EVAL_ONLY benchmark results",
                path_in_repo="training_data",
            )
        elapsed = time.time() - t_start
        print(f"[hf-job] EVAL_ONLY DONE in {elapsed/60:.1f} min")
        return 0

    if SMOKE:
        collect_seeds, eval_seeds, epochs = 2, 2, 1
    else:
        collect_seeds, eval_seeds, epochs = 50, 10, 5  # 10 seeds * 3 diff = 30 ep eval

    if COLLECT_OVERRIDE:
        collect_seeds = int(COLLECT_OVERRIDE)
    if EVAL_OVERRIDE:
        eval_seeds = int(EVAL_OVERRIDE)
    if EPOCHS_OVERRIDE:
        epochs = int(EPOCHS_OVERRIDE)
    print(f"[hf-job] pipeline knobs: collect={collect_seeds}, eval={eval_seeds}, epochs={epochs}")

    print(f"[hf-job] [1/4] collect oracle demos (n_seeds={collect_seeds})")
    collect_demonstrations(n_seeds=collect_seeds)

    print(f"[hf-job] [2/4] evaluate Random + Oracle baselines (n_seeds={eval_seeds})")
    evaluate_agents(n_seeds=eval_seeds)

    print(f"[hf-job] [3/4] SFT base={BASE_MODEL} epochs={epochs} bs=1 seq={MAX_SEQ} bf16+grad-ckpt")
    print(f"[hf-job] hf_transfer enabled: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    sft_hb = _heartbeat("sft", interval=20.0)
    sft_t0 = time.time()
    try:
        run_sft(
            model_name=BASE_MODEL,
            output_dir="sft_output",
            epochs=epochs,
            batch_size=1,
            max_seq_length=MAX_SEQ,
        )
    finally:
        sft_hb.set()
        print(f"[hf-job] SFT phase done in {time.time() - sft_t0:.1f}s", flush=True)

    # Push the adapter NOW, before eval. Eval can be slow / cancellable; we
    # never want to lose the actual training artefact to a slow eval phase.
    if UPLOAD_REPO and Path("sft_output").exists():
        _push_adapter("sft_output", UPLOAD_REPO, "post-SFT adapter (eval pending)")

    if EVAL_BASE_TOO:
        print("[hf-job] [4/4a] evaluate BASE model (no adapter) for delta")
        _eval_trained_with_peft(
            base_model_id=BASE_MODEL,
            adapter_dir="sft_output",
            n_seeds=eval_seeds,
            parse_fn=_forgiving_parse,
            max_new_tokens=EVAL_MAX_NEW_TOK,
            max_steps_per_ep=EVAL_MAX_STEPS,
            is_base=True,
        )

    print(
        f"[hf-job] [4/4] eval trained LoRA  n_seeds={eval_seeds}  "
        f"max_new_tok={EVAL_MAX_NEW_TOK}  max_steps={EVAL_MAX_STEPS}  "
        f"(stops on '}}', logs per-episode)"
    )
    _eval_trained_with_peft(
        base_model_id=BASE_MODEL,
        adapter_dir="sft_output",
        n_seeds=eval_seeds,
        parse_fn=_forgiving_parse,
        max_new_tokens=EVAL_MAX_NEW_TOK,
        max_steps_per_ep=EVAL_MAX_STEPS,
    )
    plot_rewards()

    if UPLOAD_REPO and Path("training_data").exists():
        _push_adapter(
            "training_data",
            UPLOAD_REPO,
            "post-eval training_data",
            path_in_repo="training_data",
        )

    if DO_GRPO:
        from train import run_grpo  # noqa: E402

        print("[hf-job] GRPO stretch: 100 episodes on top of sft_output")
        run_grpo(model_name="./sft_output", n_episodes=100)

    if UPLOAD_REPO:
        from huggingface_hub import HfApi, create_repo  # noqa: E402

        api = HfApi()
        print(f"[hf-job] creating/upserting repo {UPLOAD_REPO}")
        create_repo(UPLOAD_REPO, exist_ok=True, repo_type="model")

        if Path("sft_output").exists():
            print(f"[hf-job] uploading sft_output/ -> {UPLOAD_REPO}")
            api.upload_folder(
                folder_path="sft_output",
                repo_id=UPLOAD_REPO,
                repo_type="model",
                commit_message="hf-job: SFT LoRA adapter (smoke)" if SMOKE else "hf-job: SFT LoRA adapter",
            )

        if Path("training_data").exists():
            print(f"[hf-job] uploading training_data/ -> {UPLOAD_REPO}/training_data")
            api.upload_folder(
                folder_path="training_data",
                repo_id=UPLOAD_REPO,
                path_in_repo="training_data",
                repo_type="model",
                commit_message="hf-job: refreshed training_data artefacts",
            )

        if DO_GRPO and Path("grpo_output").exists():
            print(f"[hf-job] uploading grpo_output/ -> {UPLOAD_REPO}/grpo_output")
            api.upload_folder(
                folder_path="grpo_output",
                repo_id=UPLOAD_REPO,
                path_in_repo="grpo_output",
                repo_type="model",
                commit_message="hf-job: GRPO checkpoint",
            )

    elapsed = time.time() - t_start
    print(f"[hf-job] DONE in {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
