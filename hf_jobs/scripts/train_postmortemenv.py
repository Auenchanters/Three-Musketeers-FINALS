#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.26",
#   "pydantic>=2.0",
# ]
# ///
"""
PostmortemEnv — live training run on Hugging Face Jobs.

This script is the **canonical, reproducible** training run that powers the
"Live training" chart in the PostmortemEnv frontend. It is designed to be
launched two ways:

1. **Hugging Face Jobs** (recommended):

       hf jobs uv run \\
           --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim \\
           --flavor cpu-basic \\
           --timeout 30m \\
           --secrets HF_TOKEN=$HF_TOKEN \\
           https://raw.githubusercontent.com/Auenchanters/Three-Musketeers-FINALS/UTK-Frontend/hf_jobs/scripts/train_postmortemenv.py \\
           -- \\
           --task task1_recent_deploy \\
           --episodes 300 \\
           --metrics-repo Auenchanters/postmortemenv-live-metrics \\
           --run-id $(uuidgen | head -c 12)

2. **Google Colab**: open the matching notebook in
   ``hf_jobs/notebooks/train_postmortemenv_colab.ipynb`` and run all cells.

What it does
------------
* Pulls the PostmortemEnv code from this GitHub repo (so the script is
  fully self-contained — no need to push the env to PyPI).
* Builds the same task-specific action menu as ``web/training_loop.py``
  (every commit / config / infra event, plus correlated ``commit+config``
  / ``commit+infra`` pair candidates so the harder tasks are learnable).
* Runs on-policy REINFORCE on the env's own reward signal, identical
  algorithm to the in-process mirror.
* Pushes a streaming ``metrics.jsonl`` to ``--metrics-repo`` on the Hub
  every ``--push-every`` episodes, so the frontend chart can ``hf_hub_download``
  it and animate the curve as it grows.

Why this design
---------------
Per the team's "no Oracle, only HF / Colab" rule, training **must** run on
Hugging Face infrastructure (or Colab). The frontend never assumes the
training already exists; it polls the metrics repo on Hub. If the repo
is empty / missing, the chart shows "no run yet — kick one off".

Output schema (one JSON record per line of metrics.jsonl)
---------------------------------------------------------
First record (type=baselines)::

    {"type": "baselines", "task_id": ..., "random": 0.21,
     "n_episodes": 300, "runtime": "hf-jobs",
     "started_at": <unix>, "run_id": <uuid12>}

Per-episode records (type=metric)::

    {"type": "metric", "episode": <int>, "score": <float>,
     "rolling_mean": <float>, "best": <float>,
     "lift_over_random": <float>, "elapsed_sec": <float>,
     "run_id": <uuid12>}

Final record (type=done)::

    {"type": "done", "final_rolling_mean": <float>,
     "lift_over_random": <float>, "n_episodes": <int>, "run_id": <uuid12>}

The frontend's poller skips any line whose ``run_id`` doesn't match the
session it subscribed to, so multiple training runs can share one repo
without interfering with each other.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1: pull the PostmortemEnv source. The HF Jobs container is empty
# except for the UV runtime, so we shallow-clone the repo for the env code.
# ---------------------------------------------------------------------------

REPO_URL = os.environ.get(
    "POSTMORTEMENV_REPO_URL",
    "https://github.com/Auenchanters/Three-Musketeers-FINALS.git",
)
REPO_BRANCH = os.environ.get("POSTMORTEMENV_REPO_BRANCH", "UTK-Frontend")


def _clone_repo() -> Path:
    """Shallow-clone the PostmortemEnv repo into a temp dir and return it."""
    target = Path(tempfile.mkdtemp(prefix="postmortemenv-")) / "src"
    print(f"[setup] cloning {REPO_URL} (branch {REPO_BRANCH}) -> {target}")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            REPO_BRANCH,
            REPO_URL,
            str(target),
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return target


# ---------------------------------------------------------------------------
# Step 2: argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", default="task1_recent_deploy",
        help="task_id to train on (one of the 5 hand-crafted scenarios)",
    )
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.4)
    p.add_argument("--baseline-decay", type=float, default=0.92)
    p.add_argument("--max-episode-steps", type=int, default=24)
    p.add_argument(
        "--metrics-repo", default=None,
        help=(
            "HF dataset repo to push metrics.jsonl to "
            "(e.g. 'Auenchanters/postmortemenv-live-metrics'). "
            "If omitted, metrics are written to ./metrics.jsonl only."
        ),
    )
    p.add_argument(
        "--metrics-path-in-repo", default="metrics.jsonl",
        help="filename inside --metrics-repo for the streaming metrics",
    )
    p.add_argument(
        "--push-every", type=int, default=10,
        help="push metrics.jsonl to Hub every N episodes (lower = more live, "
             "higher = fewer Hub API calls)",
    )
    p.add_argument(
        "--run-id", default=None,
        help="explicit run id (defaults to a fresh uuid12) — the frontend "
             "uses this to filter records that belong to its session",
    )
    p.add_argument(
        "--no-network-baseline", action="store_true",
        help="skip the random-baseline estimation (12 random rollouts) and "
             "use a hardcoded 0.10 — useful in offline tests",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 3: REINFORCE training loop. Code is duplicated from
# web/training_loop.py rather than imported, because UV scripts run with a
# bare working directory and we want this file to be reviewable in
# isolation by judges / Colab users without bouncing through git imports.
# ---------------------------------------------------------------------------


def _candidate_actions(scenario: dict, ActionType) -> list[dict]:
    services = list(scenario.get("service_graph", {}).keys()) or [
        s["name"] for s in scenario.get("services", []) or []
    ]
    commits = [c["hash"] for c in scenario.get("commits", []) or []]
    configs = [c["config_id"] for c in scenario.get("config_changes", []) or []]
    traces = [t["trace_id"] for t in scenario.get("traces", []) or []]
    infras = [e["event_id"] for e in scenario.get("infra_events", []) or []]

    actions: list[dict] = []
    for svc in services:
        actions.append({"action_type": ActionType.QUERY_LOGS.value, "service": svc})
    for tr in traces:
        actions.append({"action_type": ActionType.FETCH_TRACE.value, "trace_id": tr})
    for h in commits:
        actions.append({"action_type": ActionType.DIFF_COMMIT.value, "commit_hash": h})
    for cid in configs:
        actions.append({"action_type": ActionType.INSPECT_CONFIG.value, "config_id": cid})
    for eid in infras:
        actions.append({"action_type": ActionType.INSPECT_INFRA.value, "event_id": eid})

    single = commits + configs + infras
    pairs: list[str] = []
    for c in commits:
        for cfg in configs:
            pairs.append(f"{c}+{cfg}")
        for ev in infras:
            pairs.append(f"{c}+{ev}")
    cause_candidates = single + pairs
    for cand in cause_candidates:
        actions.append({"action_type": ActionType.HYPOTHESIZE.value, "cause_entity_id": cand})
    for cand in cause_candidates:
        actions.append({
            "action_type": ActionType.SUBMIT.value,
            "final_cause": cand,
            "final_chain": [],
        })
    return actions


def _state_key(env) -> tuple:
    step = getattr(env.state, "steps_taken", 0)
    if step <= 2:
        phase = 0
    elif step <= 5:
        phase = 1
    elif step <= 9:
        phase = 2
    else:
        phase = 3
    n_facts = len(getattr(env, "_known_facts", []))
    facts_bucket = 0 if n_facts == 0 else 1 if n_facts <= 2 else 2 if n_facts <= 4 else 3
    hyp_bucket = min(len(getattr(env, "_hypotheses_submitted", [])), 3)
    return (phase, facts_bucket, hyp_bucket)


@dataclass
class _Policy:
    menu: list[dict]
    rng: random.Random
    logits: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        self.logits = {}

    def _row(self, s):
        row = self.logits.get(s)
        if row is None:
            row = [0.0] * len(self.menu)
            self.logits[s] = row
        return row

    def probs(self, s):
        row = self._row(s)
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        z = sum(exps) or 1.0
        return [e / z for e in exps]

    def sample(self, s):
        probs = self.probs(s)
        r = self.rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return len(probs) - 1

    def update(self, traj, advantage, lr):
        for s, a in traj:
            probs = self.probs(s)
            row = self._row(s)
            for j in range(len(row)):
                grad = (1.0 - probs[a]) if j == a else (-probs[j])
                row[j] += lr * advantage * grad


def _build_action(d: dict, Action):
    return Action(
        action_type=d.get("action_type", "query_logs"),
        service=d.get("service"),
        keyword=d.get("keyword"),
        trace_id=d.get("trace_id"),
        commit_hash=d.get("commit_hash"),
        config_id=d.get("config_id"),
        event_id=d.get("event_id"),
        cause_entity_id=d.get("cause_entity_id"),
        chain=d.get("chain"),
        final_cause=d.get("final_cause"),
        final_chain=d.get("final_chain"),
    )


def _run_episode(env, policy, task_id, max_steps, Action):
    env.reset(task_id=task_id)
    traj = []
    for _ in range(max_steps):
        s = _state_key(env)
        a_idx = policy.sample(s)
        try:
            obs = env.step(_build_action(policy.menu[a_idx], Action))
        except Exception:
            traj.append((s, a_idx))
            continue
        traj.append((s, a_idx))
        if obs.done:
            break
    return env.get_final_score(), traj


def _estimate_random_baseline(env, task_id, scenario, ActionType, Action, n: int = 12):
    rng = random.Random(0xC0FFEE)
    menu = _candidate_actions(scenario, ActionType)
    scores = []
    for _ in range(n):
        env.reset(task_id=task_id)
        for _ in range(24):
            try:
                obs = env.step(_build_action(rng.choice(menu), Action))
            except Exception:
                continue
            if obs.done:
                break
        scores.append(env.get_final_score())
    return float(sum(scores) / max(len(scores), 1))


# ---------------------------------------------------------------------------
# Step 4: HF Hub uploader. Streams metrics.jsonl by re-uploading the file
# every ``--push-every`` episodes. We use ``HfApi.upload_file`` (rather than
# repo_create + commit) because the file size stays small (~30 KB for 300
# episodes) so a full re-upload is a few hundred ms — well within the
# poll interval the frontend uses.
# ---------------------------------------------------------------------------


def _maybe_init_repo(api, repo_id: str) -> None:
    """Create the dataset repo if it doesn't exist yet (idempotent)."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        print(f"[hub] creating dataset repo {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def _push_metrics(api, repo_id: str, path_in_repo: str, body: bytes) -> None:
    api.upload_file(
        path_or_fileobj=io.BytesIO(body),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"streaming metrics @ {time.strftime('%H:%M:%S')}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or uuid.uuid4().hex[:12]
    started_at = time.time()
    print(f"[run] task={args.task}  episodes={args.episodes}  run_id={run_id}")

    # 1. Make the env importable.
    src = _clone_repo()
    sys.path.insert(0, str(src))

    from data.generator import load_scenario  # noqa: E402
    from engine.environment import PostmortemEnvironment  # noqa: E402
    from models.action import Action, ActionType  # noqa: E402

    # 2. Build menu, baseline, policy, env.
    scenario = load_scenario(args.task)
    menu = _candidate_actions(scenario, ActionType)
    if not menu:
        print(f"[error] empty action menu for task {args.task}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    policy = _Policy(menu=menu, rng=rng)
    env = PostmortemEnvironment()

    if args.no_network_baseline:
        random_baseline = 0.10
    else:
        random_baseline = _estimate_random_baseline(
            env, args.task, scenario, ActionType, Action, n=12
        )
    print(f"[run] random_baseline = {random_baseline:.3f}  menu_size = {len(menu)}")

    # 3. Set up Hub uploader (if a repo was provided).
    api = None
    if args.metrics_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            _maybe_init_repo(api, args.metrics_repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[hub] setup failed ({exc!r}); falling back to local file only",
                  file=sys.stderr)
            api = None

    # 4. Local file fallback (also useful for Colab where the user may
    #    download the file manually instead of going through Hub).
    local_path = Path("metrics.jsonl")
    local_path.write_text("")

    def emit(rec: dict) -> None:
        rec = {**rec, "run_id": run_id}
        line = json.dumps(rec) + "\n"
        with open(local_path, "a") as fh:
            fh.write(line)

    def push_to_hub() -> None:
        if api is None:
            return
        try:
            _push_metrics(
                api,
                args.metrics_repo,
                args.metrics_path_in_repo,
                local_path.read_bytes(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[hub] upload failed: {exc!r}", file=sys.stderr)

    # 5. Emit baselines record.
    emit({
        "type": "baselines",
        "task_id": args.task,
        "random": round(random_baseline, 4),
        "action_menu_size": len(menu),
        "n_episodes": args.episodes,
        "runtime": "hf-jobs",
        "started_at": started_at,
    })
    push_to_hub()

    # 6. Train.
    baseline = random_baseline
    window: list[float] = []
    window_size = 20
    best = 0.0
    decay = args.baseline_decay
    lr = args.learning_rate

    for ep in range(1, args.episodes + 1):
        score, traj = _run_episode(env, policy, args.task, args.max_episode_steps, Action)
        advantage = score - baseline
        policy.update(traj, advantage, lr)
        baseline = decay * baseline + (1.0 - decay) * score

        window.append(score)
        if len(window) > window_size:
            window.pop(0)
        rolling = sum(window) / len(window)
        if score > best:
            best = score

        emit({
            "type": "metric",
            "episode": ep,
            "score": round(score, 4),
            "rolling_mean": round(rolling, 4),
            "best": round(best, 4),
            "lift_over_random": round(rolling - random_baseline, 4),
            "elapsed_sec": round(time.time() - started_at, 2),
        })

        if ep % args.push_every == 0:
            push_to_hub()
            print(
                f"[ep {ep:>4}/{args.episodes}] score={score:.3f} "
                f"rolling={rolling:.3f}  +{rolling - random_baseline:+.3f} vs random"
            )

    final_mean = sum(window) / max(len(window), 1)
    emit({
        "type": "done",
        "final_rolling_mean": round(final_mean, 4),
        "lift_over_random": round(final_mean - random_baseline, 4),
        "n_episodes": args.episodes,
        "runtime": "hf-jobs",
        "elapsed_sec": round(time.time() - started_at, 2),
    })
    push_to_hub()

    print(
        f"[done] final_rolling_mean={final_mean:.3f}  "
        f"lift_over_random={final_mean - random_baseline:+.3f}  "
        f"random_baseline={random_baseline:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
