"""Agent-comparison bar chart for PostmortemEnv.

Reads whatever evaluation JSONs are present in ``training_data/`` and
plots a single bar chart with **every available agent** side-by-side:

    Random  |  SFT (best epoch)  |  Llama-3.2-1B  |  GPT-4o-mini  |
    Claude Haiku 4.5  |  Oracle

Skips any source that's missing — so the same script works whether or
not you've run the frontier benchmark yet. Output goes to
``training_data/agent_comparison.png`` (a separate file from the
existing 4-panel ``reward_curves.png`` so neither overwrites the other).

Usage::

    python scripts/plot_agent_comparison.py
    python scripts/plot_agent_comparison.py --output docs/agent_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "training_data"


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _load_baseline(path: Path) -> Optional[Dict[str, float]]:
    """Read ``evaluation_results.json`` -> {Random, Oracle} mean scores."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out: Dict[str, float] = {}
    for label, key in (("Random", "random"), ("Oracle", "oracle")):
        rows = data.get(key) or []
        if rows:
            out[label] = _mean([r["score"] for r in rows])
    return out or None


def _load_sft(path: Path) -> Optional[Tuple[str, float, int]]:
    """Read ``sft_eval_results.json`` -> (label, best_mean, best_epoch)."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    epoch_results = data.get("epoch_results") or {}
    if not epoch_results:
        return None
    best_epoch = -1
    best_mean = -1.0
    for ep_str, rows in epoch_results.items():
        ep_mean = _mean([r["score"] for r in rows])
        if ep_mean > best_mean:
            best_mean = ep_mean
            best_epoch = int(ep_str)
    short_model = (data.get("model") or "?").split("/")[-1]
    label = f"SFT\n({short_model}, ep{best_epoch})"
    return label, round(best_mean, 4), best_epoch


def _load_frontier(path: Path) -> List[Tuple[str, float, int]]:
    """Read ``frontier_results.json`` -> [(label, mean, n_runs), ...]."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    summary = data.get("summary") or {}
    rows: List[Tuple[str, float, int]] = []
    for model, bucket in summary.items():
        rows.append((model, float(bucket["mean_score"]), int(bucket["n_runs"])))
    return rows


def collect_agents(data_dir: Path) -> List[Dict[str, object]]:
    """Build the ordered agent list, skipping anything we have no data for."""
    agents: List[Dict[str, object]] = []

    baseline = _load_baseline(data_dir / "evaluation_results.json")
    if baseline and "Random" in baseline:
        agents.append({
            "label": "Random",
            "score": round(baseline["Random"], 4),
            "color": "#e07a7a",
            "kind": "baseline",
        })

    sft = _load_sft(data_dir / "sft_eval_results.json")
    if sft is not None:
        label, score, _ = sft
        agents.append({
            "label": label,
            "score": score,
            "color": "#7aa9d6",
            "kind": "sft",
        })

    for label, score, n in _load_frontier(data_dir / "frontier_results.json"):
        agents.append({
            "label": f"{label}\n(n={n})",
            "score": round(score, 4),
            "color": "#9b7ad6",
            "kind": "frontier",
        })

    if baseline and "Oracle" in baseline:
        agents.append({
            "label": "Oracle",
            "score": round(baseline["Oracle"], 4),
            "color": "#7ad6a7",
            "kind": "baseline",
        })

    return agents


def render(agents: List[Dict[str, object]], output: Path) -> None:
    if not agents:
        print(
            "No agent data found. Run `python train.py evaluate --n-seeds 10` "
            "first (and optionally `python scripts/run_frontier_benchmark.py`).",
            file=sys.stderr,
        )
        sys.exit(1)

    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    labels = [a["label"] for a in agents]
    scores = [a["score"] for a in agents]
    colors = [a["color"] for a in agents]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(agents)), 5.5))
    bars = ax.bar(labels, scores, color=colors, alpha=0.9, edgecolor="white", linewidth=1.2)

    # Per-bar score label
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontweight="bold", fontsize=10,
        )

    # Reference line at Random for "lift over baseline" intuition
    random_score = next((a["score"] for a in agents if a["label"] == "Random"), None)
    oracle_score = next((a["score"] for a in agents if a["label"] == "Oracle"), None)
    if random_score is not None:
        ax.axhline(y=random_score, color="#e07a7a", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"Random baseline ({random_score:.3f})")
    if oracle_score is not None:
        ax.axhline(y=oracle_score, color="#7ad6a7", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"Oracle ceiling ({oracle_score:.3f})")

    ax.set_ylim(0, max(1.0, max(scores) * 1.15))
    ax.set_ylabel("Mean episode score", fontsize=11)
    ax.set_title("PostmortemEnv — agent comparison", fontsize=14, fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    if random_score is not None or oracle_score is not None:
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    if random_score is not None and oracle_score is not None:
        gap = oracle_score - random_score
        fig.text(
            0.5, 0.02,
            f"Reward gap (Oracle − Random) = {gap:.3f}. "
            f"All scores in (0.01, 0.99). See training_data/*.json for raw rows.",
            ha="center", fontsize=9, style="italic", color="#555555",
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output}")
    print()
    print("Agents on chart:")
    for a in agents:
        print(f"  {a['label']:<30} {a['score']:.4f}  ({a['kind']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Directory containing the evaluation JSONs.")
    p.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR / "agent_comparison.png",
                   help="Where to write the PNG.")
    args = p.parse_args()

    agents = collect_agents(args.data_dir)
    render(agents, args.output)


if __name__ == "__main__":
    main()
