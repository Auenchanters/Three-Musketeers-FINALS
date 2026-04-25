"""
Generate publication-quality plots for the PostmortemEnv README.

Outputs:
  training_data/reward_curves.png    -- main 3-panel agent comparison
  training_data/agent_comparison.png -- single-panel score comparison
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DATA_DIR = Path("training_data")
AGENT_CONFIG = {
    "random":  {"label": "Random baseline",  "color": "#e07070", "linestyle": "--", "zorder": 1},
    "trained": {"label": "Trained (SFT/GRPO)", "color": "#5cad7a", "linestyle": "-",  "zorder": 3},
    "oracle":  {"label": "Oracle upper bound", "color": "#4a86b8", "linestyle": "-",  "zorder": 2},
}


def load_results() -> dict:
    path = DATA_DIR / "evaluation_results.json"
    if not path.exists():
        raise FileNotFoundError(
            "evaluation_results.json not found. "
            "Run: python train.py evaluate --n-seeds 10 [--model ./grpo_output]"
        )
    with open(path) as f:
        return json.load(f)


def mean_score(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(r["score"] for r in episodes) / len(episodes)


def cumulative_avg(episodes: list) -> list:
    """Average cumulative reward per step across all episodes."""
    if not episodes:
        return []
    all_rewards = [r.get("rewards", []) for r in episodes]
    max_len = max((len(r) for r in all_rewards), default=0)
    avgs = []
    for step in range(max_len):
        vals = []
        for ep in all_rewards:
            if step < len(ep):
                vals.append(sum(ep[:step + 1]))
            elif ep:
                vals.append(sum(ep))
        avgs.append(sum(vals) / len(vals) if vals else 0.0)
    return avgs


def plot_main(results: dict):
    present = [n for n in ["random", "trained", "oracle"] if results.get(n)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "PostmortemEnv — Agent Performance Across 10 Seeds x 3 Difficulties",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Panel 1: mean score bar chart
    ax = axes[0]
    means = [mean_score(results[n]) for n in present]
    colors = [AGENT_CONFIG[n]["color"] for n in present]
    labels = [AGENT_CONFIG[n]["label"] for n in present]
    bars = ax.bar(labels, means, color=colors, alpha=0.88, width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean final score (0 - 1)", fontsize=11)
    ax.set_title("Overall mean score", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    if "random" in present:
        ax.axhline(
            mean_score(results["random"]),
            color=AGENT_CONFIG["random"]["color"],
            linestyle="--", alpha=0.45, linewidth=1.2,
            label="Random baseline"
        )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + 0.015,
            "%.3f" % m, ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.tick_params(axis="x", labelsize=9)

    # Panel 2: per-difficulty grouped bars
    ax = axes[1]
    difficulties = ["easy", "medium", "hard"]
    width = 0.75 / len(present)
    for idx, agent_name in enumerate(present):
        diff_means = [
            mean_score([r for r in results[agent_name] if r["difficulty"] == d])
            for d in difficulties
        ]
        offsets = [i + (idx - len(present) / 2 + 0.5) * width for i in range(3)]
        ax.bar(
            offsets, diff_means, width * 0.9,
            color=AGENT_CONFIG[agent_name]["color"],
            label=AGENT_CONFIG[agent_name]["label"],
            alpha=0.88
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Easy (40 steps)", "Medium (75 steps)", "Hard (120 steps)"], fontsize=9)
    ax.set_ylabel("Mean final score (0 - 1)", fontsize=11)
    ax.set_title("Score by difficulty tier", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8)

    # Panel 3: cumulative reward per step
    ax = axes[2]
    for agent_name in present:
        cum = cumulative_avg(results[agent_name])
        cfg = AGENT_CONFIG[agent_name]
        ax.plot(
            range(len(cum)), cum,
            label="%s (final %.3f)" % (cfg["label"], mean_score(results[agent_name])),
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
            zorder=cfg["zorder"],
        )
    ax.set_xlabel("Step within episode", fontsize=11)
    ax.set_ylabel("Mean cumulative reward", fontsize=11)
    ax.set_title("Cumulative reward per step", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = DATA_DIR / "reward_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved %s" % out)
    plt.close()


def plot_comparison(results: dict):
    """Single-panel score comparison — used in the README."""
    present = [n for n in ["random", "trained", "oracle"] if results.get(n)]
    means = [mean_score(results[n]) for n in present]
    labels = [AGENT_CONFIG[n]["label"] for n in present]
    colors = [AGENT_CONFIG[n]["color"] for n in present]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, means, color=colors, alpha=0.88, height=0.45)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean final score (0 - 1)", fontsize=11)
    ax.set_title(
        "PostmortemEnv — Agent Comparison (n=10 seeds x 3 difficulties)",
        fontsize=11, fontweight="bold"
    )
    for bar, m in zip(bars, means):
        ax.text(
            m + 0.01, bar.get_y() + bar.get_height() / 2,
            "%.3f" % m, va="center", fontsize=10, fontweight="bold"
        )
    if "random" in present:
        ax.axvline(
            mean_score(results["random"]),
            color=AGENT_CONFIG["random"]["color"],
            linestyle="--", alpha=0.45, linewidth=1.2,
        )
    ax.invert_yaxis()
    plt.tight_layout()
    out = DATA_DIR / "agent_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved %s" % out)
    plt.close()


if __name__ == "__main__":
    results = load_results()
    plot_main(results)
    plot_comparison(results)
    print("Done. Commit training_data/reward_curves.png and training_data/agent_comparison.png.")
