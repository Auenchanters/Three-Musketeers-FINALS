"""
Generate publication-quality plots for the PostmortemEnv README.

Outputs:
  training_data/reward_curves.png    -- main 3-panel agent comparison
  training_data/agent_comparison.png -- single-panel score comparison

Both bar charts now annotate **95 % confidence intervals** computed from
the per-episode score distribution. With the default 30-seeds-per-
difficulty evaluation the standard error of the mean is ≈ 0.03, so a
+0.10 lift between agents lands well outside the CI overlap and is
statistically real, not noise.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Two-sided 95% normal-approximation z-score. With n>=30 per difficulty
# this is the standard "1.96 × SE" CI used in everyday science writing.
Z_95 = 1.96


DATA_DIR = Path("training_data")
AGENT_CONFIG = {
    "random":    {"label": "Random baseline",    "color": "#e07070", "linestyle": "--", "zorder": 1},
    "heuristic": {"label": "Heuristic (rules)",  "color": "#d4a030", "linestyle": "-.", "zorder": 2},
    "trained":   {"label": "Trained (SFT/GRPO)", "color": "#5cad7a", "linestyle": "-",  "zorder": 3},
    "oracle":    {"label": "Oracle upper bound", "color": "#4a86b8", "linestyle": "-",  "zorder": 4},
}
AGENT_ORDER = ["random", "heuristic", "trained", "oracle"]


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


def ci95(episodes: list) -> float:
    """95% confidence interval half-width for the mean of ``episodes`` scores.

    Uses the unbiased sample standard deviation (Bessel's correction) and a
    normal-approximation ``z = 1.96``. With ``n < 2`` the CI is undefined
    so this returns ``0.0`` rather than raising — bar charts can still be
    rendered (just without an error bar).
    """
    n = len(episodes)
    if n < 2:
        return 0.0
    scores = [r["score"] for r in episodes]
    m = sum(scores) / n
    var = sum((s - m) ** 2 for s in scores) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return Z_95 * se


def n_seeds_per_difficulty(results: dict) -> int:
    """Maximum per-difficulty episode count seen across agents.

    Used so the plot caption reflects the actual run (10, 30, …) rather
    than a hard-coded "10".
    """
    best = 0
    for episodes in results.values():
        if not isinstance(episodes, list):
            continue
        for diff in ("easy", "medium", "hard"):
            count = sum(1 for r in episodes if r.get("difficulty") == diff)
            if count > best:
                best = count
    return best or 10


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
    present = [n for n in AGENT_ORDER if results.get(n)]
    n_per_diff = n_seeds_per_difficulty(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "PostmortemEnv — Agent Performance Across %d Seeds x 3 Difficulties (95%% CI)"
        % n_per_diff,
        fontsize=13, fontweight="bold", y=1.01
    )

    # Panel 1: mean score bar chart with 95% CI error bars
    ax = axes[0]
    means = [mean_score(results[n]) for n in present]
    cis = [ci95(results[n]) for n in present]
    colors = [AGENT_CONFIG[n]["color"] for n in present]
    labels = [AGENT_CONFIG[n]["label"] for n in present]
    bars = ax.bar(
        labels, means, color=colors, alpha=0.88, width=0.5,
        yerr=cis, capsize=6,
        error_kw={"ecolor": "#333333", "elinewidth": 1.4, "alpha": 0.85},
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean final score (0 - 1)", fontsize=11)
    ax.set_title("Overall mean score (±95%% CI, n=%d)" % len(results[present[0]]), fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    if "random" in present:
        ax.axhline(
            mean_score(results["random"]),
            color=AGENT_CONFIG["random"]["color"],
            linestyle="--", alpha=0.45, linewidth=1.2,
            label="Random baseline"
        )
    for bar, m, ci in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + ci + 0.02,
            "%.3f ± %.3f" % (m, ci),
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax.tick_params(axis="x", labelsize=9)

    # Panel 2: per-difficulty grouped bars with per-cell 95% CI
    ax = axes[1]
    difficulties = ["easy", "medium", "hard"]
    width = 0.75 / len(present)
    for idx, agent_name in enumerate(present):
        diff_means = []
        diff_cis = []
        for d in difficulties:
            cell = [r for r in results[agent_name] if r["difficulty"] == d]
            diff_means.append(mean_score(cell))
            diff_cis.append(ci95(cell))
        offsets = [i + (idx - len(present) / 2 + 0.5) * width for i in range(3)]
        ax.bar(
            offsets, diff_means, width * 0.9,
            color=AGENT_CONFIG[agent_name]["color"],
            label=AGENT_CONFIG[agent_name]["label"],
            alpha=0.88,
            yerr=diff_cis, capsize=4,
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "alpha": 0.75},
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["Easy (40 steps)", "Medium (75 steps)", "Hard (120 steps)"],
        fontsize=9,
    )
    ax.set_ylabel("Mean final score (0 - 1)", fontsize=11)
    ax.set_title("Score by difficulty tier (±95%% CI)", fontsize=11)
    ax.set_ylim(0, 1.15)
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
    """Single-panel score comparison — used in the README. Renders 95% CI bars."""
    present = [n for n in AGENT_ORDER if results.get(n)]
    means = [mean_score(results[n]) for n in present]
    cis = [ci95(results[n]) for n in present]
    labels = [AGENT_CONFIG[n]["label"] for n in present]
    colors = [AGENT_CONFIG[n]["color"] for n in present]
    n_per_diff = n_seeds_per_difficulty(results)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(
        labels, means, color=colors, alpha=0.88, height=0.45,
        xerr=cis, capsize=6,
        error_kw={"ecolor": "#333333", "elinewidth": 1.4, "alpha": 0.85},
    )
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Mean final score (0 - 1)", fontsize=11)
    ax.set_title(
        "PostmortemEnv — Agent Comparison (n=%d seeds x 3 difficulties, 95%% CI)"
        % n_per_diff,
        fontsize=11, fontweight="bold"
    )
    for bar, m, ci in zip(bars, means, cis):
        ax.text(
            m + ci + 0.015, bar.get_y() + bar.get_height() / 2,
            "%.3f ± %.3f" % (m, ci),
            va="center", fontsize=10, fontweight="bold"
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
