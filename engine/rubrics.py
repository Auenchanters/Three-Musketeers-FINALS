"""Composable Rubric system for PostmortemEnv.

OpenEnv best practice: decompose evaluation into independent, verifiable
rubrics rather than a single monolithic score.  Each ``Rubric`` is an
autonomous scoring dimension with its own weight, range, and description.

The ``RubricSet`` collects them and produces both a weighted aggregate
and a per-rubric breakdown, making reward signals transparent and auditable.

Design rationale
────────────────
1. **Independent signals reduce reward hacking.**  A single scalar is one
   surface to exploit; five orthogonal rubrics force the optimizer to
   satisfy all of them simultaneously.
2. **Inspectable.**  Every training step logs *why* the score is what it
   is (e.g. "cause_correctness: 1.0, investigation_quality: 0.35").
3. **Extensible.**  Adding a new criterion (e.g. "safety") requires one
   ``Rubric(...)`` call and a weight rebalance — no code surgery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class RubricScore:
    """Result from evaluating a single rubric."""
    name: str
    raw_score: float       # in [0, 1]
    weight: float          # in [0, 1], defined by the Rubric
    weighted_score: float  # raw_score * weight
    description: str = ""


@dataclass(frozen=True)
class Rubric:
    """A single, independent evaluation dimension.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"cause_correctness"``).
    weight : float
        Relative weight in the total score.  The ``RubricSet`` normalises
        weights so they sum to 1.0.
    description : str
        One-sentence explanation shown in reward breakdowns.
    evaluator : Callable[..., float]
        A pure function ``(**kwargs) -> float`` returning a score in [0, 1].
    """
    name: str
    weight: float
    description: str
    evaluator: Callable[..., float]

    def score(self, **kwargs: Any) -> RubricScore:
        raw = max(0.0, min(1.0, self.evaluator(**kwargs)))
        return RubricScore(
            name=self.name,
            raw_score=round(raw, 4),
            weight=self.weight,
            weighted_score=round(raw * self.weight, 4),
            description=self.description,
        )


class RubricSet:
    """Ordered collection of rubrics that produces a composite score."""

    def __init__(self, rubrics: List[Rubric]) -> None:
        self.rubrics = list(rubrics)
        total_w = sum(r.weight for r in self.rubrics)
        if total_w <= 0:
            raise ValueError("Total rubric weight must be positive.")
        # Normalise so weights sum to 1.0
        self._norm = total_w

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """Run every rubric and return composite + breakdown.

        Returns
        -------
        dict with keys:
          * ``"total"`` — normalised weighted sum in [0, 1]
          * ``"breakdown"`` — list of ``RubricScore`` dicts
          * ``"rubric_names"`` — ordered list of rubric names
        """
        results: List[RubricScore] = []
        weighted_sum = 0.0
        for rubric in self.rubrics:
            rs = rubric.score(**kwargs)
            adjusted = RubricScore(
                name=rs.name,
                raw_score=rs.raw_score,
                weight=round(rubric.weight / self._norm, 4),
                weighted_score=round(rs.raw_score * rubric.weight / self._norm, 4),
                description=rs.description,
            )
            results.append(adjusted)
            weighted_sum += adjusted.weighted_score

        return {
            "total": round(min(max(weighted_sum, 0.0), 1.0), 4),
            "breakdown": [
                {
                    "rubric": r.name,
                    "raw_score": r.raw_score,
                    "weight": r.weight,
                    "weighted_score": r.weighted_score,
                    "description": r.description,
                }
                for r in results
            ],
            "rubric_names": [r.name for r in results],
        }


# ──────────────────────────────────────────────────────────────────────
# PostmortemEnv Rubrics  (composable; judges can inspect these)
# ──────────────────────────────────────────────────────────────────────

def _cause_correctness(cause_match: bool = False, partial_credit: float = 0.0,
                       **_: Any) -> float:
    """1.0 if exact match, partial_credit (0–1) for correlated partial, else 0."""
    if cause_match:
        return 1.0
    return partial_credit


def _chain_accuracy(chain_similarity: float = 0.0, **_: Any) -> float:
    """Jaccard node+edge similarity of predicted vs ground-truth chain."""
    return chain_similarity


def _efficiency(steps_used: int = 0, max_steps: int = 40, **_: Any) -> float:
    """Bonus for solving quickly: max(0, 1 − steps/max_steps)."""
    if max_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - steps_used / max_steps)


def _investigation_quality(facts_found: int = 0, total_facts: int = 1,
                           action_types_used: int = 0, **_: Any) -> float:
    """How thoroughly the agent explored: fact coverage × action diversity."""
    coverage = min(1.0, facts_found / max(1, total_facts))
    # Bonus for using 4+ distinct action types (out of 8)
    diversity = min(1.0, action_types_used / 4.0) if action_types_used > 0 else 0.0
    return 0.6 * coverage + 0.4 * diversity


def _anti_gaming(wrong_hypotheses: int = 0, repeated_queries: int = 0,
                 hypothesis_attempts: int = 0, **_: Any) -> float:
    """Penalises signs of reward-hacking behaviour."""
    score = 1.0
    score -= 0.20 * wrong_hypotheses         # −0.20 per wrong guess
    score -= 0.05 * repeated_queries          # −0.05 per repeated query
    # Penalise excessive hypothesis attempts even if some are correct
    if hypothesis_attempts > 3:
        score -= 0.10 * (hypothesis_attempts - 3)
    return max(0.0, score)


POSTMORTEM_RUBRICS = RubricSet([
    Rubric(
        name="cause_correctness",
        weight=0.40,
        description="Did the agent identify the correct root cause?",
        evaluator=_cause_correctness,
    ),
    Rubric(
        name="chain_accuracy",
        weight=0.25,
        description="How accurately did the agent reconstruct the causal chain?",
        evaluator=_chain_accuracy,
    ),
    Rubric(
        name="efficiency",
        weight=0.15,
        description="How quickly did the agent reach the answer?",
        evaluator=_efficiency,
    ),
    Rubric(
        name="investigation_quality",
        weight=0.10,
        description="How thoroughly and diversely did the agent investigate?",
        evaluator=_investigation_quality,
    ),
    Rubric(
        name="anti_gaming",
        weight=0.10,
        description="Penalty for brute-force guessing, repeated queries, and exploitation patterns.",
        evaluator=_anti_gaming,
    ),
])
"""Default rubric set for PostmortemEnv terminal scoring."""

RUBRIC_ANTI_GAMING_NOTES = {
    "cause_correctness": (
        "Cannot be gamed by guessing: wrong hypotheses trigger -0.12 per attempt "
        "in the reward_calculator and -0.20 in the anti_gaming rubric. "
        "An agent that brute-forces causes will score 0.0 here while accumulating "
        "heavy anti_gaming penalties."
    ),
    "chain_accuracy": (
        "Uses Jaccard similarity on directed node+edge sets, not substring matching. "
        "Submitting a superset of the true chain does not yield full marks — "
        "precision and recall are both required."
    ),
    "efficiency": (
        "Rewards solving in fewer steps. An agent that finds the correct cause "
        "after 38 of 40 allowed steps scores near 0 on this rubric even with "
        "perfect cause_correctness. Forces the optimizer toward targeted querying."
    ),
    "investigation_quality": (
        "Rewards both fact coverage (breadth) and action-type diversity. "
        "An agent that only calls query_logs will score 0 on diversity regardless "
        "of how many facts it finds."
    ),
    "anti_gaming": (
        "Independent of all terminal rubrics. Repeated (service, keyword) pairs "
        "trigger escalating penalties before the terminal score is computed. "
        "This signal reaches the optimizer during GRPO's per-step reward, "
        "not just at episode end."
    ),
}
