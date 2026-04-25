"""
Curated model registry for the live UI.

Single source of truth for the dropdown the judges use to pick a model.
Each entry records the HF repo path, parameter count, context window, tier
("free" → run on the server's HF token; "paid" → require a user-supplied
HF token), and an estimated USD cost per run (5K input + 2K output tokens
at HuggingFace Inference pricing as of 2026-04).

Costs are estimates only — display them with a "~" prefix in the UI and
explain that real cost depends on traffic and prompt size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Tier = Literal["free", "paid"]


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    repo: str
    params_b: float
    context_window: int
    tier: Tier
    est_cost_usd: float
    blurb: str
    license: str

    def public_dict(self) -> dict:
        return asdict(self)


MODELS: list[ModelInfo] = [
    # ---- Free tier (≤ $1, server's HF token) -------------------------------
    ModelInfo(
        id="qwen2.5-1.5b-instruct",
        display_name="Qwen2.5 1.5B Instruct",
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.5,
        context_window=32_768,
        tier="free",
        est_cost_usd=0.02,
        blurb="Compact, JSON-clean, fastest free option. The default investigator.",
        license="Apache-2.0",
    ),
    ModelInfo(
        id="llama-3.2-3b-instruct",
        display_name="Llama 3.2 3B Instruct",
        repo="meta-llama/Llama-3.2-3B-Instruct",
        params_b=3.0,
        context_window=131_072,
        tier="free",
        est_cost_usd=0.08,
        blurb="Meta's lightweight reasoner. Long context, balanced.",
        license="Llama 3.2 Community",
    ),
    ModelInfo(
        id="phi-3.5-mini-instruct",
        display_name="Phi-3.5 mini Instruct",
        repo="microsoft/Phi-3.5-mini-instruct",
        params_b=3.8,
        context_window=131_072,
        tier="free",
        est_cost_usd=0.10,
        blurb="Microsoft's reasoning-tuned small model. Strong tool-use.",
        license="MIT",
    ),
    ModelInfo(
        id="qwen2.5-7b-instruct",
        display_name="Qwen2.5 7B Instruct",
        repo="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.0,
        context_window=131_072,
        tier="free",
        est_cost_usd=0.15,
        blurb="The strongest free pick. Reliable JSON, good investigation depth.",
        license="Apache-2.0",
    ),
    ModelInfo(
        id="mistral-7b-instruct-v0.3",
        display_name="Mistral 7B Instruct v0.3",
        repo="mistralai/Mistral-7B-Instruct-v0.3",
        params_b=7.0,
        context_window=32_768,
        tier="free",
        est_cost_usd=0.18,
        blurb="Mistral's classic. Fast, articulate, function-calling capable.",
        license="Apache-2.0",
    ),
    # ---- Paid tier (> $1, bring your own HF token) -------------------------
    ModelInfo(
        id="llama-3.1-70b-instruct",
        display_name="Llama 3.1 70B Instruct",
        repo="meta-llama/Llama-3.1-70B-Instruct",
        params_b=70.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=1.20,
        blurb="Frontier-class open weights. Best Llama for hard cases.",
        license="Llama 3.1 Community",
    ),
    ModelInfo(
        id="qwen2.5-72b-instruct",
        display_name="Qwen2.5 72B Instruct",
        repo="Qwen/Qwen2.5-72B-Instruct",
        params_b=72.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=1.30,
        blurb="Top-tier reasoning, multilingual. Pairs with PyTorch backends.",
        license="Qwen License",
    ),
    ModelInfo(
        id="deepseek-r1-distill-llama-70b",
        display_name="DeepSeek R1 Distill (Llama 70B)",
        repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        params_b=70.0,
        context_window=32_768,
        tier="paid",
        est_cost_usd=1.50,
        blurb="Reasoning-distilled. Tends to think harder, cost a bit more.",
        license="MIT",
    ),
]


def get_model(model_id: str) -> ModelInfo | None:
    for m in MODELS:
        if m.id == model_id:
            return m
    return None


def public_list(free_tier_available: bool) -> list[dict]:
    """Serializable form for GET /api/models."""
    out = []
    for m in MODELS:
        d = m.public_dict()
        d["free_tier_available"] = free_tier_available if m.tier == "free" else True
        out.append(d)
    return out
