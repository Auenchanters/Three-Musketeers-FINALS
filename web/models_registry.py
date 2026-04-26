"""
Curated model registry for the live UI.

Single source of truth for the dropdown the judges use to pick a model.
Each entry records the HF repo path, parameter count, context window, tier
and an estimated USD cost per run (5K input + 2K output tokens at
HuggingFace Inference pricing as of 2026-04).

All models are paid-tier — users must supply their own HF token with
Inference permission from huggingface.co/settings/tokens.

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
    # Optional: a single HF Inference Provider this model is *only* served
    # by, e.g. ``"featherless-ai"``. The frontend uses this to warn the
    # user up-front that they need to enable that provider in their HF
    # account settings before clicking Run — otherwise the request 400s
    # with ``model_not_supported`` from the router. ``None`` for models
    # served by ≥2 providers (the safe default case).
    requires_provider: str | None = None

    def public_dict(self) -> dict:
        return asdict(self)


# Provider availability is verified against
# https://huggingface.co/api/models/<repo>?expand[]=inferenceProviderMapping
# as of 2026-04-26. Each entry is annotated with the providers that currently
# serve it on the HF Inference Providers router. Models with NO live provider
# (Llama-3.2-3B-Instruct, Phi-3.5-mini-instruct, Mistral-7B-Instruct-v0.3 as of
# 2026-04) were dropped — they 400 with `model_not_supported` for every user
# regardless of which providers they have enabled, so showing them in the UI
# is just a guaranteed dead-end.
#
# Default order: the first free entry is what new visitors land on. We pick a
# model that is live on the broadest set of providers so the request succeeds
# without the user having to manually enable a niche provider in their HF
# account settings (https://huggingface.co/settings/inference-providers).
MODELS: list[ModelInfo] = [
    # ---- All models require a user-supplied HF token -----------------------
    # Default. Live on 5 providers — Cerebras + Novita are typically enabled
    # by default on fresh HF accounts, so this is the safest first-run pick.
    ModelInfo(
        id="llama-3.1-8b-instruct",
        display_name="Llama 3.1 8B Instruct",
        repo="meta-llama/Llama-3.1-8B-Instruct",
        params_b=8.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=0.06,
        blurb="Meta's reliable 8B reasoner. Live on 5 providers — bring your own HF token.",
        license="Llama 3.1 Community",
    ),
    # Live on Together (almost always default-enabled) + Featherless.
    ModelInfo(
        id="qwen2.5-7b-instruct",
        display_name="Qwen2.5 7B Instruct",
        repo="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=0.15,
        blurb="Strongest small model. Reliable JSON, good investigation depth. HF token required.",
        license="Apache-2.0",
    ),
    # Newer Qwen3 line — live on nscale + fireworks-ai + featherless-ai.
    ModelInfo(
        id="qwen3-8b",
        display_name="Qwen3 8B",
        repo="Qwen/Qwen3-8B",
        params_b=8.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=0.10,
        blurb="Qwen's 2026 reasoning-tuned line. Strong CoT, JSON-clean. HF token required.",
        license="Apache-2.0",
    ),
    # Code-tuned 7B on nscale + featherless-ai.
    ModelInfo(
        id="qwen2.5-coder-7b-instruct",
        display_name="Qwen2.5 Coder 7B Instruct",
        repo="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_b=7.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=0.12,
        blurb="Code-tuned 7B. Best at structured action JSON + commit diffs. HF token required.",
        license="Apache-2.0",
    ),
    # ---- Larger paid models (> $1, bring your own HF token) ----------------
    # 9 providers — the broadest reach in the registry. Replaces Llama-3.1-70B,
    # which is now scaleway-only (one-provider-deep is a common 400 trigger).
    ModelInfo(
        id="llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B Instruct",
        repo="meta-llama/Llama-3.3-70B-Instruct",
        params_b=70.0,
        context_window=131_072,
        tier="paid",
        est_cost_usd=1.20,
        blurb="Frontier 70B. Live on Groq, Together, Novita, Fireworks, Sambanova + 4 more.",
        license="Llama 3.3 Community",
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
