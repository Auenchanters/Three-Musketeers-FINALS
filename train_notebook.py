# PostmortemEnv Training Notebook
# ================================
# This notebook demonstrates training an LLM on PostmortemEnv.
# Run in Google Colab with a T4 GPU for best results.
#
# Meta PyTorch OpenEnv Hackathon Apr '26
# Team: Three Musketeers (Utkarsh, Mohit, Tanush)
#
# Pipeline:
#   1. Install deps
#   2. Verify env
#   3. Collect oracle demos
#   4. Evaluate baselines (Random vs Oracle) → real reward gap
#   5. SFT the 1B Llama on oracle demos
#   6. Evaluate the SFT-trained model on held-out seeds
#   7. Plot Random vs SFT vs Oracle
#   8. (stretch) GRPO with environment rewards

# %%
# Cell 1: Install dependencies
# !pip install -q -r requirements-train.txt

# %%
# Cell 2: Setup
import sys, os, json, random
from pathlib import Path

# If running from cloned repo
sys.path.insert(0, ".")

# %%
# Cell 3: Verify environment works
from engine.environment import PostmortemEnvironment
from models.action import Action, ActionType
from data.generator import load_scenario

env = PostmortemEnvironment()
obs = env.reset(task_id="task1_recent_deploy")
print("Environment initialized!")
print(f"  Task: {obs.task_description[:100]}")
print(f"  Services: {[s.name for s in obs.services]}")
print(f"  Budget: {obs.remaining_budget} steps")
print(f"  Commits: {len(obs.available_commits)}")
print()

# Quick action test
action = Action(action_type=ActionType.QUERY_LOGS, service="data", keyword="error")
obs = env.step(action)
print(f"After query_logs('data', 'error'):")
print(f"  Reward: {obs.reward}")
print(f"  Facts: {len(obs.known_facts)}")
print(f"  Result: {obs.query_result[:200]}")

# %%
# Cell 4: Collect oracle demonstrations
from train import collect_demonstrations, evaluate_agents, _reset_with_scenario, _format_obs_compact
from training_utils import SYSTEM_PROMPT, action_from_dict, parse_action_json

demos_path = collect_demonstrations(n_seeds=20, output_path="training_data")
print(f"\nDemonstrations saved to: {demos_path}")

# %%
# Cell 5: Evaluate baseline reward gap (Random vs Oracle — real data)
baseline_results = evaluate_agents(n_seeds=10)

# %%
# Cell 6: Plot Random vs Oracle
from train import plot_rewards
plot_rewards()

from IPython.display import Image, display
if Path("training_data/reward_curves.png").exists():
    display(Image("training_data/reward_curves.png"))

# %%
# Cell 7: SFT on oracle demonstrations (requires GPU — Colab T4)
from train import run_sft
run_sft(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    demos_path="training_data/oracle_demos.jsonl",
    output_dir="sft_output",
    epochs=3,
    batch_size=2,
    lr=2e-5,
)

# %%
# Cell 8: Evaluate the SFT-trained model on held-out seeds
#
# We pit the SFT-trained policy against 10 fresh seeds per difficulty and record
# its final episode score. This is the "real reward curve" number — the trained
# mean must beat the random mean to claim improvement.

from data.seed_generator import generate_scenario
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SFT_DIR = "sft_output"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 200
MAX_STEPS_PER_EVAL = 15  # cap steps to keep eval fast

print("Loading SFT model for evaluation...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
sft_model = PeftModel.from_pretrained(base, SFT_DIR)
sft_model.eval()


def _sample_action(obs_text: str) -> dict:
    """Run the SFT model once and parse the first JSON object it emits."""
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{obs_text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(sft_model.device)
    with torch.no_grad():
        out = sft_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_action_json(text)


sft_results = []
env = PostmortemEnvironment()
print("Evaluating SFT model on 10 held-out seeds per difficulty...")
for difficulty in ["easy", "medium", "hard"]:
    for i in range(10):
        # Use a *different* seed namespace than evaluate_agents() (which uses 1000+i).
        # Held-out: 5000+i so train/eval don't overlap.
        seed = random.Random(5000 + i).randint(0, 2**31)
        scenario = generate_scenario(seed, difficulty)
        _reset_with_scenario(env, scenario)
        obs = env._build_observation()
        episode_rewards = []
        for step in range(MAX_STEPS_PER_EVAL):
            obs_text = _format_obs_compact(obs.model_dump())
            action_dict = _sample_action(obs_text)
            try:
                action = action_from_dict(action_dict)
                obs = env.step(action)
                episode_rewards.append(float(obs.reward))
                if obs.done:
                    break
            except Exception as e:
                episode_rewards.append(0.01)
                break
        # Force submit if still running
        if not env._done:
            try:
                obs = env.step(Action(
                    action_type=ActionType.SUBMIT,
                    final_cause=action_dict.get("final_cause", "unknown"),
                    final_chain=action_dict.get("final_chain", [{"service": "data", "effect": "unknown"}]),
                ))
                episode_rewards.append(float(obs.reward))
            except Exception:
                pass
        final_score = env.state.final_score or 0.01
        sft_results.append({
            "difficulty": difficulty,
            "seed": seed,
            "score": final_score,
            "rewards": episode_rewards,
            "n_steps": len(episode_rewards),
        })
    scores = [r["score"] for r in sft_results if r["difficulty"] == difficulty]
    print(f"  {difficulty}: mean={sum(scores)/len(scores):.3f}  n={len(scores)}")

with open("training_data/sft_eval_results.json", "w") as f:
    json.dump(sft_results, f, indent=2)

sft_mean = sum(r["score"] for r in sft_results) / len(sft_results)
random_mean = sum(r["score"] for r in baseline_results["random"]) / len(baseline_results["random"])
oracle_mean = sum(r["score"] for r in baseline_results["oracle"]) / len(baseline_results["oracle"])
print(f"\n  Random baseline: {random_mean:.3f}")
print(f"  SFT trained:     {sft_mean:.3f}")
print(f"  Oracle ceiling:  {oracle_mean:.3f}")
print(f"  SFT lift over random: +{sft_mean - random_mean:.3f}")

# %%
# Cell 9: Plot Random vs SFT vs Oracle (real training curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

diffs = ["easy", "medium", "hard"]
random_by_diff = [sum(r["score"] for r in baseline_results["random"] if r["difficulty"] == d) /
                  max(1, sum(1 for r in baseline_results["random"] if r["difficulty"] == d))
                  for d in diffs]
sft_by_diff = [sum(r["score"] for r in sft_results if r["difficulty"] == d) /
               max(1, sum(1 for r in sft_results if r["difficulty"] == d))
               for d in diffs]
oracle_by_diff = [sum(r["score"] for r in baseline_results["oracle"] if r["difficulty"] == d) /
                  max(1, sum(1 for r in baseline_results["oracle"] if r["difficulty"] == d))
                  for d in diffs]

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(diffs))
width = 0.25
ax.bar([i - width for i in x], random_by_diff, width, label="Random", color="coral", alpha=0.85)
ax.bar(list(x), sft_by_diff, width, label="SFT (Llama-3.2-1B)", color="seagreen", alpha=0.85)
ax.bar([i + width for i in x], oracle_by_diff, width, label="Oracle", color="steelblue", alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(["Easy", "Medium", "Hard"])
ax.set_ylabel("Mean Episode Score")
ax.set_title("PostmortemEnv — Random vs SFT vs Oracle (real data)")
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("training_data/reward_curves.png", dpi=150, bbox_inches="tight")
plt.close()

display(Image("training_data/reward_curves.png"))

# %%
# Cell 10: GRPO with environment rewards
# We run the GRPO loop to show self-improvement based on environment rewards.

from train import run_grpo
run_grpo(
    model_name="./sft_output",
    output_dir="grpo_output",
    n_episodes=100,
    batch_size=4,
    lr=1e-6,
)

# %%
# Cell 11: Final summary
print("=" * 60)
print("PostmortemEnv Training Pipeline Summary")
print("=" * 60)
print()
print("Environment: PostmortemEnv v1.0.0")
print("Theme: 3.1 World Modeling (Professional)")
print("Sub-prize target: Scaler AI Labs")
print("Team: Three Musketeers")
print()
print("Results:")

eval_path = Path("training_data/evaluation_results.json")
sft_path = Path("training_data/sft_eval_results.json")
if eval_path.exists():
    with open(eval_path) as f:
        res = json.load(f)
    r_mean = sum(r["score"] for r in res["random"]) / len(res["random"])
    o_mean = sum(r["score"] for r in res["oracle"]) / len(res["oracle"])
    print(f"  Random:  {r_mean:.3f}")
    if sft_path.exists():
        with open(sft_path) as f:
            sres = json.load(f)
        s_mean = sum(r["score"] for r in sres) / len(sres)
        print(f"  SFT:     {s_mean:.3f}  (lift over random: +{s_mean - r_mean:.3f})")
    print("  Oracle:  {o_mean:.3f}")
    print()
    print("  The GRPO loop provides the RL self-improvement signal")
    print("  the 20% rubric item asks for, driven by the deterministic environment reward.")

print()
print("Repro commands:")
print("  python train.py collect --n-seeds 20")
print("  python train.py evaluate --n-seeds 10")
print("  python train.py sft --model meta-llama/Llama-3.2-1B-Instruct")
print("  python train.py plot")
print("=" * 60)
