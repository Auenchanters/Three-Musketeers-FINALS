# PostmortemEnv Training Notebook
# ================================
# This notebook demonstrates training an LLM on PostmortemEnv.
# Run in Google Colab with a T4 GPU for best results.
#
# Meta PyTorch OpenEnv Hackathon Apr '26
# Team: Three Musketeers (Utkarsh, Mohit, Tanush)
#
# Steps:
#   1. Install dependencies
#   2. Clone & setup PostmortemEnv
#   3. Collect oracle demonstrations
#   4. Show reward gap (random vs oracle)
#   5. SFT on oracle demonstrations
#   6. GRPO with environment rewards
#   7. Plot training curves

# %%
# Cell 1: Install dependencies
# !pip install -q openenv-core trl peft datasets transformers accelerate bitsandbytes
# !pip install -q matplotlib httpx openai websockets pydantic

# %%
# Cell 2: Setup
import sys, os, json, random, math
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
from train import collect_demonstrations, evaluate_agents, _reset_with_scenario

demos_path = collect_demonstrations(n_seeds=20, output_path="training_data")
print(f"\nDemonstrations saved to: {demos_path}")

# %%
# Cell 5: Evaluate reward gap
results = evaluate_agents(n_seeds=10)

# %%
# Cell 6: Plot reward curves
from train import plot_rewards
plot_rewards()

# Display the plots
from IPython.display import Image, display
if Path("training_data/reward_curves.png").exists():
    display(Image("training_data/reward_curves.png"))
if Path("training_data/learning_curve.png").exists():
    display(Image("training_data/learning_curve.png"))

# %%
# Cell 7: SFT Training (requires GPU)
# Uncomment to run on Colab T4
#
# from train import run_sft
# run_sft(
#     model_name="meta-llama/Llama-3.2-1B-Instruct",
#     demos_path="training_data/oracle_demos.jsonl",
#     output_dir="sft_output",
#     epochs=3,
#     batch_size=2,
#     lr=2e-5,
# )

# %%
# Cell 8: GRPO Training (requires GPU)
# Uncomment to run on Colab T4
#
# from train import run_grpo
# run_grpo(
#     model_name="./sft_output",
#     output_dir="grpo_output",
#     n_episodes=100,
#     batch_size=4,
#     lr=1e-6,
# )

# %%
# Cell 9: Final summary
print("=" * 60)
print("PostmortemEnv Training Pipeline Summary")
print("=" * 60)
print()
print("Environment: PostmortemEnv v1.0.0")
print("Theme: 3.1 World Modeling (Professional)")
print("Team: Three Musketeers")
print()
print("Key Results:")

# Load eval results
eval_path = Path("training_data/evaluation_results.json")
if eval_path.exists():
    with open(eval_path) as f:
        res = json.load(f)
    r_mean = sum(r["score"] for r in res["random"]) / len(res["random"])
    o_mean = sum(r["score"] for r in res["oracle"]) / len(res["oracle"])
    print(f"  Random agent:  {r_mean:.3f}")
    print(f"  Oracle agent:  {o_mean:.3f}")
    print(f"  Reward gap:    {o_mean - r_mean:.3f} ({(o_mean - r_mean) * 100:.1f}%)")
    print()
    print("  This gap demonstrates PostmortemEnv provides a meaningful")
    print("  training signal for improving LLM causal reasoning.")

print()
print("Training Pipeline:")
print("  1. Collect oracle demos:  python train.py collect")
print("  2. SFT on demonstrations: python train.py sft")
print("  3. GRPO with rewards:     python train.py grpo")
print("  4. Plot curves:           python train.py plot")
print("=" * 60)
