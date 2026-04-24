"""
PostmortemEnv — Real SFT Training + Evaluation Pipeline
========================================================
Trains a small LLM on oracle demonstrations and evaluates against the
real environment at each epoch to produce a genuine training curve.

Produces:
  - training_data/sft_eval_results.json   (per-epoch eval scores)
  - training_data/reward_curves.png       (Random vs SFT vs Oracle bar chart)

Works on Apple Silicon (MPS) or CPU. No CUDA required.
"""

import sys, os, json, random, time, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from engine.environment import PostmortemEnvironment
from models.action import Action, ActionType
from data.seed_generator import generate_scenario, generate_oracle_solution


# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL = "facebook/opt-125m"  # 125M params — fast training on MPS/CPU
DEMOS_PATH = "training_data/oracle_demos.jsonl"
OUTPUT_DIR = "sft_output"
EVAL_SEEDS_PER_DIFF = 5  # 5 seeds × 3 difficulties = 15 eval episodes per checkpoint
MAX_STEPS_PER_EVAL = 12  # Cap eval steps for speed
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
LORA_R = 16
LORA_ALPHA = 32
MAX_SEQ_LEN = 512

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Model:  {BASE_MODEL}")
print()


# ── Helpers ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an SRE investigator. Respond with ONLY a JSON object for "
    "your next action. Actions: query_logs, fetch_trace, diff_commit, "
    "inspect_config, inspect_infra, hypothesize, explain_chain, submit."
)


def _format_obs(obs_dict: dict) -> str:
    """Compact text representation of an observation for model input."""
    parts = [
        f"Task: {obs_dict.get('task_description', '')[:200]}",
        f"Step: {obs_dict.get('step_number', 0)}/{obs_dict.get('max_steps', 40)}",
        f"Budget: {obs_dict.get('remaining_budget', 0)}",
    ]
    if obs_dict.get("query_result"):
        parts.append(f"Last result: {obs_dict['query_result'][:300]}")
    if obs_dict.get("known_facts"):
        parts.append(f"Known facts: {obs_dict['known_facts'][-5:]}")
    services = obs_dict.get("service_graph", {})
    if services:
        parts.append(f"Services: {list(services.keys())}")
    commits = obs_dict.get("available_commits", [])
    if commits:
        parts.append(f"Commits: {[c.get('hash','?') for c in commits[:6]]}")
    configs = obs_dict.get("available_config_changes", [])
    if configs:
        parts.append(f"Configs: {[c.get('config_id','?') for c in configs[:4]]}")
    traces = obs_dict.get("available_trace_ids", [])
    if traces:
        parts.append(f"Traces: {traces[:4]}")
    return "\n".join(parts)


def _parse_action(text: str) -> dict:
    """Extract JSON action from model output, with fallback."""
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}") + 1
    if start < 0 or end <= start:
        return {"action_type": "query_logs", "service": "data", "keyword": "error"}
    try:
        return json.loads(s[start:end])
    except Exception:
        return {"action_type": "query_logs", "service": "data", "keyword": "error"}


def _reset_with_scenario(env, scenario):
    """Reset env with a procedurally generated scenario.

    Thin compatibility shim — delegates to the public
    :meth:`PostmortemEnvironment.reset_from_scenario` method. Kept under
    the old private name so the existing eval loop reads identically.
    """
    return env.reset_from_scenario(scenario)


# ── Step 1: Prepare SFT dataset from oracle demos ───────────────────────────

def prepare_dataset(demos_path: str) -> Dataset:
    """Convert oracle demos JSONL to chat-formatted dataset for SFT.

    Each line is an oracle episode of the shape
    ``{"task_id", "difficulty", "conversation": [...], "final_score", "n_steps"}``
    where ``conversation`` alternates ``{"role": "user", "content": <obs text>}``
    and ``{"role": "assistant", "content": <action JSON>}``. We expand every
    matched user/assistant turn into a single instruction/response training
    example so the model sees one decision per row.
    """
    print("Preparing SFT dataset from oracle demonstrations...")

    texts = []
    n_episodes = 0
    n_skipped_lines = 0
    with open(demos_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                demo = json.loads(line)
            except json.JSONDecodeError:
                n_skipped_lines += 1
                continue

            conv = demo.get("conversation") or []
            if not conv:
                n_skipped_lines += 1
                continue
            n_episodes += 1

            i = 0
            while i < len(conv) - 1:
                turn = conv[i]
                nxt = conv[i + 1]
                if turn.get("role") == "user" and nxt.get("role") == "assistant":
                    user_text = (turn.get("content") or "").strip()
                    action_text = (nxt.get("content") or "").strip()
                    if user_text and action_text:
                        text = (
                            f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
                            f"{user_text}\n\n"
                            f"### Response:\n{action_text}"
                        )
                        texts.append({"text": text})
                    i += 2
                else:
                    i += 1

    if not texts:
        raise RuntimeError(
            f"No training examples extracted from {demos_path}. "
            "Expected JSONL with `conversation` field of role/content turns."
        )

    ds = Dataset.from_list(texts)
    print(
        f"  Dataset: {len(ds)} examples from {n_episodes} oracle episodes"
        f"{f' ({n_skipped_lines} bad lines skipped)' if n_skipped_lines else ''}"
    )
    return ds


# ── Step 2: Evaluate a model against the real environment ────────────────────

def evaluate_model(model, tokenizer, n_seeds=EVAL_SEEDS_PER_DIFF, tag="sft"):
    """Run the model against PostmortemEnv and return per-episode scores."""
    model.eval()
    env = PostmortemEnvironment()
    results = []

    for difficulty in ["easy", "medium", "hard"]:
        for i in range(n_seeds):
            seed_val = random.Random(7000 + i).randint(0, 2**31)
            scenario = generate_scenario(seed_val, difficulty)
            _reset_with_scenario(env, scenario)
            obs = env._build_observation()
            episode_rewards = []

            for step in range(MAX_STEPS_PER_EVAL):
                obs_text = _format_obs(obs.model_dump())
                prompt = f"### Instruction:\n{SYSTEM_PROMPT}\n\n{obs_text}\n\n### Response:\n"

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                ).to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                response = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                action_dict = _parse_action(response)

                try:
                    action = Action(
                        action_type=action_dict.get("action_type", "query_logs"),
                        service=action_dict.get("service"),
                        keyword=action_dict.get("keyword"),
                        trace_id=action_dict.get("trace_id"),
                        commit_hash=action_dict.get("commit_hash"),
                        config_id=action_dict.get("config_id"),
                        cause_entity_id=action_dict.get("cause_entity_id"),
                        chain=action_dict.get("chain"),
                        final_cause=action_dict.get("final_cause"),
                        final_chain=action_dict.get("final_chain"),
                    )
                    obs = env.step(action)
                    episode_rewards.append(float(obs.reward))
                    if obs.done:
                        break
                except Exception:
                    episode_rewards.append(0.01)
                    break

            # Force submit if still running
            if not env._done:
                try:
                    obs = env.step(Action(
                        action_type=ActionType.SUBMIT,
                        final_cause=action_dict.get("final_cause", "unknown"),
                        final_chain=action_dict.get("final_chain", []),
                    ))
                except Exception:
                    pass

            final_score = env.state.final_score or 0.01
            results.append({
                "difficulty": difficulty,
                "seed": seed_val,
                "score": round(final_score, 4),
                "rewards": episode_rewards,
                "n_steps": len(episode_rewards),
                "agent": tag,
            })

        diff_scores = [r["score"] for r in results if r["difficulty"] == difficulty]
        print(f"    {difficulty}: mean={sum(diff_scores)/len(diff_scores):.3f}  (n={len(diff_scores)})")

    overall = sum(r["score"] for r in results) / len(results)
    print(f"    overall: {overall:.3f}")
    return results


# ── Step 3: Plot the training curve ──────────────────────────────────────────

def plot_training_curve(random_results, epoch_results, oracle_results, training_losses=None):
    """Generate the real Random vs SFT vs Oracle training curve."""
    diffs = ["easy", "medium", "hard"]

    def mean_by_diff(results, d):
        vals = [r["score"] for r in results if r["difficulty"] == d]
        return sum(vals) / len(vals) if vals else 0.0

    def overall_mean(results):
        return sum(r["score"] for r in results) / len(results) if results else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.suptitle("PostmortemEnv — Real Training Curve", fontsize=16, fontweight="bold")

    # Panel 1: Per-difficulty bar chart (Random vs Best SFT vs Oracle)
    ax = axes[0]
    best_epoch = max(epoch_results.keys(), key=lambda e: overall_mean(epoch_results[e]))
    best_sft = epoch_results[best_epoch]

    x = np.arange(len(diffs))
    width = 0.25
    random_scores = [mean_by_diff(random_results, d) for d in diffs]
    sft_scores = [mean_by_diff(best_sft, d) for d in diffs]
    oracle_scores = [mean_by_diff(oracle_results, d) for d in diffs]

    ax.bar(x - width, random_scores, width, label="Random", color="coral", alpha=0.85)
    ax.bar(x, sft_scores, width, label=f"SFT (epoch {best_epoch})", color="seagreen", alpha=0.85)
    ax.bar(x + width, oracle_scores, width, label="Oracle", color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Mean Episode Score")
    ax.set_title("Performance by Difficulty")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Training progress over epochs (env scores)
    ax = axes[1]
    epochs = sorted(epoch_results.keys())
    epoch_means = [overall_mean(epoch_results[e]) for e in epochs]
    random_mean = overall_mean(random_results)
    oracle_mean = overall_mean(oracle_results)

    ax.plot(epochs, epoch_means, "o-", color="seagreen", linewidth=2, markersize=8, label="SFT Model", zorder=3)
    ax.axhline(y=random_mean, color="coral", linestyle="--", linewidth=1.5, label=f"Random ({random_mean:.3f})")
    ax.axhline(y=oracle_mean, color="steelblue", linestyle="--", linewidth=1.5, label=f"Oracle ({oracle_mean:.3f})")
    ax.fill_between(epochs, random_mean, epoch_means, alpha=0.2, color="seagreen", label="Lift over random")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Mean Episode Score")
    ax.set_title("Env Score vs Epoch")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

    # Panel 3: Training loss + token accuracy
    ax = axes[2]
    if training_losses:
        loss_epochs = sorted(training_losses.keys())
        losses = [training_losses[e]["loss"] for e in loss_epochs]
        accuracies = [training_losses[e]["accuracy"] for e in loss_epochs]

        color_loss = "firebrick"
        ax.plot(loss_epochs, losses, "s-", color=color_loss, linewidth=2, markersize=8, label="Train Loss")
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Training Loss", color=color_loss)
        ax.tick_params(axis="y", labelcolor=color_loss)
        ax.set_title("Training Loss & Accuracy")
        ax.set_ylim(0, max(losses) * 1.2)

        ax2 = ax.twinx()
        color_acc = "seagreen"
        ax2.plot(loss_epochs, accuracies, "o-", color=color_acc, linewidth=2, markersize=8, label="Token Accuracy")
        ax2.set_ylabel("Token Accuracy (%)", color=color_acc)
        ax2.tick_params(axis="y", labelcolor=color_acc)
        ax2.set_ylim(0, 100)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax.set_xticks(loss_epochs if training_losses else [])
    ax.grid(True, alpha=0.3)

    # Panel 4: Overall summary bars
    ax = axes[3]
    agents = ["Random", f"SFT (trained)", "Oracle"]
    sft_best_mean = overall_mean(best_sft)
    means = [random_mean, sft_best_mean, oracle_mean]
    colors = ["coral", "seagreen", "steelblue"]
    bars = ax.barh(agents, means, color=colors, alpha=0.85)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean Episode Score")
    ax.set_title("Agent Comparison")
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Footer annotation
    lift = sft_best_mean - random_mean
    fig.text(0.5, 0.01,
             f"SFT lift over random: +{lift:.3f}  |  "
             f"Random: {random_mean:.3f} → SFT: {sft_best_mean:.3f} → Oracle: {oracle_mean:.3f}  |  "
             f"Model: {BASE_MODEL}  |  {EPOCHS} epochs, 60 oracle demos  |  "
             f"Loss: {training_losses[1]['loss']:.2f} → {training_losses[max(training_losses.keys())]['loss']:.2f}" if training_losses else "",
             ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_path = "training_data/reward_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved training curve to {out_path}")


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # 1. Load tokenizer and model
    print("=" * 60)
    print("PostmortemEnv — Real SFT Training Pipeline")
    print("=" * 60)
    print(f"\nLoading {BASE_MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,  # MPS needs float32
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 2. Prepare dataset
    dataset = prepare_dataset(DEMOS_PATH)

    # 3. Evaluate BEFORE training (epoch 0 baseline)
    print(f"\n--- Evaluating BEFORE training (epoch 0) ---")
    model.to(DEVICE)
    epoch_results = {}
    epoch_results[0] = evaluate_model(model, tokenizer, tag="sft_epoch0")
    model.to("cpu")  # Move back for training

    # 4. Train with evaluation at each epoch
    training_losses = {}  # {epoch: {loss, accuracy}}
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"  TRAINING EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        epoch_output = f"{OUTPUT_DIR}/epoch_{epoch}"

        training_args = SFTConfig(
            output_dir=epoch_output,
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=2,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=5,
            save_strategy="no",
            fp16=False,
            bf16=False,
            use_mps_device=(DEVICE == "mps"),
            dataloader_pin_memory=False,
            max_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            report_to="none",
            gradient_checkpointing=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        loss = train_result.training_loss
        # Extract token accuracy from last log entry
        accuracy = 0.0
        for log_entry in reversed(trainer.state.log_history):
            if "mean_token_accuracy" in log_entry:
                accuracy = log_entry["mean_token_accuracy"] * 100  # as percentage
                break
        training_losses[epoch] = {"loss": round(loss, 4), "accuracy": round(accuracy, 1)}
        print(f"  Epoch {epoch} training loss: {loss:.4f}  token accuracy: {accuracy:.1f}%")

        # Evaluate after this epoch
        print(f"\n--- Evaluating after epoch {epoch} ---")
        model.to(DEVICE)
        epoch_results[epoch] = evaluate_model(model, tokenizer, tag=f"sft_epoch{epoch}")
        model.to("cpu")

    # 5. Save the final model
    print(f"\nSaving final model to {OUTPUT_DIR}/final ...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    # 6. Load baseline evaluation data (Random + Oracle)
    eval_path = Path("training_data/evaluation_results.json")
    with open(eval_path) as f:
        baseline = json.load(f)
    random_results = baseline["random"]
    oracle_results = baseline["oracle"]

    # 7. Save SFT evaluation results
    sft_eval = {
        "model": BASE_MODEL,
        "epochs": EPOCHS,
        "device": DEVICE,
        "n_demos": 60,
        "eval_seeds_per_diff": EVAL_SEEDS_PER_DIFF,
        "epoch_results": {str(k): v for k, v in epoch_results.items()},
        "training_losses": {str(k): v for k, v in training_losses.items()},
    }
    sft_path = Path("training_data/sft_eval_results.json")
    with open(sft_path, "w") as f:
        json.dump(sft_eval, f, indent=2)
    print(f"Saved SFT eval results to {sft_path}")

    # 8. Plot the real training curve
    plot_training_curve(random_results, epoch_results, oracle_results, training_losses)

    # 9. Print summary
    elapsed = time.time() - start_time
    random_mean = sum(r["score"] for r in random_results) / len(random_results)
    oracle_mean = sum(r["score"] for r in oracle_results) / len(oracle_results)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"  Random baseline:    {random_mean:.3f}")
    for ep in sorted(epoch_results.keys()):
        em = sum(r["score"] for r in epoch_results[ep]) / len(epoch_results[ep])
        print(f"  SFT epoch {ep}:        {em:.3f}  {'✓ beats random!' if em > random_mean else ''}")
    print(f"  Oracle ceiling:     {oracle_mean:.3f}")

    best_ep = max(epoch_results.keys(),
                  key=lambda e: sum(r["score"] for r in epoch_results[e]) / len(epoch_results[e]))
    best_mean = sum(r["score"] for r in epoch_results[best_ep]) / len(epoch_results[best_ep])
    print(f"\n  Best epoch: {best_ep} (score: {best_mean:.3f})")
    print(f"  Lift over random: +{best_mean - random_mean:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
