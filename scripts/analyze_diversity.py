"""
Compute diversity statistics across procedurally generated scenarios.
Run: python scripts/analyze_diversity.py
Paste the output block into the README under a 'Scenario Diversity' section.
"""

import random
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.seed_generator import generate_scenario


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def scenario_fingerprint(scenario: dict) -> frozenset:
    """A hashable representation of a scenario's key facts."""
    facts = set()
    facts.add("root_cause:" + str(scenario.get("root_cause_entity", "")))
    for svc in scenario.get("services", []):
        facts.add("svc:" + svc.get("name", ""))
    for commit in scenario.get("commits", [])[:3]:
        facts.add("commit:" + commit.get("hash", ""))
    return frozenset(facts)


def main(n_samples: int = 500):
    print("Analyzing %d scenarios across 3 difficulty tiers..." % n_samples)

    root_causes = []
    service_distributions = {"frontend": 0, "auth": 0, "data": 0, "batch": 0}
    chain_lengths = []
    log_densities = []
    fingerprints = []

    rng = random.Random(0)
    difficulties = ["easy", "medium", "hard"]

    for i in range(n_samples):
        seed = rng.randint(0, 2**31)
        diff = difficulties[i % 3]
        scenario = generate_scenario(seed, diff)

        root_causes.append(scenario.get("root_cause_entity", "unknown"))

        for svc in scenario.get("services", []):
            name = svc.get("name", "")
            if name in service_distributions:
                service_distributions[name] += 1

        chain = scenario.get("causal_chain", [])
        chain_lengths.append(len(chain))

        total_logs = sum(
            len(svc.get("logs", [])) for svc in scenario.get("services", [])
        )
        log_densities.append(total_logs)

        fingerprints.append(scenario_fingerprint(scenario))

    # Jaccard pairwise similarity on a random sample of 200 pairs
    sample_pairs = [(rng.randint(0, n_samples - 1), rng.randint(0, n_samples - 1))
                    for _ in range(200)]
    similarities = [
        jaccard(set(fingerprints[a]), set(fingerprints[b]))
        for a, b in sample_pairs if a != b
    ]
    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0

    unique_causes = len(set(root_causes))
    total_svcs = sum(service_distributions.values()) or 1
    avg_chain = sum(chain_lengths) / len(chain_lengths)
    std_chain = (sum((x - avg_chain) ** 2 for x in chain_lengths) / len(chain_lengths)) ** 0.5
    avg_logs = sum(log_densities) / len(log_densities)

    print("\nScenario Diversity Analysis (n=%d)" % n_samples)
    print("=" * 50)
    print("Unique root cause entities : %d" % unique_causes)
    print("Service distribution       :")
    for name, count in service_distributions.items():
        pct = count / total_svcs * 100
        print("  %-12s : %5.1f%%" % (name, pct))
    print("Causal chain length        : mean=%.1f  std=%.1f" % (avg_chain, std_chain))
    print("Log count per scenario     : mean=%.1f" % avg_logs)
    print("Pairwise Jaccard similarity: mean=%.3f (lower = more diverse)" % mean_sim)
    print("=" * 50)
    print("\nInterpretation:")
    print("  A Jaccard similarity < 0.35 indicates the scenarios are meaningfully")
    print("  distinct. An agent cannot memorize one scenario's solution and apply")
    print("  it directly to another.")

    out = {
        "n_samples": n_samples,
        "unique_root_causes": unique_causes,
        "service_distribution_pct": {k: round(v / total_svcs * 100, 1) for k, v in service_distributions.items()},
        "chain_length_mean": round(avg_chain, 2),
        "chain_length_std": round(std_chain, 2),
        "log_density_mean": round(avg_logs, 1),
        "pairwise_jaccard_mean": round(mean_sim, 3),
    }
    out_path = Path("training_data") / "diversity_stats.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nStats saved to %s" % out_path)


if __name__ == "__main__":
    main()
