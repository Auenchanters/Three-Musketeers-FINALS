"""
PostmortemEnv — Procedural Scenario Seed Generator

Generates unlimited deterministic outage scenarios from a set of failure
templates and randomized parameters. Each seed produces a unique but
coherent frozen telemetry bundle with ground truth.

Design:
- 5 failure templates × parameterized variations = unlimited scenarios
- Deterministic: same seed → exact same scenario
- Each scenario has 4 services, realistic logs/traces/commits/configs
- Oracle-labeled relevant facts for reward computation
"""

import json
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ------------------------------------------------------------------
# Failure Templates
# ------------------------------------------------------------------

FAILURE_TEMPLATES = {
    "connection_pool": {
        "description_template": "The '{target_service}' service experienced a spike in {error_type} errors at {incident_time} UTC. A recent deployment changed the connection pool configuration, reducing capacity and causing cascading failures.",
        "root_cause_type": "commit",
        "chain_template": [
            {"service": "{target_service}", "effect": "connection_pool_exhaustion"},
            {"service": "{upstream_1}", "effect": "upstream_timeout"},
            {"service": "{upstream_2}", "effect": "5xx_errors_to_users"},
        ],
    },
    "oom_cascade": {
        "description_template": "A cascading outage affected all services starting at {incident_time} UTC. The '{target_service}' service crashed repeatedly, propagating failures through the dependency chain.",
        "root_cause_type": "commit",
        "chain_template": [
            {"service": "{target_service}", "effect": "oom_crash_loop"},
            {"service": "{upstream_1}", "effect": "dependency_timeout"},
            {"service": "{upstream_2}", "effect": "service_degradation"},
            {"service": "{upstream_3}", "effect": "complete_unavailability"},
        ],
    },
    "config_drift": {
        "description_template": "The '{target_service}' service began returning errors at {incident_time} UTC after a configuration change modified a critical parameter. The change appeared safe but interacted poorly with the current load pattern.",
        "root_cause_type": "config",
        "chain_template": [
            {"service": "{target_service}", "effect": "resource_limit_exceeded"},
            {"service": "{upstream_1}", "effect": "cascading_timeout"},
            {"service": "{upstream_2}", "effect": "user_facing_errors"},
        ],
    },
    "failover_bug": {
        "description_template": "A network event at {incident_time} UTC triggered an automatic failover in the '{target_service}' service. A bug in the failover logic left the service in a broken state with zero active connections.",
        "root_cause_type": "correlated",
        "chain_template": [
            {"service": "{target_service}", "effect": "network_degradation"},
            {"service": "{target_service}", "effect": "failover_triggered"},
            {"service": "{upstream_1}", "effect": "connection_failure"},
            {"service": "{upstream_1}", "effect": "zero_connections_state"},
            {"service": "{upstream_2}", "effect": "complete_service_failure"},
        ],
    },
    "memory_leak": {
        "description_template": "The '{target_service}' service experienced gradually increasing memory usage starting {hours_before}h before the outage at {incident_time} UTC. A recent deployment introduced a memory leak in the {leak_component} component.",
        "root_cause_type": "commit",
        "chain_template": [
            {"service": "{target_service}", "effect": "memory_leak_gc_pressure"},
            {"service": "{target_service}", "effect": "response_latency_spike"},
            {"service": "{upstream_1}", "effect": "timeout_cascade"},
            {"service": "{upstream_2}", "effect": "circuit_breaker_open"},
        ],
    },
}

# Realistic names and components
SERVICE_NAMES = ["frontend", "auth", "data", "batch"]
DEVELOPER_NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"]
CONFIG_KEYS = [
    "MAX_CONNECTIONS", "POOL_SIZE", "TIMEOUT_MS", "RETRY_COUNT",
    "CACHE_TTL", "RATE_LIMIT", "MEMORY_LIMIT", "WORKER_COUNT",
    "QUEUE_DEPTH", "BATCH_SIZE", "LOG_LEVEL", "HEALTH_CHECK_INTERVAL",
]
ERROR_TYPES = [
    "ConnectionPoolExhausted", "TimeoutError", "OutOfMemoryError",
    "CircuitBreakerOpen", "DNSResolutionFailed", "TLSHandshakeTimeout",
    "RateLimitExceeded", "DiskSpaceExhausted", "DeadlockDetected",
]
OPERATIONS = [
    "/api/v1/users", "/api/v1/auth/validate", "/api/v1/dashboard",
    "/internal/data/query", "/internal/batch/process", "/api/v1/search",
    "/internal/health", "/api/v1/settings", "/internal/metrics",
]
COMMIT_PREFIXES = [
    "feat:", "fix:", "perf:", "refactor:", "chore:", "hotfix:", "security:",
]
COMMIT_MESSAGES = [
    "migrate to new {component} library",
    "optimize {component} for high throughput",
    "add caching layer for {component}",
    "refactor {component} error handling",
    "update {component} connection parameters",
    "introduce {component} rate limiting",
    "add monitoring for {component}",
    "fix edge case in {component}",
    "upgrade {component} dependency version",
    "simplify {component} retry logic",
]
COMPONENTS = [
    "connection pool", "session manager", "query engine", "batch processor",
    "auth validator", "cache layer", "rate limiter", "load balancer",
    "message queue", "health checker", "circuit breaker", "DNS resolver",
]
INFRA_EVENT_TYPES = [
    "network_maintenance", "az_failover", "dns_update", "certificate_rotation",
    "scaling_event", "storage_migration", "kernel_update", "load_balancer_drain",
]


def _hash_seed(seed: int, salt: str = "") -> str:
    """Deterministic hash for generating IDs."""
    h = hashlib.md5(f"{seed}:{salt}".encode()).hexdigest()
    return h[:8]


def _random_timestamp(rng: random.Random, base: datetime, hours_range: Tuple[float, float]) -> str:
    """Generate a timestamp within a range relative to base."""
    offset_hours = rng.uniform(*hours_range)
    ts = base + timedelta(hours=offset_hours)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_log_entries(
    rng: random.Random,
    service: str,
    base_time: datetime,
    is_target: bool,
    error_type: str,
    n_entries: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate realistic log entries for a service."""
    entries = []
    levels_normal = ["INFO", "INFO", "INFO", "DEBUG", "WARN"]
    levels_error = ["ERROR", "ERROR", "CRITICAL", "WARN", "ERROR"]

    for i in range(n_entries):
        log_id = f"log-{service[:3]}-{seed}-{i:03d}"
        is_during_incident = i >= n_entries * 0.6  # last 40% are during incident

        if is_during_incident and is_target:
            level = rng.choice(levels_error)
            relevant = rng.random() < 0.6
            messages = [
                f"{error_type}: Cannot acquire resource — pool exhausted ({rng.randint(0,50)}/{rng.randint(10,20)} active)",
                f"Request failed after {rng.randint(3,5)} retries: {error_type}",
                f"Health check FAILED — {error_type} on primary endpoint",
                f"Thread pool saturation: {rng.randint(40,50)}/50 threads blocked",
                f"Circuit breaker OPEN after {rng.randint(5,15)} consecutive failures",
                f"Response latency p99 = {rng.randint(5000,15000)}ms (threshold: 2000ms)",
                f"Cascading failure detected from downstream dependency",
                f"Memory usage at {rng.randint(85,99)}% — GC pressure critical",
            ]
            message = rng.choice(messages)
        elif is_during_incident and not is_target:
            level = rng.choice(["WARN", "ERROR", "INFO"])
            relevant = rng.random() < 0.3
            messages = [
                f"Upstream {service} returning {rng.choice(['503', '504', '500'])} errors",
                f"Elevated latency on {service}-dependent routes: {rng.randint(2000,8000)}ms",
                f"Fallback cache serving stale data",
                f"Error rate crossed {rng.randint(5,20)}% threshold",
                f"Retry budget exhausted for {service} dependency",
            ]
            message = rng.choice(messages)
        else:
            level = rng.choice(levels_normal)
            relevant = False
            messages = [
                f"Health check passed, all endpoints responding",
                f"Deployment v{rng.randint(1,5)}.{rng.randint(0,20)}.{rng.randint(0,9)} rolled out successfully",
                f"Cache warmer completed, {rng.randint(500,5000)} keys refreshed",
                f"Connection pool utilization: {rng.randint(10,60)}%",
                f"Request processing nominal, latency p99={rng.randint(20,200)}ms",
                f"Scheduled maintenance task completed",
                f"Log rotation completed, {rng.randint(5,20)} files archived",
            ]
            message = rng.choice(messages)

        offset_hours = -24 + (i / n_entries) * 26  # spread across 24h before + 2h during
        ts = base_time + timedelta(hours=offset_hours)

        entries.append({
            "id": log_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": level,
            "message": message,
            "relevant": relevant,
        })

    return entries


def _generate_traces(
    rng: random.Random,
    services: List[str],
    base_time: datetime,
    target_service: str,
    error_type: str,
    n_traces: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate distributed traces."""
    traces = []
    for i in range(n_traces):
        trace_id = f"trace-{seed}-{i:03d}"
        is_error_trace = i >= n_traces * 0.5

        offset_hours = -2 + (i / n_traces) * 4
        ts = base_time + timedelta(hours=offset_hours)

        spans = []
        n_spans = rng.randint(1, min(4, len(services)))
        used_services = rng.sample(services, n_spans)

        for svc in used_services:
            if is_error_trace and svc == target_service:
                spans.append({
                    "service": svc,
                    "operation": rng.choice(OPERATIONS),
                    "duration_ms": rng.randint(5000, 15000),
                    "status": "ERROR",
                    "error": f"{error_type}: resource unavailable",
                })
            elif is_error_trace:
                spans.append({
                    "service": svc,
                    "operation": rng.choice(OPERATIONS),
                    "duration_ms": rng.randint(3000, 8000),
                    "status": rng.choice(["ERROR", "SLOW"]),
                    "error": f"upstream {target_service} timeout",
                })
            else:
                spans.append({
                    "service": svc,
                    "operation": rng.choice(OPERATIONS),
                    "duration_ms": rng.randint(5, 200),
                    "status": "OK",
                })

        traces.append({
            "trace_id": trace_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spans": spans,
            "relevant": is_error_trace and any(s["service"] == target_service for s in spans),
        })

    return traces


def _generate_commits(
    rng: random.Random,
    services: List[str],
    base_time: datetime,
    target_service: str,
    root_cause_component: str,
    n_commits: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate commits, including the root cause commit. Returns (commits, culprit_hash)."""
    commits = []
    culprit_hash = ""

    # Position the culprit commit near the incident
    culprit_index = rng.randint(n_commits // 2, n_commits - 2)

    for i in range(n_commits):
        commit_hash = f"commit-{_hash_seed(seed, f'c{i}')}"
        service = rng.choice(services) if i != culprit_index else target_service
        offset_hours = -24 + (i / n_commits) * 25
        ts = base_time + timedelta(hours=offset_hours)
        author = f"{rng.choice(DEVELOPER_NAMES)}@company.com"
        component = root_cause_component if i == culprit_index else rng.choice(COMPONENTS)
        prefix = rng.choice(COMMIT_PREFIXES)
        msg_template = rng.choice(COMMIT_MESSAGES)
        message = f"{prefix} {msg_template.format(component=component)}"

        is_culprit = i == culprit_index
        if is_culprit:
            culprit_hash = commit_hash
            diff = (
                f"--- a/{service}-service/{component.replace(' ', '_')}.go\n"
                f"+++ b/{service}-service/{component.replace(' ', '_')}.go\n"
                f"-// Old: stable configuration\n"
                f"-config.MaxSize = 100\n"
                f"-config.Timeout = 30 * time.Second\n"
                f"+// New: reduced limits for 'optimization'\n"
                f"+config.MaxSize = {rng.randint(5, 20)}\n"
                f"+config.Timeout = {rng.randint(1, 5)} * time.Second\n"
                f"+// WARNING: This may cause issues under high load"
            )
        else:
            diff = (
                f"--- a/{service}-service/{component.replace(' ', '_')}.go\n"
                f"+++ b/{service}-service/{component.replace(' ', '_')}.go\n"
                f"+// Minor change in {component}"
            )

        commits.append({
            "hash": commit_hash,
            "service": service,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author": author,
            "message": message,
            "diff": diff,
            "relevant": is_culprit,
        })

    return commits, culprit_hash


def _generate_config_changes(
    rng: random.Random,
    services: List[str],
    base_time: datetime,
    target_service: str,
    n_configs: int,
    seed: int,
    root_cause_type: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate config changes, optionally including a root-cause config."""
    configs = []
    culprit_config_id = ""
    culprit_index = rng.randint(n_configs // 2, n_configs - 2) if root_cause_type == "config" else -1

    for i in range(n_configs):
        config_id = f"cfg-{seed}-{i:03d}"
        service = target_service if i == culprit_index else rng.choice(services)
        offset_hours = -24 + (i / n_configs) * 25
        ts = base_time + timedelta(hours=offset_hours)
        key = rng.choice(CONFIG_KEYS)

        is_culprit = i == culprit_index
        if is_culprit:
            culprit_config_id = config_id
            old_val = str(rng.randint(50, 200))
            new_val = str(rng.randint(5, 20))
            desc = f"Reduced {key} — may cause resource starvation under load"
        else:
            old_val = str(rng.randint(10, 100))
            new_val = str(rng.randint(10, 100))
            desc = f"Adjusted {key} for routine maintenance"

        configs.append({
            "config_id": config_id,
            "service": service,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "key": key,
            "old_value": old_val,
            "new_value": new_val,
            "description": desc,
            "relevant": is_culprit,
        })

    return configs, culprit_config_id


def _generate_infra_events(
    rng: random.Random,
    base_time: datetime,
    n_events: int,
    seed: int,
    root_cause_type: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate infra events."""
    events = []
    culprit_event_id = ""
    culprit_index = 0 if root_cause_type == "correlated" else -1

    for i in range(n_events):
        event_id = f"infra-{seed}-{i:03d}"
        offset_hours = -12 + (i / n_events) * 14
        ts = base_time + timedelta(hours=offset_hours)
        event_type = rng.choice(INFRA_EVENT_TYPES)

        is_culprit = i == culprit_index
        if is_culprit:
            culprit_event_id = event_id
            desc = f"Network degradation detected on primary subnet. Packet loss 15%, latency spike 450ms. Triggered automatic failover procedures."
        else:
            desc = f"Routine {event_type.replace('_', ' ')} completed. No impact reported."

        events.append({
            "event_id": event_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": event_type,
            "description": desc,
            "relevant": is_culprit,
        })

    return events, culprit_event_id


def generate_scenario(seed: int, difficulty: str = "easy") -> Dict[str, Any]:
    """
    Generate a complete scenario from a seed.

    Args:
        seed: Integer seed for deterministic generation.
        difficulty: 'easy', 'medium', or 'hard'.

    Returns:
        Complete scenario dict ready for use by PostmortemEnvironment.
    """
    rng = random.Random(seed)

    # Select failure template based on difficulty
    if difficulty == "easy":
        template_name = rng.choice(["connection_pool", "memory_leak", "config_drift"])
        max_steps = 40
        n_logs_per_service = rng.randint(10, 15)
        n_traces = rng.randint(12, 18)
        n_commits = rng.randint(8, 12)
        n_configs = rng.randint(4, 7)
        n_infra = rng.randint(1, 3)
    elif difficulty == "medium":
        template_name = rng.choice(["oom_cascade", "connection_pool", "memory_leak"])
        max_steps = 75
        n_logs_per_service = rng.randint(15, 25)
        n_traces = rng.randint(20, 35)
        n_commits = rng.randint(10, 15)
        n_configs = rng.randint(8, 14)
        n_infra = rng.randint(2, 4)
    else:  # hard
        template_name = rng.choice(["failover_bug", "oom_cascade", "config_drift"])
        max_steps = 120
        n_logs_per_service = rng.randint(25, 40)
        n_traces = rng.randint(30, 50)
        n_commits = rng.randint(12, 18)
        n_configs = rng.randint(12, 22)
        n_infra = rng.randint(3, 6)

    template = FAILURE_TEMPLATES[template_name]

    # Pick target service and build dependency order
    services = list(SERVICE_NAMES)
    target_idx = rng.randint(0, 3)
    target_service = services[target_idx]

    # Build graph (always same structure: frontend → auth → data → batch)
    service_graph = {
        "frontend": ["auth"],
        "auth": ["data"],
        "data": ["batch"],
        "batch": [],
    }

    # Find upstream services
    upstreams = []
    chain_order = ["batch", "data", "auth", "frontend"]
    target_pos = chain_order.index(target_service)
    for s in chain_order[target_pos + 1:]:
        upstreams.append(s)

    # Incident time
    base_date = datetime(2026, 4, rng.randint(10, 20), rng.randint(0, 23), rng.randint(0, 59), tzinfo=timezone.utc)
    incident_duration = timedelta(minutes=rng.randint(8, 25))
    incident_window = {
        "start": base_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (base_date + incident_duration).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    error_type = rng.choice(ERROR_TYPES)
    root_cause_component = rng.choice(COMPONENTS)

    # Generate all telemetry
    logs = {}
    for svc in services:
        logs[svc] = _generate_log_entries(
            rng, svc, base_date,
            is_target=(svc == target_service),
            error_type=error_type,
            n_entries=n_logs_per_service,
            seed=seed,
        )

    traces = _generate_traces(rng, services, base_date, target_service, error_type, n_traces, seed)
    commits, culprit_commit = _generate_commits(rng, services, base_date, target_service, root_cause_component, n_commits, seed)
    config_changes, culprit_config = _generate_config_changes(rng, services, base_date, target_service, n_configs, seed, template["root_cause_type"])
    infra_events, culprit_infra = _generate_infra_events(rng, base_date, n_infra, seed, template["root_cause_type"])

    # Build ground truth
    cause_type = template["root_cause_type"]
    if cause_type == "commit":
        cause = culprit_commit
    elif cause_type == "config":
        cause = culprit_config
    elif cause_type == "correlated":
        cause = f"{culprit_commit}+{culprit_infra}"
    else:
        cause = culprit_commit

    # Build chain from template
    chain = []
    for step in template["chain_template"]:
        svc = step["service"]
        svc = svc.replace("{target_service}", target_service)
        if "{upstream_1}" in svc and len(upstreams) > 0:
            svc = upstreams[0]
        elif "{upstream_2}" in svc and len(upstreams) > 1:
            svc = upstreams[1]
        elif "{upstream_3}" in svc and len(upstreams) > 2:
            svc = upstreams[2]
        elif "{upstream_" in svc:
            svc = upstreams[-1] if upstreams else target_service

        effect = step["effect"]
        chain.append({"service": svc, "effect": effect})

    # Build task ID
    task_id = f"seed_{seed}_{difficulty}"

    # Gather relevant fact IDs
    relevant_facts = []
    for svc, svc_logs in logs.items():
        for log in svc_logs:
            if log.get("relevant"):
                relevant_facts.append(log["id"])
    for trace in traces:
        if trace.get("relevant"):
            relevant_facts.append(trace["trace_id"])
    for commit in commits:
        if commit.get("relevant"):
            relevant_facts.append(commit["hash"])
    for cfg in config_changes:
        if cfg.get("relevant"):
            relevant_facts.append(cfg["config_id"])
    for evt in infra_events:
        if evt.get("relevant"):
            relevant_facts.append(evt["event_id"])

    # Description
    description = template["description_template"].format(
        target_service=target_service,
        error_type=error_type,
        incident_time=base_date.strftime("%H:%M"),
        hours_before=rng.randint(2, 8),
        leak_component=root_cause_component,
    )

    # Service metadata
    service_data = []
    for svc in services:
        error_rate = rng.uniform(80, 99) if svc == target_service else rng.uniform(0, 30)
        status = "down" if svc == target_service else rng.choice(["degraded", "healthy"])
        service_data.append({
            "name": svc,
            "status": status,
            "dependencies": service_graph[svc],
            "recent_deploy_count": sum(1 for c in commits if c["service"] == svc),
            "error_rate_during_incident": round(error_rate, 1),
        })

    # Build ground truth dict
    ground_truth: Dict[str, Any] = {
        "cause": cause,
        "cause_type": cause_type,
        "chain": chain,
    }
    if cause_type == "correlated":
        ground_truth["contributing_causes"] = [culprit_commit, culprit_infra]

    return {
        "task_id": task_id,
        "task_difficulty": difficulty,
        "task_description": description,
        "max_steps": max_steps,
        "service_graph": service_graph,
        "services": service_data,
        "incident_window": incident_window,
        "logs": logs,
        "traces": traces,
        "commits": commits,
        "config_changes": config_changes,
        "infra_events": infra_events,
        "relevant_fact_ids": relevant_facts,
        "ground_truth": ground_truth,
    }


def generate_oracle_solution(scenario: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Generate the oracle solution for a procedurally generated scenario.

    Produces an optimal action sequence that investigates efficiently
    and submits the correct answer.
    """
    rng = random.Random(seed + 9999)
    gt = scenario["ground_truth"]
    target_service = gt["chain"][0]["service"] if gt["chain"] else "data"

    actions = []

    # Step 1: Query logs of the target service for errors
    actions.append({
        "action_type": "query_logs",
        "service": target_service,
        "keyword": "error",
        "reason": f"Investigate {target_service} — highest error rate",
    })

    # Step 2: Query logs for deployment info
    actions.append({
        "action_type": "query_logs",
        "service": target_service,
        "keyword": "deploy",
        "reason": "Check recent deployments",
    })

    # Step 3: Diff the culprit commit
    cause = gt["cause"]
    commit_hash = cause.split("+")[0] if "+" in cause else cause
    if commit_hash.startswith("commit-"):
        actions.append({
            "action_type": "diff_commit",
            "commit_hash": commit_hash,
            "reason": "Most suspicious commit near incident",
        })

    # Step 4: Fetch a trace during the incident
    traces = scenario.get("traces", [])
    relevant_traces = [t for t in traces if t.get("relevant")]
    if relevant_traces:
        actions.append({
            "action_type": "fetch_trace",
            "trace_id": relevant_traces[0]["trace_id"],
            "reason": "Get cascade trace during incident",
        })

    # Step 5: Hypothesize
    actions.append({
        "action_type": "hypothesize",
        "cause_entity_id": cause,
        "reason": "Test root cause hypothesis",
    })

    # Step 6: Explain chain
    actions.append({
        "action_type": "explain_chain",
        "chain": gt["chain"],
        "reason": "Verify causal chain",
    })

    # Step 7: Submit
    actions.append({
        "action_type": "submit",
        "final_cause": cause,
        "final_chain": gt["chain"],
        "reason": "Confident in root cause and chain",
    })

    return {
        "task_id": scenario["task_id"],
        "ground_truth": gt,
        "optimal_action_sequence": actions,
    }


# ------------------------------------------------------------------
# Batch generation
# ------------------------------------------------------------------

def generate_batch(
    n_per_difficulty: int = 10,
    output_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate a batch of scenarios.

    Args:
        n_per_difficulty: How many seeds per difficulty level.
        output_dir: If provided, saves JSONs to this directory.

    Returns:
        Dict mapping task_id → scenario.
    """
    scenarios = {}
    solutions = {}

    for difficulty in ["easy", "medium", "hard"]:
        for i in range(n_per_difficulty):
            seed = hash((difficulty, i)) % (2**31)
            scenario = generate_scenario(seed, difficulty)
            solution = generate_oracle_solution(scenario, seed)
            task_id = scenario["task_id"]
            scenarios[task_id] = scenario
            solutions[task_id] = solution

    if output_dir:
        out = Path(output_dir)
        (out / "scenarios").mkdir(parents=True, exist_ok=True)
        (out / "solutions").mkdir(parents=True, exist_ok=True)
        for task_id, scenario in scenarios.items():
            with open(out / "scenarios" / f"{task_id}.json", "w") as f:
                json.dump(scenario, f, indent=2)
        for task_id, solution in solutions.items():
            with open(out / "solutions" / f"{task_id}_solution.json", "w") as f:
                json.dump(solution, f, indent=2)

    return scenarios


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PostmortemEnv scenarios")
    parser.add_argument("--n", type=int, default=10, help="Seeds per difficulty")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Preview a single scenario")
    args = parser.parse_args()

    if args.preview:
        scenario = generate_scenario(42, "medium")
        print(json.dumps(scenario, indent=2)[:3000])
        print(f"\n... ({len(json.dumps(scenario))} bytes total)")
        print(f"\nTask: {scenario['task_id']}")
        print(f"Root cause: {scenario['ground_truth']['cause']}")
        print(f"Chain: {scenario['ground_truth']['chain']}")
        print(f"Relevant facts: {len(scenario['relevant_fact_ids'])}")
    else:
        generate_batch(args.n, args.output or "data/generated")
        print(f"Generated {args.n * 3} scenarios to {args.output or 'data/generated'}/")
