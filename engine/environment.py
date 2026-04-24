"""Core PostmortemEnv engine. Implements reset(), step(), and state()."""

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple

from models import (
    Service, Observation, Action, ActionType, Reward, EnvironmentState,
)
from data.generator import load_scenario, load_solution
from engine.grader import Grader
from engine.reward_calculator import RewardCalculator

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata


_TIME_WINDOW_RE = re.compile(r"^last[_\- ]?(\d+)\s*([smhd])$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; tolerate trailing ``Z``. Returns ``None`` on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _resolve_time_window(
    spec: Optional[str],
    incident_window: Dict[str, str],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Resolve a ``time_window`` spec to a ``(start, end)`` datetime pair.

    Supported specs (case-insensitive):

    * ``"last_5m"``, ``"last_1h"``, ``"last_30s"``, ``"last_2d"`` —
      ``N`` units ending at ``incident_window["end"]``.
    * ``"during"`` / ``"incident"`` / ``"during_incident"`` — exact
      ``[incident_start, incident_end]`` range.

    Anything else (or a missing incident window) returns ``(None, None)``,
    which means *no filtering* — the call still succeeds.
    """
    if not spec:
        return None, None
    s = spec.strip().lower()
    incident_end = _parse_iso(incident_window.get("end", ""))
    incident_start = _parse_iso(incident_window.get("start", ""))

    if s in ("during", "incident", "during_incident") and incident_start and incident_end:
        return incident_start, incident_end

    m = _TIME_WINDOW_RE.match(s)
    if m and incident_end:
        n = int(m.group(1))
        unit = m.group(2).lower()
        delta = timedelta(seconds=n * _UNIT_SECONDS[unit])
        return incident_end - delta, incident_end

    return None, None


def _ts_in_window(
    ts: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> bool:
    """Whether ``ts`` falls inside ``[start, end]``. Open bounds are inclusive."""
    if start is None and end is None:
        return True
    parsed = _parse_iso(ts)
    if parsed is None:
        # Don't drop logs we can't parse — keep them visible to the agent.
        return True
    if start is not None and parsed < start:
        return False
    if end is not None and parsed > end:
        return False
    return True


class PostmortemEnvironment(Environment[Action, Observation, EnvironmentState]):
    """
    The PostmortemEnv environment.

    An epistemic RL environment where an LLM agent investigates a cloud
    outage that has already happened. The agent receives a frozen telemetry
    snapshot and must identify the root cause and causal chain within a
    limited query budget.

    Lifecycle:
        1. reset(task_id) → Observation  (start a new investigation)
        2. step(action) → Observation    (query, hypothesize, or submit)
        3. state → EnvironmentState      (inspect oracle state)
    """

    def __init__(self):
        self._task_id: str = ""
        self._difficulty: str = ""
        self._task_description: str = ""
        self._max_steps: int = 40

        # Scenario data (the frozen telemetry haystack)
        self._service_graph: Dict[str, List[str]] = {}
        self._services: List[dict] = []
        self._incident_window: Dict[str, str] = {}
        self._logs: Dict[str, List[dict]] = {}
        self._traces: List[dict] = []
        self._commits: List[dict] = []
        self._config_changes: List[dict] = []
        self._infra_events: List[dict] = []

        # Oracle ground truth (hidden from agent)
        self._ground_truth_cause: str = ""
        self._ground_truth_cause_type: str = ""
        self._ground_truth_chain: List[dict] = []
        self._contributing_causes: List[str] = []
        self._relevant_fact_ids: Set[str] = set()

        # Episode tracking
        self._facts_discovered: Set[str] = set()
        self._known_facts: List[str] = []
        self._hypotheses_submitted: List[dict] = []
        self._wrong_hypotheses: int = 0
        self._steps_taken: int = 0
        self._done: bool = False
        self._message: str = ""
        self._last_query_result: str = ""
        self._final_score: Optional[float] = None

        self._initialized: bool = False

    # --- metadata ---

    def get_metadata(self) -> EnvironmentMetadata:
        """Return rich metadata for the /metadata endpoint."""
        return EnvironmentMetadata(
            name="PostmortemEnv",
            description=(
                "An epistemic RL environment where LLM agents investigate cloud "
                "outages that have already happened. The agent receives frozen "
                "telemetry and must identify the root cause and causal chain."
            ),
            version="1.0.0",
            author="Three Musketeers (Utkarsh, Mohit, Tanush)",
        )

    # --- reset ---

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: str = "task1_recent_deploy", **kwargs) -> Observation:
        """
        Start a new investigation episode for the given task.

        Loads the frozen outage scenario from disk, initializes oracle state,
        and returns the initial observation with the service graph and task brief.
        """
        scenario = load_scenario(task_id)
        return self.reset_from_scenario(scenario, task_id=task_id)

    def reset_from_scenario(
        self,
        scenario: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Observation:
        """
        Reset the environment with an externally-supplied scenario dict.

        Same shape contract as scenarios produced by
        :func:`data.seed_generator.generate_scenario` and
        :func:`data.generator.load_scenario`. Use this for procedural seeds,
        GRPO rollouts, and curriculum tooling that need to inject a scenario
        without going through the disk-based ``load_scenario()`` lookup.

        ``task_id`` defaults to ``scenario["task_id"]`` when not supplied.
        """
        self._task_id = task_id or scenario["task_id"]
        self._difficulty = scenario["task_difficulty"]
        self._task_description = scenario.get("task_description", "Investigate the outage.")
        self._max_steps = scenario.get("max_steps", 40)

        # Load the telemetry haystack
        self._service_graph = scenario.get("service_graph", {})
        self._services = scenario.get("services", [])
        self._incident_window = scenario.get("incident_window", {})
        self._logs = scenario.get("logs", {})
        self._traces = scenario.get("traces", [])
        self._commits = scenario.get("commits", [])
        self._config_changes = scenario.get("config_changes", [])
        self._infra_events = scenario.get("infra_events", [])

        # Oracle ground truth
        gt = scenario["ground_truth"]
        self._ground_truth_cause = gt["cause"]
        self._ground_truth_cause_type = gt.get("cause_type", "commit")
        self._ground_truth_chain = gt["chain"]
        self._contributing_causes = gt.get("contributing_causes", [])
        self._relevant_fact_ids = set(scenario.get("relevant_fact_ids", []))

        # Reset episode tracking
        self._facts_discovered = set()
        self._known_facts = []
        self._hypotheses_submitted = []
        self._wrong_hypotheses = 0
        self._steps_taken = 0
        self._done = False
        self._message = f"Investigation started. {self._task_description}"
        self._last_query_result = ""
        self._final_score = None

        self._initialized = True
        return self._build_observation(reward=0.01, done=False)

    # --- step ---

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        """Process an agent action and return the updated observation."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._steps_taken += 1

        # Dispatch to handler
        handler = {
            ActionType.QUERY_LOGS: self._handle_query_logs,
            ActionType.FETCH_TRACE: self._handle_fetch_trace,
            ActionType.DIFF_COMMIT: self._handle_diff_commit,
            ActionType.INSPECT_CONFIG: self._handle_inspect_config,
            ActionType.INSPECT_INFRA: self._handle_inspect_infra,
            ActionType.HYPOTHESIZE: self._handle_hypothesize,
            ActionType.EXPLAIN_CHAIN: self._handle_explain_chain,
            ActionType.SUBMIT: self._handle_submit,
        }.get(action.action_type)

        if handler is None:
            reward = RewardCalculator.invalid_action_reward(
                f"Unknown action type: {action.action_type}"
            )
        else:
            reward = handler(action)

        self._message = reward.message

        # Check if max steps reached
        if self._steps_taken >= self._max_steps and not self._done:
            self._done = True
            # Auto-submit with empty answer on budget exhaustion
            if self._final_score is None:
                self._final_score = Grader.compute_final_score(
                    submitted_cause="",
                    submitted_chain=None,
                    ground_truth_cause=self._ground_truth_cause,
                    ground_truth_chain=self._ground_truth_chain,
                    cause_type=self._ground_truth_cause_type,
                    steps_used=self._steps_taken,
                    max_steps=self._max_steps,
                    n_wrong_hypotheses=self._wrong_hypotheses,
                    contributing_causes=self._contributing_causes,
                )
            self._message += " | Query budget exhausted. Episode ended."

        # Compute reward for observation
        if self._done and self._final_score is not None:
            raw_reward = self._final_score
        else:
            raw_reward = reward.value

        # Clamp to (0.01, 0.99)
        clamped = round(min(max(float(raw_reward), 0.01), 0.99), 4)

        obs = self._build_observation(reward=clamped, done=self._done)
        obs.metadata = {
            "steps_taken": self._steps_taken,
            "facts_discovered": len(self._facts_discovered),
            "wrong_hypotheses": self._wrong_hypotheses,
        }
        return obs

    # --- state ---

    @property
    def state(self) -> EnvironmentState:
        """Return full internal state (god-mode, for grading/debugging)."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return EnvironmentState(
            task_id=self._task_id,
            task_difficulty=self._difficulty,
            ground_truth_cause=self._ground_truth_cause,
            ground_truth_cause_type=self._ground_truth_cause_type,
            ground_truth_chain=self._ground_truth_chain,
            relevant_fact_ids=list(self._relevant_fact_ids),
            facts_discovered=list(self._facts_discovered),
            total_relevant_facts=len(self._relevant_fact_ids),
            hypotheses_submitted=list(self._hypotheses_submitted),
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            done=self._done,
            final_score=self._final_score,
        )

    # --- action handlers ---

    def _handle_query_logs(self, action: Action) -> Reward:
        """Search logs by service and keyword. Returns matching log lines.

        Optional ``action.time_window`` (e.g. ``"last_5m"``, ``"last_1h"``,
        ``"during_incident"``) restricts results to a window anchored on the
        scenario's ``incident_window``. Unparseable specs degrade to no
        filtering rather than raising — keyword-only queries continue to work.
        """
        service = action.service
        keyword = action.keyword or ""

        if not service:
            return RewardCalculator.invalid_action_reward(
                "query_logs requires a 'service' parameter."
            )

        if service not in self._logs:
            available = ", ".join(self._logs.keys())
            return RewardCalculator.invalid_action_reward(
                f"Unknown service '{service}'. Available: {available}"
            )

        service_logs = self._logs[service]

        window_start, window_end = _resolve_time_window(
            action.time_window, self._incident_window
        )

        kw_lower = keyword.lower()

        def _matches(log: dict) -> bool:
            if kw_lower and not (
                kw_lower in log.get("message", "").lower()
                or kw_lower in log.get("level", "").lower()
            ):
                return False
            return _ts_in_window(log.get("timestamp", ""), window_start, window_end)

        matching = [log for log in service_logs if _matches(log)]

        # Cap at 20 results
        matching = matching[:20]

        # Check for relevant facts discovered
        new_relevant = 0
        for log in matching:
            log_id = log.get("id", "")
            if log_id in self._relevant_fact_ids and log_id not in self._facts_discovered:
                self._facts_discovered.add(log_id)
                new_relevant += 1
                fact_desc = f"[{service}] {log.get('level', 'INFO')}: {log.get('message', '')[:80]}"
                self._known_facts.append(fact_desc)

        # Format result
        window_note = (
            f" window: '{action.time_window}'" if action.time_window else ""
        )
        if matching:
            lines = []
            for log in matching:
                lines.append(
                    f"[{log.get('timestamp', '')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                )
            self._last_query_result = (
                f"=== Logs for {service} (keyword: '{keyword}'{window_note}) ===\n"
                + "\n".join(lines)
            )
        else:
            self._last_query_result = (
                f"No log entries found for service '{service}' "
                f"matching keyword '{keyword}'{window_note}."
            )

        return RewardCalculator.query_reward(new_relevant)

    def _handle_fetch_trace(self, action: Action) -> Reward:
        """Fetch a specific distributed trace by ID."""
        trace_id = action.trace_id
        if not trace_id:
            return RewardCalculator.invalid_action_reward(
                "fetch_trace requires a 'trace_id' parameter."
            )

        # Find the trace
        trace = None
        for t in self._traces:
            if t.get("trace_id") == trace_id:
                trace = t
                break

        if trace is None:
            available = [t.get("trace_id", "") for t in self._traces[:10]]
            return RewardCalculator.invalid_action_reward(
                f"Trace '{trace_id}' not found. Available trace IDs: {available}"
            )

        # Check for relevant facts
        new_relevant = 0
        if trace.get("relevant", False) and trace_id not in self._facts_discovered:
            self._facts_discovered.add(trace_id)
            new_relevant += 1
            # Summarize the trace as a fact
            spans = trace.get("spans", [])
            if spans:
                services_involved = [s.get("service", "") for s in spans]
                fact_desc = f"Trace {trace_id}: {' → '.join(services_involved)}"
                if any(s.get("status") == "ERROR" for s in spans):
                    errors = [s.get("error", "") for s in spans if s.get("error")]
                    fact_desc += f" [ERRORS: {'; '.join(errors[:3])}]"
                self._known_facts.append(fact_desc)

        # Format trace
        lines = [f"=== Trace: {trace_id} (timestamp: {trace.get('timestamp', '')}) ==="]
        for span in trace.get("spans", []):
            status = span.get("status", "OK")
            duration = span.get("duration_ms", 0)
            line = f"  {span.get('service', '?')} | {span.get('operation', '?')} | {duration}ms | {status}"
            if span.get("error"):
                line += f" | ERROR: {span['error']}"
            lines.append(line)
        self._last_query_result = "\n".join(lines)

        return RewardCalculator.query_reward(new_relevant)

    def _handle_diff_commit(self, action: Action) -> Reward:
        """Show the diff for a specific commit."""
        commit_hash = action.commit_hash
        if not commit_hash:
            return RewardCalculator.invalid_action_reward(
                "diff_commit requires a 'commit_hash' parameter."
            )

        # Find the commit
        commit = None
        for c in self._commits:
            if c.get("hash") == commit_hash:
                commit = c
                break

        if commit is None:
            available = [c.get("hash", "") for c in self._commits[:10]]
            return RewardCalculator.invalid_action_reward(
                f"Commit '{commit_hash}' not found. Available: {available}"
            )

        # Check for relevant facts
        new_relevant = 0
        if commit.get("relevant", False) and commit_hash not in self._facts_discovered:
            self._facts_discovered.add(commit_hash)
            new_relevant += 1
            fact_desc = f"Commit {commit_hash} ({commit.get('service', '?')}): {commit.get('message', '')[:80]}"
            self._known_facts.append(fact_desc)

        # Format commit
        lines = [
            f"=== Commit: {commit_hash} ===",
            f"Service: {commit.get('service', '?')}",
            f"Author: {commit.get('author', '?')}",
            f"Timestamp: {commit.get('timestamp', '?')}",
            f"Message: {commit.get('message', '')}",
            f"",
            f"--- Diff ---",
            commit.get("diff", "(no diff available)"),
        ]
        self._last_query_result = "\n".join(lines)

        return RewardCalculator.query_reward(new_relevant)

    def _handle_inspect_config(self, action: Action) -> Reward:
        """Inspect a specific config change."""
        config_id = action.config_id
        if not config_id:
            return RewardCalculator.invalid_action_reward(
                "inspect_config requires a 'config_id' parameter."
            )

        # Find the config change
        config = None
        for c in self._config_changes:
            if c.get("config_id") == config_id:
                config = c
                break

        if config is None:
            available = [c.get("config_id", "") for c in self._config_changes[:10]]
            return RewardCalculator.invalid_action_reward(
                f"Config '{config_id}' not found. Available: {available}"
            )

        # Check for relevant facts
        new_relevant = 0
        if config.get("relevant", False) and config_id not in self._facts_discovered:
            self._facts_discovered.add(config_id)
            new_relevant += 1
            fact_desc = f"Config {config_id} ({config.get('service', '?')}): {config.get('key', '')} changed from '{config.get('old_value', '')}' to '{config.get('new_value', '')}'"
            self._known_facts.append(fact_desc)

        # Format config
        lines = [
            f"=== Config Change: {config_id} ===",
            f"Service: {config.get('service', '?')}",
            f"Timestamp: {config.get('timestamp', '?')}",
            f"Key: {config.get('key', '?')}",
            f"Old Value: {config.get('old_value', '?')}",
            f"New Value: {config.get('new_value', '?')}",
            f"Description: {config.get('description', '')}",
        ]
        self._last_query_result = "\n".join(lines)

        return RewardCalculator.query_reward(new_relevant)

    def _handle_inspect_infra(self, action: Action) -> Reward:
        """Inspect a specific infrastructure event by ID.

        Infrastructure events (DNS updates, AZ failovers, certificate
        rotations, etc.) appear in the observation but were previously
        unreachable — there was no action to drill into them. With this
        handler, an event whose underlying detail is a relevant fact
        contributes to the info-gain reward exactly like log/trace/commit
        discovery.
        """
        event_id = action.event_id
        if not event_id:
            return RewardCalculator.invalid_action_reward(
                "inspect_infra requires an 'event_id' parameter."
            )

        event = next(
            (e for e in self._infra_events if e.get("event_id") == event_id),
            None,
        )
        if event is None:
            available = [e.get("event_id", "") for e in self._infra_events[:10]]
            return RewardCalculator.invalid_action_reward(
                f"Infra event '{event_id}' not found. Available: {available}"
            )

        new_relevant = 0
        if event.get("relevant", False) and event_id not in self._facts_discovered:
            self._facts_discovered.add(event_id)
            new_relevant += 1
            fact_desc = (
                f"Infra event {event_id} ({event.get('type', '?')}): "
                f"{event.get('description', '')[:120]}"
            )
            self._known_facts.append(fact_desc)

        lines = [
            f"=== Infrastructure Event: {event_id} ===",
            f"Type: {event.get('type', '?')}",
            f"Timestamp: {event.get('timestamp', '?')}",
            f"Description: {event.get('description', '')}",
        ]
        self._last_query_result = "\n".join(lines)

        return RewardCalculator.query_reward(new_relevant)

    def _handle_hypothesize(self, action: Action) -> Reward:
        """Submit a candidate root cause for feedback (non-terminal)."""
        cause_id = action.cause_entity_id
        if not cause_id:
            return RewardCalculator.invalid_action_reward(
                "hypothesize requires a 'cause_entity_id' parameter."
            )

        is_correct = Grader.check_cause_match(
            cause_id,
            self._ground_truth_cause,
            self._ground_truth_cause_type,
            self._contributing_causes,
        )

        self._hypotheses_submitted.append({
            "cause_entity_id": cause_id,
            "result": "correct" if is_correct else "incorrect",
        })

        if is_correct:
            self._last_query_result = f"CORRECT: '{cause_id}' matches the ground truth root cause."
            return RewardCalculator.hypothesis_correct_reward()
        else:
            self._wrong_hypotheses += 1
            self._last_query_result = f"INCORRECT: '{cause_id}' is NOT the root cause. Keep investigating."
            return RewardCalculator.hypothesis_wrong_reward()

    def _handle_explain_chain(self, action: Action) -> Reward:
        """Submit a causal chain hypothesis for feedback (non-terminal)."""
        chain = action.chain
        if not chain:
            return RewardCalculator.invalid_action_reward(
                "explain_chain requires a 'chain' parameter (list of {{service, effect}})."
            )

        similarity = Grader.compute_chain_similarity(chain, self._ground_truth_chain)

        self._last_query_result = (
            f"Chain similarity: {similarity:.2f}\n"
            f"Your chain has {len(chain)} steps. "
            f"Ground truth has {len(self._ground_truth_chain)} steps."
        )

        return RewardCalculator.chain_feedback_reward(similarity)

    def _handle_submit(self, action: Action) -> Reward:
        """Terminal action — submit final cause and chain for grading."""
        final_cause = action.final_cause or ""
        final_chain = action.final_chain

        self._done = True

        # Compute final score
        self._final_score = Grader.compute_final_score(
            submitted_cause=final_cause,
            submitted_chain=final_chain,
            ground_truth_cause=self._ground_truth_cause,
            ground_truth_chain=self._ground_truth_chain,
            cause_type=self._ground_truth_cause_type,
            steps_used=self._steps_taken,
            max_steps=self._max_steps,
            n_wrong_hypotheses=self._wrong_hypotheses,
            contributing_causes=self._contributing_causes,
        )

        self._last_query_result = (
            f"=== INVESTIGATION COMPLETE ===\n"
            f"Submitted cause: {final_cause}\n"
            f"Submitted chain: {final_chain}\n"
            f"Ground truth cause: {self._ground_truth_cause}\n"
            f"Ground truth chain: {self._ground_truth_chain}\n"
            f"Final score: {self._final_score:.4f}\n"
            f"Steps used: {self._steps_taken}/{self._max_steps}\n"
            f"Wrong hypotheses: {self._wrong_hypotheses}\n"
            f"Facts discovered: {len(self._facts_discovered)}/{len(self._relevant_fact_ids)}"
        )

        return RewardCalculator.submit_reward(self._final_score)

    # --- helpers ---

    def _build_observation(self, reward: float = 0.01, done: bool = False) -> Observation:
        """Build the agent-visible observation."""
        # Build service list
        services = [Service(**s) for s in self._services]

        # Build available commits (visible metadata, not diffs)
        available_commits = [
            {
                "hash": c["hash"],
                "service": c["service"],
                "timestamp": c["timestamp"],
                "message": c["message"],
            }
            for c in self._commits
        ]

        # Build available config changes (visible metadata)
        available_configs = [
            {
                "config_id": c["config_id"],
                "service": c["service"],
                "timestamp": c["timestamp"],
                "description": c.get("description", ""),
            }
            for c in self._config_changes
        ]

        # Build available trace IDs
        available_traces = [t["trace_id"] for t in self._traces]

        # Build available infra events
        available_infra = [
            {
                "event_id": e["event_id"],
                "timestamp": e["timestamp"],
                "description": e["description"],
            }
            for e in self._infra_events
        ]

        # Clamp reward
        try:
            clamped_reward = float(reward)
            if clamped_reward != clamped_reward:  # NaN
                clamped_reward = 0.01
            safe_reward = round(min(max(clamped_reward, 0.01), 0.99), 4)
        except (ValueError, TypeError):
            safe_reward = 0.01

        return Observation(
            task_description=self._task_description,
            query_result=self._last_query_result,
            remaining_budget=max(0, self._max_steps - self._steps_taken),
            known_facts=list(self._known_facts),
            service_graph=self._service_graph,
            services=services,
            incident_window=self._incident_window,
            available_commits=available_commits,
            available_config_changes=available_configs,
            available_trace_ids=available_traces,
            available_infra_events=available_infra,
            step_number=self._steps_taken,
            max_steps=self._max_steps,
            message=self._message,
            hypotheses_submitted=len(self._hypotheses_submitted),
            wrong_hypotheses=self._wrong_hypotheses,
            reward=safe_reward,
            done=done,
        )

    def get_final_score(self) -> float:
        """Get the final graded score for the episode."""
        if self._final_score is not None:
            return self._final_score
        return 0.01
