"""Action models for the PostmortemEnv agent."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum
from openenv.core.env_server.types import Action as BaseAction


class ActionType(str, Enum):
    """The investigation and mitigation actions available to the agent."""
    QUERY_LOGS = "query_logs"
    FETCH_TRACE = "fetch_trace"
    DIFF_COMMIT = "diff_commit"
    INSPECT_CONFIG = "inspect_config"
    INSPECT_INFRA = "inspect_infra"
    DISCOVER_TOPOLOGY = "discover_topology"
    HYPOTHESIZE = "hypothesize"
    EXPLAIN_CHAIN = "explain_chain"
    SUBMIT = "submit"


class Action(BaseAction):
    """
    What the agent does at each step.

    Every action requires an action_type. Different actions use different
    parameter subsets:

    - query_logs: service, keyword (optional: time_window)
    - fetch_trace: trace_id
    - diff_commit: commit_hash
    - inspect_config: config_id
    - inspect_infra: event_id
    - discover_topology: service (narrative mode only — reveals that
      service's dependencies; omit ``service`` to reveal everything)
    - hypothesize: cause_entity_id
    - explain_chain: chain (ordered list of {service, effect})
    - submit: final_cause, final_chain
    """
    action_type: ActionType = Field(description="The type of investigation action to perform")

    # query_logs parameters
    service: Optional[str] = Field(default=None, description="Target service name for query_logs")
    keyword: Optional[str] = Field(default=None, description="Search keyword for query_logs")
    time_window: Optional[str] = Field(default=None, description="Time window filter, e.g. 'last_5m', 'last_1h'")

    # fetch_trace parameters
    trace_id: Optional[str] = Field(default=None, description="Trace ID to fetch")

    # diff_commit parameters
    commit_hash: Optional[str] = Field(default=None, description="Commit hash to diff")

    # inspect_config parameters
    config_id: Optional[str] = Field(default=None, description="Config change ID to inspect")

    # inspect_infra parameters
    event_id: Optional[str] = Field(default=None, description="Infrastructure event ID to inspect")

    # hypothesize parameters
    cause_entity_id: Optional[str] = Field(default=None, description="Entity ID for hypothesis (e.g. 'commit-abc123')")

    # explain_chain parameters
    chain: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Ordered causal chain: [{service, effect}, ...]"
    )


    # submit parameters
    final_cause: Optional[str] = Field(default=None, description="Final root cause entity ID")
    final_chain: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Final causal chain: [{service, effect}, ...]"
    )

    # general
    reason: Optional[str] = Field(default=None, description="Agent's reasoning (for logging)")
