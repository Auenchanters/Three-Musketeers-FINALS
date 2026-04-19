"""Observation models: what the agent sees at each step of the investigation."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from openenv.core.env_server.types import Observation as BaseObservation


class Service(BaseModel):
    """A service node in the 4-service cloud architecture."""
    name: str = Field(description="Service name, e.g. 'frontend', 'auth', 'data', 'batch'")
    status: str = Field(default="impacted", description="Current status during the incident")
    dependencies: List[str] = Field(default_factory=list, description="Services this service depends on")
    recent_deploy_count: int = Field(default=0, description="Number of deploys in the last 24h")
    error_rate_during_incident: Optional[float] = Field(default=None, description="Error rate (%) during the incident window")


class Observation(BaseObservation):
    """
    What the agent sees after each query/action.

    The agent does NOT receive the full telemetry haystack. Instead, each
    step() returns the result of whatever query the agent submitted.
    The service dependency graph is always fully visible.
    """
    task_description: str = Field(description="Natural language description of the incident to investigate")
    query_result: str = Field(default="", description="Text result from the last query (log lines, trace, diff, etc.)")
    remaining_budget: int = Field(ge=0, description="Steps remaining in the query budget")
    known_facts: List[str] = Field(default_factory=list, description="Facts the agent has confirmed so far")
    service_graph: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Service dependency DAG: service_name -> [upstream_dependencies]"
    )
    services: List[Service] = Field(default_factory=list, description="Services involved in the incident")
    incident_window: Dict[str, str] = Field(
        default_factory=dict,
        description="Incident time window: {'start': ISO timestamp, 'end': ISO timestamp}"
    )
    available_commits: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of recent commits visible to the agent: [{hash, service, timestamp, message}]"
    )
    available_config_changes: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of recent config changes visible: [{config_id, service, timestamp, description}]"
    )
    available_trace_ids: List[str] = Field(
        default_factory=list,
        description="List of trace IDs the agent can fetch"
    )
    available_infra_events: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of infra events: [{event_id, timestamp, description}]"
    )
    step_number: int = Field(ge=0, description="Current step in the episode")
    max_steps: int = Field(gt=0, description="Maximum steps allowed")
    message: str = Field(default="", description="Environment feedback after last action")
    hypotheses_submitted: int = Field(default=0, description="Number of hypotheses submitted so far")
    wrong_hypotheses: int = Field(default=0, description="Number of incorrect hypotheses submitted")
