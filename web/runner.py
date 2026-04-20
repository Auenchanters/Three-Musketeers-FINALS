"""
Run manager + FastAPI routes for the live UI.

A "run" is a single agent episode. When started it spawns a background task
that iterates the agent generator and pushes events onto an asyncio.Queue.
Clients subscribe via SSE (/api/stream/{run_id}) and receive the same events
as they're produced. The full event log is also persisted to disk so late
subscribers / replays see the whole history.

Events written to the stream:
    reset, step, thought, usage, done, error

The manager stops the container from holding open file handles or memory by
pruning runs older than MAX_RUNS when a new one starts.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from data.generator import get_all_tasks
from engine.environment import PostmortemEnvironment
from web import curriculum
from web.agents import LLMAgent, OracleAgent, RandomAgent

RUNS_DIR = Path(__file__).resolve().parent.parent / "training_data" / "runs"
MAX_RUNS = 50
MAX_EVENTS_PER_RUN = 400

router = APIRouter(prefix="/api", tags=["live"])


@dataclass
class Run:
    run_id: str
    agent_type: str
    task_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    status: str = "starting"  # starting | running | done | error
    final_score: float | None = None
    steps_used: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


class _RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = asyncio.Lock()

    async def create(self, agent_type: str, task_id: str) -> Run:
        async with self._lock:
            if len(self._runs) >= MAX_RUNS:
                oldest = sorted(self._runs.values(), key=lambda r: r.started_at)[0]
                self._runs.pop(oldest.run_id, None)
            run = Run(run_id=uuid.uuid4().hex[:12], agent_type=agent_type, task_id=task_id)
            self._runs[run.run_id] = run
            return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        items = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
        return [
            {
                "run_id": r.run_id,
                "agent_type": r.agent_type,
                "task_id": r.task_id,
                "status": r.status,
                "final_score": r.final_score,
                "steps_used": r.steps_used,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
            }
            for r in items
        ]


_store = _RunStore()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class StartRunBody(BaseModel):
    agent: str = "oracle"  # oracle | random | llm
    task_id: str = "task1_recent_deploy"
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None


class StartRunResponse(BaseModel):
    run_id: str
    agent: str
    task_id: str


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _make_agent(body: StartRunBody):
    kind = body.agent.lower().strip()
    if kind == "oracle":
        return OracleAgent()
    if kind == "random":
        return RandomAgent()
    if kind == "llm":
        return LLMAgent(provider=body.provider, model=body.model, api_key=body.api_key)
    raise HTTPException(400, f"Unknown agent type: {body.agent}")


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------


async def _run_agent(run: Run, agent, task_id: str) -> None:
    env = PostmortemEnvironment()
    run.status = "running"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        async for event in agent.run(env, task_id):
            await _push(run, event)
            if len(run.events) >= MAX_EVENTS_PER_RUN:
                await _push(
                    run,
                    {"type": "error", "message": "event cap reached; stopping"},
                )
                break
            if event.get("type") == "done":
                run.final_score = float(event.get("score", 0.0))
                run.steps_used = int(event.get("steps", 0))
            if event.get("type") == "error":
                run.status = "error"
                break
        else:
            if run.status != "error":
                run.status = "done"
    except Exception as exc:  # noqa: BLE001
        await _push(run, {"type": "error", "message": f"runner crash: {exc}"})
        run.status = "error"
    finally:
        run.ended_at = time.time()
        # Update curriculum only for terminal, scored runs
        if run.final_score is not None and run.status == "done":
            try:
                delta = curriculum.record_run(
                    agent_id=getattr(agent, "agent_id", run.agent_type),
                    task_id=task_id,
                    score=run.final_score,
                    steps_used=run.steps_used,
                    max_steps=env.state.max_steps if getattr(env, "_initialized", False) else 40,
                )
                await _push(run, {"type": "curriculum", **delta})
            except Exception as exc:  # noqa: BLE001
                await _push(run, {"type": "error", "message": f"curriculum update failed: {exc}"})
        # Close all subscriber queues with sentinel
        for q in run.subscribers:
            await q.put(None)
        # Persist event log
        try:
            (RUNS_DIR / f"{run.run_id}.json").write_text(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "agent_type": run.agent_type,
                        "task_id": run.task_id,
                        "status": run.status,
                        "final_score": run.final_score,
                        "steps_used": run.steps_used,
                        "started_at": run.started_at,
                        "ended_at": run.ended_at,
                        "events": run.events,
                    },
                    indent=2,
                )
            )
        except OSError:
            pass


async def _push(run: Run, event: dict[str, Any]) -> None:
    run.events.append(event)
    for q in list(run.subscribers):
        await q.put(event)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/tasks")
async def list_tasks() -> JSONResponse:
    from data.generator import load_scenario

    items = []
    for tid in get_all_tasks(include_generated=False):
        try:
            scen = load_scenario(tid)
            items.append(
                {
                    "task_id": tid,
                    "name": scen.get("task_name") or tid,
                    "difficulty": scen.get("task_difficulty", "?"),
                    "description": scen.get("task_description", ""),
                    "max_steps": scen.get("max_steps", 40),
                    "curriculum": curriculum.get_task_state(tid),
                }
            )
        except Exception:
            continue
    return JSONResponse(items)


@router.get("/curriculum")
async def get_curriculum() -> JSONResponse:
    return JSONResponse(curriculum.load_state())


@router.get("/runs")
async def list_runs() -> JSONResponse:
    return JSONResponse(_store.list())


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    run = _store.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return JSONResponse(
        {
            "run_id": run.run_id,
            "agent_type": run.agent_type,
            "task_id": run.task_id,
            "status": run.status,
            "final_score": run.final_score,
            "steps_used": run.steps_used,
            "events": run.events,
        }
    )


@router.post("/runs")
async def start_run(body: StartRunBody) -> StartRunResponse:
    agent = _make_agent(body)
    run = await _store.create(body.agent, body.task_id)
    asyncio.create_task(_run_agent(run, agent, body.task_id))
    return StartRunResponse(run_id=run.run_id, agent=body.agent, task_id=body.task_id)


@router.get("/stream/{run_id}")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    run = _store.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")

    queue: asyncio.Queue = asyncio.Queue()
    run.subscribers.append(queue)

    # Replay any events already produced so late subscribers see the whole run
    backlog = list(run.events)

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            for ev in backlog:
                yield _sse(ev)
            while True:
                if await request.is_disconnected():
                    break
                ev = await queue.get()
                if ev is None:
                    yield _sse({"type": "_eof"})
                    break
                yield _sse(ev)
        finally:
            try:
                run.subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event: dict[str, Any]) -> bytes:
    payload = json.dumps(event, default=str)
    return f"event: {event.get('type', 'message')}\ndata: {payload}\n\n".encode("utf-8")
