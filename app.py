"""
PostmortemEnv — FastAPI Server using OpenEnv core.

Exposes:
  * OpenEnv-standard endpoints (/reset, /step, /state, /health, /ws, /docs)
    wired via openenv.core.env_server.create_app
  * Live console UI at /        (static HTML/CSS/JS, no build step)
  * Streaming agent runner at /api/* (see web.runner)

Runs on port 7860 natively for HuggingFace Spaces.
"""

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_app

from models.action import Action
from models.observation import Observation
from engine.environment import PostmortemEnvironment
from web.runner import router as live_router

app = create_app(
    env=PostmortemEnvironment,
    action_cls=Action,
    observation_cls=Observation,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(live_router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/manifest.json", include_in_schema=False)
async def manifest() -> dict:
    return {
        "name": "PostmortemEnv",
        "short_name": "PostmortemEnv",
        "description": "Epistemic RL environment for cloud outage investigation",
        "version": "1.0.0",
        "endpoints": {
            "openenv": ["/health", "/reset", "/step", "/state", "/schema", "/ws"],
            "live": ["/api/tasks", "/api/curriculum", "/api/runs", "/api/stream/{run_id}"],
        },
    }
