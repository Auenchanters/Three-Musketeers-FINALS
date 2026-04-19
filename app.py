"""
PostmortemEnv — FastAPI Server using OpenEnv core

Exposes standard endpoints (and WebSockets) conforming to meta-pytorch/OpenEnv API.
Runs on port 7860 natively.
"""

from openenv.core.env_server import create_app
from models.action import Action
from models.observation import Observation
from engine.environment import PostmortemEnvironment

app = create_app(
    env=PostmortemEnvironment,
    action_cls=Action,
    observation_cls=Observation,
)


@app.get("/")
async def root():
    return {
        "name": "PostmortemEnv",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/ws"],
    }
