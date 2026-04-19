"""Server entry point. Exposes PostmortemEnvironment via HTTP and WebSocket."""

try:
    from openenv.core.env_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from models.action import Action
    from models.observation import Observation
    from engine.environment import PostmortemEnvironment
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.action import Action
    from models.observation import Observation
    from engine.environment import PostmortemEnvironment


app = create_app(
    env=PostmortemEnvironment,
    action_cls=Action,
    observation_cls=Observation,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    Enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8000
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
