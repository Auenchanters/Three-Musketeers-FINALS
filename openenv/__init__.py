# Shim: makes `from openenv.core.env_server.types import ...` resolve locally.
# The upstream openenv-core package uses Python 3.10+ features (kw_only=True)
# that break on 3.9. This shim replicates the exact base classes the project
# needs using Pydantic instead of dataclasses so they're compatible with the
# Pydantic models in models/observation.py.
