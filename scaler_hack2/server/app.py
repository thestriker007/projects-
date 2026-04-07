"""FastAPI server app for SmartRadioEnvironment."""
from __future__ import annotations

from openenv.core import create_fastapi_app

from .smart_radio_environment import SmartRadioEnvironment
from models import SpectrumAction, SpectrumObservation

def create_app(task_id: int = 1):
    """Create the FastAPI app for a given task."""
    def env_factory():
        return SmartRadioEnvironment(task_id=task_id)
    return create_fastapi_app(env_factory, action_cls=SpectrumAction, observation_cls=SpectrumObservation)


# Default app for task 1 (can be overridden by env var)
import os
_task_id = int(os.environ.get("TASK_ID", "1"))
app = create_app(_task_id)


def main():
    """Entry point for the [project.scripts] server."""
    import uvicorn
    _port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=_port)


if __name__ == "__main__":
    main()
