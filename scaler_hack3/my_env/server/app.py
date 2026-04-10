from openenv.core import create_fastapi_app
import logging
from .environment import AdaptiveUIEnv
from models import UIAction, UIObservation
import os

logging.basicConfig(level=logging.INFO)

def create_app(task_id: str = "easy"):
    def env_factory():
        return AdaptiveUIEnv(task_id=task_id)
    return create_fastapi_app(env_factory, action_cls=UIAction, observation_cls=UIObservation)

_task_id = os.environ.get("TASK_ID", "ui_easy_retention")
app = create_app(_task_id)

def main():
    import uvicorn
    _port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=_port)

if __name__ == "__main__":
    main()
