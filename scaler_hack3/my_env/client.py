from __future__ import annotations
from openenv.core import EnvClient
from models import UIAction, UIObservation
from pydantic import BaseModel

# Environment states can be typed if state endpoint heavily used
class AdaptiveUIState(BaseModel):
    task_id: str
    grader_score: float

class AdaptiveUIClient(EnvClient[UIAction, UIObservation, AdaptiveUIState]):
    """
    Client for the AdaptiveUI OpenEnv server.
    
    Usage (async):
        async with AdaptiveUIClient(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(UIAction(...))
    """
    action_type = UIAction
    observation_type = UIObservation
    state_type = AdaptiveUIState
