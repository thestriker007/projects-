from openenv.core.env_server import create_web_interface_app
import logging
from server.environment import AdaptiveUIEnv
from models import UIAction, UIObservation

logging.basicConfig(level=logging.INFO)

# A proxy factory pattern to map dynamically if needed, 
# although openenv can instantiate directly via class path.
# Register schemas globally for fast API if OpenEnv needs them implicitly.
# We will use openenv's factory entry.
app = create_web_interface_app(
    environment_class=AdaptiveUIEnv,
    action_model=UIAction,
    observation_model=UIObservation
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
