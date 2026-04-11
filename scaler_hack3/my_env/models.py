from pydantic import BaseModel, Field
from typing import Literal
from openenv.core import Action, Observation, State

class UIState(State):
    grader_score: float = Field(0.01, description="Current grader score")
    
class UIAction(Action):
    layout: Literal["grid", "list", "hero"] = Field(..., description="UI layout type")
    cta_style: Literal["subtle", "bold", "pulsing"] = Field(..., description="Call-to-action intensity")
    color_scheme: Literal["light", "dark", "high-contrast"] = Field(..., description="Color theme of the UI")
    font_size: Literal["small", "medium", "large"] = Field(..., description="Readability scaling")

class UIObservation(Observation):
    annoyance_level: float = Field(..., description="Estimated user frustration [0-1]")
    engagement_score: float = Field(..., description="User activity level [0-1]")
    organic_exit_risk: float = Field(..., description="Probability of user leaving naturally [0-1]")
    session_duration: float = Field(..., description="How long the user has been in the session (steps)")
