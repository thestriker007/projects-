import numpy as np
import math
from typing import Dict, Any, Optional
from openenv.core.env_server import Environment, StepResult
from models import UIAction, UIObservation
from pydantic import BaseModel

class UIState(BaseModel):
    annoyance: float = 0.0
    engagement: float = 0.0
    organic_decay: float = 0.0
    session_duration: float = 0.0
    clicked: bool = False

class AdaptiveUIEnv(Environment):
    def __init__(self, task_id: str = "easy", decay_rate: float = 0.01, annoyance_multiplier: float = 1.0):
        super().__init__()
        self.task_id = task_id
        self.decay_rate = decay_rate
        self.annoyance_multiplier = annoyance_multiplier
        self.state = UIState()

    def reset(self, config: Optional[Dict[str, Any]] = None) -> UIObservation:
        self.state = UIState()
        return self._get_obs()

    def _get_obs(self) -> UIObservation:
        return UIObservation(
            annoyance_level=float(np.clip(self.state.annoyance, 0, 1)),
            engagement_score=float(np.clip(self.state.engagement, 0, 1)),
            organic_exit_risk=float(np.clip(self.state.organic_decay, 0, 1)),
            session_duration=self.state.session_duration
        )

    def _squash(self, val: float) -> float:
        return float(np.clip(val, 0.01, 0.99))

    def _sigmoid_squash(self, r: float, r_mid: float = 0.5, k: float = 10.0) -> float:
        sig = 1.0 / (1.0 + math.exp(-k * (r - r_mid)))
        return float(np.clip(0.01 + 0.98 * sig, 0.01, 0.99))

    def step(self, action: UIAction) -> StepResult:
        self.state.session_duration += 1.0
        
        # 1. Physics: Simulate User Reaction based on UI Choices
        friction = 0.0
        engagement_boost = 0.0

        if action.cta_style == "pulsing":
            friction += 0.2
            engagement_boost += 0.15
        elif action.cta_style == "bold":
            friction += 0.05
            engagement_boost += 0.1
        else: # subtle
            friction -= 0.05
            engagement_boost -= 0.05

        if action.layout == "hero":
            friction += 0.1
        elif action.layout == "grid":
            engagement_boost += 0.05

        if action.color_scheme == "high-contrast":
            friction += 0.1
            engagement_boost += 0.1
        elif action.color_scheme == "dark":
            friction -= 0.05

        if action.font_size == "small":
            friction += 0.1
        elif action.font_size == "large":
            engagement_boost += 0.05

        friction *= self.annoyance_multiplier
        
        # Apply state changes
        self.state.annoyance = float(np.clip(self.state.annoyance + friction, 0.0, 1.0))
        self.state.engagement = float(np.clip(self.state.engagement + engagement_boost, 0.0, 1.0))
        
        # Organic decay models the user's natural drop-off over time
        self.state.organic_decay = float(np.clip(self.state.organic_decay + self.decay_rate, 0.0, 1.0))

        if self.state.engagement > 0.8:
            self.state.clicked = True

        # Termination Conditions
        done = False
        
        if self.state.annoyance > 0.85:
            done = True
        elif self.state.organic_decay > 0.95:
            done = True
        elif self.state.clicked and self.task_id != "easy":
            done = True
        elif self.state.session_duration >= 30 and self.task_id == "easy":
            done = True

        # 2. Causal Attribution & Reward Calculation
        reward = 0.5
        
        if self.task_id in ["easy", "ui_easy_retention"]:
            # Task: Maintain engagement for 30s with a "patient" user.
            s = 0.1 + 0.8 * min(self.state.session_duration / 30.0, 0.99)
            reward = self._squash(s)
            
        elif self.task_id in ["medium", "ui_med_conversion"]:
            # Task: Guide a "distracted" user to click a CTA.
            s = 0.2
            if self.state.clicked:
                s += 0.79
            reward = self._squash(s)
            
        elif self.task_id in ["hard", "ui_hard_fatigue"]:
            # Task: Maximize clicks while keeping "Annoyance" < 0.4.
            s = float(np.clip(0.99 - self.state.annoyance, 0.01, 0.99))
            reward = self._squash(s)
            
        return StepResult(observation=self._get_obs(), reward=reward, done=done)
