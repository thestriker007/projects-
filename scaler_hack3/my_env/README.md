---
title: Adaptive UI Gym
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Adaptive-UI-Gym
A Reinforcement Learning environment built for the Scaler Meta PyTorch Hackathon that simulates user behavior to help AI agents learn real-time UI personalization.

## Motivation
Modern web applications frequently face a tradeoff between aesthetic/conversion intensity and user patience. Most environments focus on raw conversions. **Adaptive-UI-Gym** uniquely distinguishes between **Organic Churn** (a user dropping off naturally) and **UI-Induced Churn** (a user leaving out of frustration due to aggressive UI design).

By dynamically tracking an `annoyance_level` against an `engagement_score`, the environment teaches agents to respect the user's emotional state over time, creating a "safety-first" behavior simulation model.

## Action Space
The agent has granular control over the user interface presentation through a strictly typed `UIAction` Pydantic model:
- `layout` (Literal["grid", "list", "hero"]): The UI layout type.
- `cta_style` (Literal["subtle", "bold", "pulsing"]): The Call-To-Action intensity.
- `color_scheme` (Literal["light", "dark", "high-contrast"]): The thematic style affecting readability and aesthetic load.
- `font_size` (Literal["small", "medium", "large"]): Readability scaling.

## Observation Space
The environment returns a tightly coupled `UIObservation` that models the user state:
- `annoyance_level` (float [0.0 - 1.0]): Friction caused by aggressive or conflicting UI elements.
- `engagement_score` (float [0.0 - 1.0]): Probability the user will complete their desired goal/CTA.
- `organic_exit_risk` (float [0.0 - 1.0]): Natural decay of user attention over a session.
- `session_duration` (float): Tracked consecutive steps within the current episode.

## Task Difficulty Spectrum
3 explicitly configured tasks dictate the grading boundaries.

1. **Patient User Onboarding (Easy)**
   - Expects a highly tolerant user. Maintains a decay rate of `0.01`.
   - **Agent Goal**: Keep the user engaged for a full 30-step threshold.

2. **Incentivized Goal Path (Medium)**
   - Represents a standard user with high distraction parameters (`decay_rate: 0.05`).
   - **Agent Goal**: Navigate the user toward a specific CTA "click" while ensuring low annoyance.

3. **High-Friction Balancing (Hard)**
   - The user is extremely irritable. Frustration compounds rapidly (`annoyance_multiplier: 2.0`).
   - **Agent Goal**: Ensure a CTA click but the agent suffers massive penalizations if `annoyance_level` spikes above `0.4`.

## Environment Setup & Inference
### 1. Requirements
Ensure you are using Python 3.10+ and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Baseline Grader Baseline Eval
A baseline `inference.py` comes pre-packaged. It relies on the Hugging Face Serverless endpoint via the `openai` python client schema. Run with:
```bash
export API_BASE_URL="https://router.huggingface.co/hf-inference/v1"
export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
export HF_TOKEN="<YOUR_TOKEN_HERE>"
python inference.py
```
*Note: Due to our robust Sigmoid boundary squashing formulation, baseline reward output perfectly bounds within standard floating parameters of `[0.01 - 0.99]` preventing divide by zero states for the scaler autograder!*

### 3. OpenEnv Execution & Docker 
To boot the FastAPI OpenEnv directly:
```bash
docker build -t adaptive-ui-gym .
docker run -p 8000:8000 adaptive-ui-gym
```
