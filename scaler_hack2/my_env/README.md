---
title: SmartRadio OpenEnv
emoji: 📻
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# SmartRadio-RL: Dynamic Spectrum Access via Reinforcement Learning

> **Hackathon**: Scaler School of Technology × Meta PyTorch (OpenEnv) Hackathon  
> **Environment type**: OpenEnv (step/reset/state API, Docker, Hugging Face Spaces)

## What is This?

A **cognitive radio environment** where an AI agent learns to act as a Radio Network Engineer — finding unused RF spectrum ("white spaces") to transmit data, without ever disrupting licensed Primary Users.

In real-world wireless networks (5G, Wi-Fi, radar coexistence), spectrum is scarce. Static allocation wastes 70–90% of licensed spectrum at any given moment. **Dynamic Spectrum Access (DSA)** lets secondary devices opportunistically use idle channels — but they must instantly yield when the primary user returns.

This environment trains agents to do exactly that.

---

## Action Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `channel_id` | `int` | `[0, 7]` | Which RF channel to transmit on (discrete) |
| `tx_power_dbm` | `float` | `[0.0, 30.0]` | Transmit power in dBm (continuous) |

## Observation Space

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `channel_occupancy` | `list[float]` | `[N_ch × T_hist]` | Flattened spectrogram. Value ∈ `[0.0, 1.0]` = occupancy probability per channel per time step |
| `last_throughput` | `float` | scalar | Shannon capacity achieved in previous step (Mbps) |
| `last_collision` | `bool` | scalar | Whether previous action caused a Primary User collision |
| `last_reward` | `float` | scalar | `R = Throughput − β·Collision − γ·Energy` from last step |
| `collision_count` | `int` | scalar | Cumulative collisions this episode |
| `total_throughput` | `float` | scalar | Cumulative throughput this episode (Mbps) |
| `task_description` | `str` | — | Human-readable description of the current task |

## Reward Function

$$R = \text{Throughput} - \beta \cdot \text{Collision} - \gamma \cdot \text{Energy}$$

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Throughput | `B·log₂(1+SNR)` Mbps | Shannon–Hartley capacity if channel is free |
| β (collision) | `10.0` | Heavy penalty for transmitting on a PU channel |
| γ (energy) | `0.1` | Small penalty for high power usage |

Reward is **dense** — computed at every step, not just episode end.

---

## Tasks

| Task | ID | Difficulty | Steps | PU Activity | Grader Formula |
|------|----|-----------|-------|-------------|----------------|
| Find Quiet Channel | 1 | Easy | 100 | p_on=5% | `1 - collision_rate/0.5` |
| Maximize Throughput | 2 | Medium | 200 | p_on=20% | `throughput_score - collision_penalty` |
| Coexist with Bursty PUs | 3 | Hard | 300 | p_on=40%, correlated | Weighted: throughput + collision + adaptation |

### Task 1 — Easy: Find the Quiet Channel
Primary users are rare. The agent must identify free channels and consistently transmit on them.

### Task 2 — Medium: Maximize Throughput Under Interference
Moderate PU activity. The agent balances throughput (use high power on free channels) vs. collision risk (avoid occupied ones). Scoring penalizes collision rate above 15%.

### Task 3 — Hard: Coexist with Bursty Primary Users
Dense, correlated PU bursts. At **step 150**, the PU pattern **inverts** (previously busy channels become free). The agent must detect and adapt to this shift without being told.

---

## Baseline Scores (LLM Agent — `gpt-4o-mini`)

| Task | Difficulty | Seed 42 | Seed 123 | Seed 456 | **Avg** |
|------|-----------|---------|---------|---------|---------|
| 1 | Easy | — | — | — | — |
| 2 | Medium | — | — | — | — |
| 3 | Hard | — | — | — | — |

> Run `python3 inference.py` to generate scores. Results fill this table.

---

## Setup & Usage

### Option 1 — Docker (recommended)

```bash
docker build -t smart-radio-env .
docker run -p 7860:7860 -e TASK_ID=1 smart-radio-env
```

Change `TASK_ID` to `1`, `2`, or `3` to select the task.

### Option 2 — Local (no Docker)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Connect with the Client

```python
from client import SmartRadioEnv
from models import SpectrumAction

with SmartRadioEnv(base_url="http://localhost:7860").sync() as env:
    obs = env.reset(seed=42)
    print(obs.to_text())   # LLM-friendly spectrum view

    result = env.step(SpectrumAction(channel_id=2, tx_power_dbm=22.0))
    print(f"Reward: {result.reward:.3f}  Done: {result.done}")

    state = env.state()
    print(f"Grader score: {state.grader_score:.4f}")
```

### Run Baseline

```bash
export HF_TOKEN="sk-..."
python3 inference.py
```

---

## Environment Motivation

Dynamic Spectrum Access is an **active research area** in wireless communications:
- Used in **5G NR unlicensed (NR-U)** coexistence with Wi-Fi
- Studied by IEEE 802.22 (Cognitive Radio standard) for TV white space
- Deployed by companies like Qualcomm, Ericsson, and AT&T for spectrum sharing

Training an RL agent on this environment has direct real-world value for evaluating spectrum management policies.
