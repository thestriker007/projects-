"""
SmartRadio-RL: Pydantic Action + Observation + State models.
These define the OpenEnv contract — what the agent sends and receives.

MDP formulation:
  Action  (A): Discrete channel selection + Continuous transmit power
  State   (S): Spectrogram — [N_channels × T_history] occupancy matrix
  Reward  (R): Throughput − β·Collision Penalty − γ·Energy Cost
"""

from __future__ import annotations

from typing import Optional
from pydantic import Field

from openenv.core import Action, Observation, State


# ──────────────────────────────────────────────
# ACTION
# ──────────────────────────────────────────────

class SpectrumAction(Action):
    """
    Cognitive radio agent's two-part decision at each time step.

    1. Discrete  — which RF channel to jump to (CH₁ … CHₙ)
    2. Continuous — how much transmit power to use (0–30 dBm)

    Higher power increases throughput (better SNR) but also increases
    the risk of interfering with primary users on adjacent channels.
    """

    channel_id: int = Field(
        ...,
        ge=0,
        description="Zero-indexed channel to transmit on (must be < n_channels)",
    )
    tx_power_dbm: float = Field(
        default=20.0,
        ge=0.0,
        le=30.0,
        description="Transmit power in dBm. Range: [0.0, 30.0]",
    )


# ──────────────────────────────────────────────
# OBSERVATION
# ──────────────────────────────────────────────

class SpectrumObservation(Observation):
    """
    What the agent sees on the spectrum waterfall (spectrogram).

    channel_occupancy is the flattened [N_channels × T_history] matrix.
    Each value ∈ [0.0, 1.0] is the estimated occupancy probability
    of that channel at that time slice (1.0 = definitely occupied by PU).

    The agent must learn to find "white spaces" (values near 0.0) and
    exploit them for transmission without colliding with primary users.
    """

    # ── Core spectrum snapshot ──────────────────
    channel_occupancy: list[float] = Field(
        ...,
        description=(
            "Flattened spectrogram [N_channels × T_history]. "
            "Index: [ch * T_history + t]. Value ∈ [0.0, 1.0]."
        ),
    )
    n_channels: int = Field(..., ge=1, description="Number of RF channels monitored")
    t_history: int = Field(..., ge=1, description="Number of historical time steps included")

    # ── Last-step feedback (partial progress signals) ──
    last_throughput: float = Field(
        default=0.0,
        description="Shannon capacity achieved in the previous step (Mbps)",
    )
    last_collision: bool = Field(
        default=False,
        description="True if the previous action caused a collision with a Primary User",
    )
    last_reward: float = Field(
        default=0.0,
        description="R = Throughput − β·Collision − γ·Energy from the previous step",
    )

    # ── Episode-level accumulators ──────────────
    collision_count: int = Field(
        default=0,
        description="Total number of Primary User collisions so far this episode",
    )
    total_throughput: float = Field(
        default=0.0,
        description="Cumulative throughput (Mbps) accumulated this episode",
    )
    step_number: int = Field(
        default=0,
        description="Current step index within the episode",
    )

    # ── Task context (helps LLM agents) ────────
    task_id: int = Field(
        default=1,
        description="Active task ID (1=Easy, 2=Medium, 3=Hard)",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of the current task objective",
    )

    def to_text(self) -> str:
        """
        Render the observation as human-readable text for LLM agents
        (Nemotron 3 Super, GPT-4o-mini, etc.).
        """
        lines = [
            f"=== SmartRadio Channel Status (Step {self.step_number}) ===",
            f"Task: {self.task_description}",
            "",
            "Channel Occupancy (0.0=free, 1.0=primary user active):",
        ]
        # Show current time slice only (most recent column)
        for ch in range(self.n_channels):
            idx = ch * self.t_history + (self.t_history - 1)
            occ = self.channel_occupancy[idx]
            bar = "█" * int(occ * 10) + "░" * (10 - int(occ * 10))
            status = "OCCUPIED" if occ > 0.5 else "FREE ✓"
            lines.append(f"  CH{ch:>2}: [{bar}] {occ:.2f}  {status}")

        lines += [
            "",
            f"Last step: throughput={self.last_throughput:.3f} Mbps, "
            f"collision={self.last_collision}, reward={self.last_reward:.3f}",
            f"Episode total: throughput={self.total_throughput:.2f} Mbps, "
            f"collisions={self.collision_count}",
            "",
            "Choose a FREE channel (low occupancy) with appropriate power.",
            'Respond ONLY with JSON: {"channel_id": <int>, "tx_power_dbm": <float 0-30>}',
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────

class SpectrumState(State):
    """
    Internal episode state — returned by env.state().
    Includes grader scores (0.0–1.0) for the current task.
    """

    task_id: int = Field(default=1, description="Current task ID")
    task_name: str = Field(default="find_quiet_channel", description="Task name")
    difficulty: str = Field(default="easy", description="Task difficulty level")
    max_steps: int = Field(default=100, description="Maximum steps in the episode")
    seed: Optional[int] = Field(default=None, description="RNG seed used for reproducibility")

    # Grader outputs
    grader_score: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Current grader score ∈ [0.0, 1.0]. Varies with agent behavior.",
    )
    collision_rate: float = Field(
        default=0.0,
        description="collision_count / steps_taken",
    )
    spectral_efficiency: float = Field(
        default=0.0,
        description="total_throughput / theoretical_max_throughput",
    )
