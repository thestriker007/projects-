"""
Task 1 — Easy: "Find the Quiet Channel"

Objective:
    Identify and consistently transmit on the least-occupied RF channel.
    Primary users are rare (p_on=0.05). The agent just needs to sense
    which channels are free and stick to them.

Difficulty: EASY
Episode length: 100 steps
PU setting: p_on=0.05, p_stay_on=0.70 (low activity)
Seed: fixed for reproducibility

Grader formula (dynamic — varies with agent behavior):
    score = max(0.0, 1.0 - (collision_rate / 0.5))
    - Random agent:  collision_rate ≈ 0.35–0.45 → score ≈ 0.10–0.30
    - Greedy agent:  collision_rate ≈ 0.10–0.20 → score ≈ 0.60–0.80
    - Optimal agent: collision_rate ≈ 0.00–0.05 → score ≈ 0.90–1.00
"""

from __future__ import annotations

TASK_ID = 1
TASK_NAME = "find_quiet_channel"
DIFFICULTY = "easy"
MAX_STEPS = 100
SEED = 42

DESCRIPTION = (
    "Find the quietest RF channel and transmit on it consistently. "
    "Primary users are rare. Avoid channels with active primary users "
    "to maximize throughput. Score = 1.0 if no collisions."
)

# PU parameters for this task
PU_P_ON = 0.05
PU_P_STAY_ON = 0.70
N_CHANNELS = 8
T_HISTORY = 5


def grade(
    collision_count: int,
    total_steps: int,
    total_throughput: float,
    theoretical_max_throughput: float,
    **kwargs,
) -> float:
    """
    Compute task 1 grader score ∈ [0.0, 1.0].

    Purely collision-rate based: the agent should learn to avoid
    primary users entirely. This produces clear score variance:
        - 0 collisions → 1.0
        - 50% collision rate → 0.0

    Args:
        collision_count:           Number of times agent hit a PU
        total_steps:               Steps completed in episode
        total_throughput:          Not used for Task 1 grading
        theoretical_max_throughput: Not used for Task 1 grading

    Returns:
        score ∈ [0.0, 1.0]
    """
    if total_steps == 0:
        return 0.0

    collision_rate = collision_count / total_steps
    # Score = 1.0 at 0% collision, 0.0 at 50% collision
    score = max(0.0, 1.0 - (collision_rate / 0.5))
    return round(score, 4)
