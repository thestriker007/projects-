"""
Task 2 — Medium: "Maximize Throughput Under Interference"

Objective:
    Achieve high data throughput while keeping collision rate below 15%.
    Primary users are moderately active (p_on=0.20). The agent must
    balance: using higher power for more throughput vs. risking collisions.

Difficulty: MEDIUM
Episode length: 200 steps
PU setting: p_on=0.20, p_stay_on=0.80 (moderate activity)
Seed: fixed for reproducibility

Grader formula (dynamic — varies with agent behavior):
    collision_penalty = max(0, collision_rate - 0.15) * 3.0
    throughput_score  = total_throughput / theoretical_max_throughput
    score = max(0, min(1, throughput_score - collision_penalty))

    - Random agent:  score ≈ 0.10–0.25
    - Greedy agent:  score ≈ 0.35–0.55
    - Optimal agent: score ≈ 0.65–0.85
"""

from __future__ import annotations

TASK_ID = 2
TASK_NAME = "maximize_throughput"
DIFFICULTY = "medium"
MAX_STEPS = 200
SEED = 123

DESCRIPTION = (
    "Maximize data throughput while keeping collisions with primary users "
    "below 15%. Choose channels wisely and tune transmit power — higher power "
    "gives more throughput but increases interference risk. "
    "Score combines throughput efficiency with collision penalty."
)

# PU parameters for this task
PU_P_ON = 0.20
PU_P_STAY_ON = 0.80
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
    Compute task 2 grader score ∈ [0.0, 1.0].

    Balances throughput achievement against collision rate.
    Collision rate > 15% incurs a penalty that reduces the throughput score.

    Args:
        collision_count:            Times the agent collided with a PU
        total_steps:                Steps completed
        total_throughput:           Cumulative throughput (Mbps)
        theoretical_max_throughput: Max possible throughput if no collisions

    Returns:
        score ∈ [0.0, 1.0]
    """
    if total_steps == 0 or theoretical_max_throughput <= 0:
        return 0.0

    collision_rate = collision_count / total_steps
    throughput_score = total_throughput / theoretical_max_throughput

    # Penalize collision rate above 15% threshold
    collision_penalty = max(0.0, collision_rate - 0.15) * 3.0

    score = max(0.0, min(1.0, throughput_score - collision_penalty))
    return round(score, 4)
