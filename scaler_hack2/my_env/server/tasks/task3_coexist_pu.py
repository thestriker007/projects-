"""
Task 3 — Hard: "Coexist with Bursty Primary Users" (Nemotron Challenge)

Objective:
    Adapt to unpredictable, correlated PU burst patterns without any
    prior knowledge of their rhythm. At step 150, the PU activity pattern
    SHIFTS (previously busy channels become quiet, and vice versa).
    A good agent detects this shift and re-adapts quickly.

Difficulty: HARD
Episode length: 300 steps
PU setting: p_on=0.40, p_stay_on=0.90, inter-channel correlation=0.35
Phase shift: at step 150, PU states flip (hardest adaptation test)
Seed: fixed for reproducibility

Grader formula (dynamic — tests genuine adaptability):
    w1, w2, w3 = 0.50, 0.35, 0.15
    throughput_norm  = total_throughput / theoretical_max
    collision_norm   = 1 - min(1, collision_count / (total_steps * 0.5))
    adaptation_score = post_shift_score / (pre_shift_score + 1e-6)  [clamped 0-1]
    score = max(0, min(1, w1*throughput + w2*collision + w3*adaptation))

    - Random agent:         score ≈ 0.10–0.20
    - Nemotron 3 Super LLM: score ≈ 0.25–0.35
    - Trained DQN:          score ≈ 0.55–0.70
"""

from __future__ import annotations

PHASE_SHIFT_STEP = 150  # When PU pattern inverts

TASK_ID = 3
TASK_NAME = "coexist_primary_users"
DIFFICULTY = "hard"
MAX_STEPS = 300
SEED = 456

DESCRIPTION = (
    "Coexist with bursty, correlated primary users across 8 channels. "
    "At step 150, the primary user activity pattern SHIFTS — channels that "
    "were busy become free, and vice versa. You must detect and adapt to "
    "this change without being told it happened. "
    "Score rewards throughput, low collision rate, and post-shift adaptation."
)

# PU parameters for this task
PU_P_ON = 0.40
PU_P_STAY_ON = 0.90
PU_CORRELATION = 0.35
N_CHANNELS = 8
T_HISTORY = 8  # Longer history helps agent detect the shift


def grade(
    collision_count: int,
    total_steps: int,
    total_throughput: float,
    theoretical_max_throughput: float,
    pre_shift_collisions: int = 0,
    post_shift_collisions: int = 0,
    pre_shift_throughput: float = 0.0,
    post_shift_throughput: float = 0.0,
    **kwargs,
) -> float:
    """
    Compute task 3 grader score ∈ [0.0, 1.0].

    Three-component weighted score:
    1. Throughput norm  (50%): How much data was transferred overall
    2. Collision norm   (35%): How well the agent avoided primary users
    3. Adaptation score (15%): Did agent recover after the phase shift?

    Args:
        collision_count:            Total collisions across full episode
        total_steps:                Total steps completed
        total_throughput:           Total throughput (Mbps)
        theoretical_max_throughput: Max possible throughput (no PU, max power)
        pre_shift_collisions:       Collisions in steps 0–149
        post_shift_collisions:      Collisions in steps 150–299
        pre_shift_throughput:       Throughput in steps 0–149
        post_shift_throughput:      Throughput in steps 150–299

    Returns:
        score ∈ [0.0, 1.0]
    """
    if total_steps == 0 or theoretical_max_throughput <= 0:
        return 0.0

    w1, w2, w3 = 0.50, 0.35, 0.15

    # Component 1: Overall throughput efficiency
    throughput_norm = min(1.0, total_throughput / theoretical_max_throughput)

    # Component 2: Collision avoidance (0 collisions → 1.0, ≥50% rate → 0.0)
    collision_rate = collision_count / total_steps
    collision_norm = max(0.0, 1.0 - min(1.0, collision_rate / 0.5))

    # Component 3: Post-shift adaptation
    # Compare throughput rate pre vs post shift — good agent improves after shift
    pre_steps = min(total_steps, PHASE_SHIFT_STEP)
    post_steps = max(0, total_steps - PHASE_SHIFT_STEP)

    if post_steps > 0 and pre_steps > 0:
        pre_rate = pre_shift_throughput / pre_steps
        post_rate = post_shift_throughput / post_steps
        # Ratio > 1 means agent improved after shift (adapted)
        # Ratio < 1 means agent got worse (failed to adapt)
        adaptation_raw = post_rate / (pre_rate + 1e-6)
        adaptation_score = min(1.0, max(0.0, adaptation_raw))
    else:
        adaptation_score = 0.5  # neutral if episode didn't reach shift

    score = w1 * throughput_norm + w2 * collision_norm + w3 * adaptation_score
    return round(max(0.0, min(1.0, score)), 4)
