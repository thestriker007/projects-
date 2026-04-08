"""
SmartRadioEnvironment — the core OpenEnv Environment implementation.

Implements the standard OpenEnv interface:
  - reset(seed, episode_id)  →  SpectrumObservation
  - step(action)             →  SpectrumObservation
  - state                    →  SpectrumState (property)

The environment runs one of 3 tasks depending on the task_id
passed via the URL query param or during reset.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from openenv.core import Environment

from models import SpectrumAction, SpectrumObservation, SpectrumState
from .primary_user import PrimaryUserSimulator
from .channel_model import (
    MAX_THROUGHPUT_MBPS,
    compute_snr,
    compute_reward,
)
from .tasks import task1_find_quiet, task2_maximize_tput, task3_coexist_pu

# ── Task registry ─────────────────────────────────────────────────
TASKS = {
    1: task1_find_quiet,
    2: task2_maximize_tput,
    3: task3_coexist_pu,
}


class SmartRadioEnvironment(Environment[SpectrumAction, SpectrumObservation, SpectrumState]):
    """
    Cognitive radio DSA environment for OpenEnv.

    A secondary user agent must learn to opportunistically transmit on
    unused RF channels (white spaces) without colliding with primary users.
    The agent observes a rolling spectrogram and chooses which channel to
    use and at what power level.

    MDP:
      S — Spectrogram [N_channels × T_history] occupancy matrix
      A — (channel_id: discrete, tx_power_dbm: continuous 0–30)
      R — Throughput − β·Collision − γ·Energy  (dense, every step)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: int = 1):
        super().__init__()

        if task_id not in TASKS:
            raise ValueError(f"task_id must be 1, 2, or 3. Got: {task_id}")

        task_module = TASKS[task_id]
        self._task_id = task_id
        self._task_module = task_module
        self._n_channels: int = task_module.N_CHANNELS
        self._t_history: int = task_module.T_HISTORY
        self._max_steps: int = task_module.MAX_STEPS
        self._difficulty: str = task_module.DIFFICULTY
        self._default_seed: int = task_module.SEED

        # Episode state (initialized in reset)
        self._rng: random.Random = random.Random(self._default_seed)
        self._pu_sim: PrimaryUserSimulator = PrimaryUserSimulator(
            n_channels=self._n_channels,
            difficulty=self._difficulty,
            seed=self._default_seed,
        )
        self._occupancy_history: list[list[float]] = []  # shape: [T_history][N_channels]
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._collision_count: int = 0
        self._total_throughput: float = 0.0
        self._pre_shift_collisions: int = 0
        self._post_shift_collisions: int = 0
        self._pre_shift_throughput: float = 0.0
        self._post_shift_throughput: float = 0.0
        self._theoretical_max: float = (
            MAX_THROUGHPUT_MBPS * self._max_steps
        )  # upper bound used for normalization

    # ──────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SpectrumObservation:
        """
        Begin a new episode.

        Args:
            seed:       RNG seed for reproducibility (uses task default if None)
            episode_id: Unique episode identifier (auto-generated if None)

        Returns:
            Initial SpectrumObservation with clean slate
        """
        effective_seed = seed if seed is not None else self._default_seed
        self._episode_id = episode_id or str(uuid.uuid4())
        self._rng = random.Random(effective_seed)

        # Reset PU simulator
        self._pu_sim = PrimaryUserSimulator(
            n_channels=self._n_channels,
            difficulty=self._difficulty,
            seed=effective_seed,
        )
        initial_pu = self._pu_sim.reset(seed=effective_seed)

        # Reset episode counters
        self._step_count = 0
        self._collision_count = 0
        self._total_throughput = 0.0
        self._pre_shift_collisions = 0
        self._post_shift_collisions = 0
        self._pre_shift_throughput = 0.0
        self._post_shift_throughput = 0.0

        # Initialize occupancy history: T_history columns of [N_channels] rows
        initial_occ = [float(p) for p in initial_pu]
        self._occupancy_history = [initial_occ[:] for _ in range(self._t_history)]

        return self._build_observation(
            last_reward=0.0,
            last_throughput=0.0,
            last_collision=False,
        )

    # ──────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────

    def step(
        self,
        action: SpectrumAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SpectrumObservation:
        """
        Execute one time step.

        1. Advance PU Markov chains
        2. Compute SNR and reward: R = Throughput − β·Collision − γ·Energy
        3. Update occupancy history
        4. Return new observation (with done=True at episode end)

        Args:
            action: SpectrumAction with channel_id and tx_power_dbm

        Returns:
            SpectrumObservation with new spectrum view + reward signal
        """
        # Validate action channel index
        if action.channel_id >= self._n_channels:
            action = SpectrumAction(
                channel_id=self._n_channels - 1,
                tx_power_dbm=action.tx_power_dbm,
            )

        # 1. Advance primary users
        pu_active = self._pu_sim.step()

        # 2. Compute SNR for chosen channel
        snr = compute_snr(
            tx_power_dbm=action.tx_power_dbm,
            channel_id=action.channel_id,
            rng=self._rng,
        )

        # 3. Compute reward: R = Throughput − β·Collision − γ·Energy
        reward, throughput, collision = compute_reward(
            chosen_channel=action.channel_id,
            tx_power_dbm=action.tx_power_dbm,
            pu_active=pu_active,
            snr_linear=snr,
        )

        # 4. Update episode state
        self._step_count += 1
        if collision:
            self._collision_count += 1
        self._total_throughput += throughput

        # Track pre/post phase-shift stats for Task 3
        if self._step_count < task3_coexist_pu.PHASE_SHIFT_STEP:
            self._pre_shift_collisions += int(collision)
            self._pre_shift_throughput += throughput
        else:
            self._post_shift_collisions += int(collision)
            self._post_shift_throughput += throughput

        # 5. Update rolling occupancy history
        new_occ = [float(p) for p in pu_active]
        self._occupancy_history.pop(0)
        self._occupancy_history.append(new_occ)

        done = self._step_count >= self._max_steps

        obs = self._build_observation(
            last_reward=reward,
            last_throughput=throughput,
            last_collision=collision,
        )
        obs.done = done
        obs.reward = reward
        return obs

    # ──────────────────────────────────────────
    # state (property)
    # ──────────────────────────────────────────

    @property
    def state(self) -> SpectrumState:
        """
        Return current episode metadata and grader score.
        Called by the OpenEnv client via env.state().
        """
        grader_score = self._compute_grader_score()
        collision_rate = (
            self._collision_count / self._step_count if self._step_count > 0 else 0.0
        )
        spectral_eff = (
            self._total_throughput / self._theoretical_max
            if self._theoretical_max > 0
            else 0.0
        )

        return SpectrumState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            task_name=self._task_module.TASK_NAME,
            difficulty=self._difficulty,
            max_steps=self._max_steps,
            seed=self._default_seed,
            grader_score=grader_score,
            collision_rate=round(collision_rate, 4),
            spectral_efficiency=round(min(1.0, spectral_eff), 4),
        )

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _build_observation(
        self,
        last_reward: float,
        last_throughput: float,
        last_collision: bool,
    ) -> SpectrumObservation:
        """Flatten [T_history][N_channels] history into 1D list."""
        flat = []
        for t in range(self._t_history):
            for ch in range(self._n_channels):
                flat.append(self._occupancy_history[t][ch])

        return SpectrumObservation(
            channel_occupancy=flat,
            n_channels=self._n_channels,
            t_history=self._t_history,
            last_throughput=round(last_throughput, 4),
            last_collision=last_collision,
            last_reward=round(last_reward, 4),
            collision_count=self._collision_count,
            total_throughput=round(self._total_throughput, 4),
            step_number=self._step_count,
            task_id=self._task_id,
            task_description=self._task_module.DESCRIPTION,
            done=False,
            reward=last_reward,
        )

    def _compute_grader_score(self) -> float:
        """Dispatch to the active task's grade() function."""
        raw_score = self._task_module.grade(
            collision_count=self._collision_count,
            total_steps=max(self._step_count, 1),
            total_throughput=self._total_throughput,
            theoretical_max_throughput=self._theoretical_max,
            pre_shift_collisions=self._pre_shift_collisions,
            post_shift_collisions=self._post_shift_collisions,
            pre_shift_throughput=self._pre_shift_throughput,
            post_shift_throughput=self._post_shift_throughput,
        )
        return max(0.0001, min(0.9999, float(raw_score)))
