"""
Primary User (PU) Simulator using 2-state Markov Chains.

Each RF channel has an independent PU governed by a Hidden Markov Model:
  - State 0 = OFF (channel free for secondary user)
  - State 1 = ON  (primary user actively transmitting)

Difficulty presets control the ON/OFF transition probabilities:
  - Easy:   Low PU activity — agent has lots of free spectrum
  - Medium: Balanced activity — requires smart channel selection
  - Hard:   Bursty correlated PU — requires online adaptation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# Transition probability presets per difficulty
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {
        "p_on": 0.05,        # probability OFF→ON (PU appears)
        "p_stay_on": 0.70,   # probability ON→ON  (PU stays)
        "correlation": 0.0,  # inter-channel correlation (0=independent)
    },
    "medium": {
        "p_on": 0.20,
        "p_stay_on": 0.80,
        "correlation": 0.0,
    },
    "hard": {
        "p_on": 0.40,
        "p_stay_on": 0.90,
        "correlation": 0.35,  # adjacent channels tend to be occupied together
    },
}


@dataclass
class PrimaryUserSimulator:
    """
    Simulates multiple Primary Users across N RF channels.

    Each channel independently follows a 2-state Markov Chain:

        p_stay_on
    ON ◄──────────── ON
    │                 ▲
    │ (1-p_stay_on)   │ p_on
    ▼                 │
    OFF ─────────────► OFF
         p_stay_off (= 1 - p_on)

    Usage:
        pu = PrimaryUserSimulator(n_channels=8, difficulty="medium", seed=42)
        pu.reset()
        active: list[bool] = pu.step()  # True = PU active on that channel
    """

    n_channels: int
    difficulty: str = "medium"
    seed: Optional[int] = None

    # Internal state (set by reset)
    _states: list[int] = field(default_factory=list, init=False)
    _rng: random.Random = field(default_factory=random.Random, init=False)
    _step_count: int = field(default=0, init=False)
    _p_on: float = field(default=0.20, init=False)
    _p_stay_on: float = field(default=0.80, init=False)
    _correlation: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        preset = DIFFICULTY_PRESETS.get(self.difficulty, DIFFICULTY_PRESETS["medium"])
        self._p_on = preset["p_on"]
        self._p_stay_on = preset["p_stay_on"]
        self._correlation = preset["correlation"]
        self._rng = random.Random(self.seed)
        self._states = [0] * self.n_channels

    def reset(self, seed: Optional[int] = None) -> list[bool]:
        """Reset all PU states to OFF. Returns initial occupancy."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._step_count = 0
        # Start with a warm-up distribution matching steady-state
        steady_state_on = self._p_on / (self._p_on + (1.0 - self._p_stay_on))
        self._states = [
            1 if self._rng.random() < steady_state_on else 0
            for _ in range(self.n_channels)
        ]
        return self._get_active()

    def step(self) -> list[bool]:
        """
        Advance all PU Markov chains by one time step.
        Returns list of booleans: True = PU active on that channel.
        """
        self._step_count += 1

        # Hard mode: mid-episode pattern shift at step 150
        if self.difficulty == "hard" and self._step_count == 150:
            # Swap which channels are "hot" — tests agent adaptation
            self._states = [1 - s for s in self._states]

        new_states: list[int] = []
        for ch in range(self.n_channels):
            current = self._states[ch]

            if current == 1:  # ON → stays ON with p_stay_on
                new_state = 1 if self._rng.random() < self._p_stay_on else 0
            else:             # OFF → turns ON with p_on
                # Correlated: check neighbors for hard mode
                neighbor_pressure = self._get_neighbor_pressure(ch)
                effective_p_on = self._p_on + self._correlation * neighbor_pressure
                effective_p_on = min(effective_p_on, 0.95)
                new_state = 1 if self._rng.random() < effective_p_on else 0

            new_states.append(new_state)

        self._states = new_states
        return self._get_active()

    def get_occupancy_snapshot(self) -> list[float]:
        """Return current occupancy as float [0.0 or 1.0] per channel."""
        return [float(s) for s in self._states]

    def _get_active(self) -> list[bool]:
        return [bool(s) for s in self._states]

    def _get_neighbor_pressure(self, channel: int) -> float:
        """Inter-channel correlation: adjacent PUs increase local PU probability."""
        if self._correlation == 0.0:
            return 0.0
        neighbors = []
        if channel > 0:
            neighbors.append(self._states[channel - 1])
        if channel < self.n_channels - 1:
            neighbors.append(self._states[channel + 1])
        if not neighbors:
            return 0.0
        return sum(neighbors) / len(neighbors)
