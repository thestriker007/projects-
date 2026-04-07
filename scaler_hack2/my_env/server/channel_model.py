"""
Channel Model: Computes SNR, path loss, and Shannon capacity.

Given a chosen channel and transmit power, returns the achievable
throughput using the Shannon–Hartley theorem:

    C = B · log₂(1 + SNR)

where SNR depends on transmit power, path loss, and noise floor.
"""

from __future__ import annotations

import math
import random


# ── Physical constants ────────────────────────────────────────────
BANDWIDTH_MHZ = 1.0          # Bandwidth per channel (MHz)
NOISE_FLOOR_DBM = -100.0     # Thermal noise floor (dBm)
PATH_LOSS_DB = 70.0          # Free-space path loss (dB), simplified fixed model
MAX_THROUGHPUT_MBPS = BANDWIDTH_MHZ * math.log2(1 + 10 ** ((30.0 - PATH_LOSS_DB - NOISE_FLOOR_DBM) / 10))


def compute_snr(tx_power_dbm: float, channel_id: int, rng: random.Random | None = None) -> float:
    """
    Compute SNR (linear scale) for a given channel and power level.

    SNR_dB = TxPower_dBm - PathLoss_dB - NoisePower_dBm + fading_dB
    SNR (linear) = 10^(SNR_dB / 10)

    Args:
        tx_power_dbm: Transmit power in dBm [0.0, 30.0]
        channel_id:   Channel index (adds small per-channel offset for variety)
        rng:          Optional RNG for stochastic fading (None = deterministic)

    Returns:
        SNR as a positive linear value
    """
    # Small per-channel path-loss variation (±3 dB) makes channels distinct
    channel_offset_db = (channel_id % 4) * 1.5 - 3.0

    # Optional Rayleigh fading: ~N(0, 3) dB standard deviation
    fading_db = 0.0
    if rng is not None:
        fading_db = rng.gauss(0.0, 2.0)

    snr_db = tx_power_dbm - PATH_LOSS_DB - NOISE_FLOOR_DBM + channel_offset_db + fading_db
    snr_linear = 10.0 ** (snr_db / 10.0)
    return max(snr_linear, 0.001)  # clamp to avoid log(0)


def compute_throughput(snr_linear: float) -> float:
    """
    Shannon–Hartley capacity: C = B · log₂(1 + SNR)

    Returns throughput in Mbps.
    """
    return BANDWIDTH_MHZ * math.log2(1.0 + snr_linear)


def compute_reward(
    chosen_channel: int,
    tx_power_dbm: float,
    pu_active: list[bool],
    snr_linear: float,
    max_power_dbm: float = 30.0,
    beta: float = 10.0,
    gamma: float = 0.1,
) -> tuple[float, float, bool]:
    """
    Canonical reward function:

        R = Throughput − β·(Collision Penalty) − γ·(Energy Cost)

    Provides dense signal at every step (not just terminal).

    Args:
        chosen_channel: Channel index the agent selected
        tx_power_dbm:   Transmit power chosen by agent
        pu_active:      List of per-channel PU activity (True = PU present)
        snr_linear:     Precomputed SNR for the chosen channel
        max_power_dbm:  System max power for normalization
        beta:           Collision penalty weight (default 10.0)
        gamma:          Energy cost weight (default 0.1)

    Returns:
        (reward, throughput_mbps, collision)
    """
    collision = pu_active[chosen_channel]
    throughput = compute_throughput(snr_linear) if not collision else 0.0
    energy_norm = tx_power_dbm / max_power_dbm  # normalized ∈ [0, 1]

    # R = Throughput − β(Collision Penalty) − γ(Energy Cost)
    reward = throughput - beta * float(collision) - gamma * energy_norm

    return float(reward), float(throughput), collision
