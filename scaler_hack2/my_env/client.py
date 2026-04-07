"""
SmartRadioEnv — OpenEnv client SDK.

Provides a Pythonic async + sync interface to the running SmartRadio server.
Compatible with both local development and the deployed HF Space.

Usage (async):
    async with SmartRadioEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset(seed=42)
        result = await env.step(SpectrumAction(channel_id=3, tx_power_dbm=20.0))

Usage (sync):
    with SmartRadioEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset(seed=42)
        result = env.step(SpectrumAction(channel_id=3, tx_power_dbm=20.0))
"""

from __future__ import annotations

from openenv.core import EnvClient

from models import SpectrumAction, SpectrumObservation, SpectrumState


class SmartRadioEnv(EnvClient[SpectrumAction, SpectrumObservation, SpectrumState]):
    """
    Client for the SmartRadio OpenEnv server.

    Inherits all async/sync connection management from EnvClient.
    Type parameters ensure actions and observations are correctly validated.
    """

    action_type = SpectrumAction
    observation_type = SpectrumObservation
    state_type = SpectrumState
