"""
Environment wrappers for Super Mario Bros.

This module provides clean wrappers around gym-super-mario-bros
optimized for reinforcement learning training.
"""

from .mario_env import (
    make_mario_env,
    MarioArenaEnvConfig,
    get_action_meanings,
    extract_info_from_env,
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
)

__all__ = [
    "make_mario_env",
    "MarioArenaEnvConfig",
    "get_action_meanings",
    "extract_info_from_env",
    "SkipFrame",
    "GrayScaleObservation",
    "ResizeObservation",
    "FrameStack",
]
