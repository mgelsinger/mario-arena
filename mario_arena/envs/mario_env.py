"""
Mario environment wrapper for reinforcement learning.

This module provides a clean wrapper around gym-super-mario-bros that:
- Applies standard preprocessing (grayscale, resize, frame stacking)
- Uses discrete action spaces suitable for RL training
- Configurable for different levels and rendering modes
- Compatible with Stable-Baselines3
"""

from dataclasses import dataclass
from typing import Literal
import gym


@dataclass
class MarioArenaEnvConfig:
    """
    Configuration for Mario Arena environment.

    Attributes:
        level_id: The gym-super-mario-bros level identifier (e.g., "SuperMarioBros-1-1-v0")
        frame_stack: Number of frames to stack for temporal information (default: 4)
        frame_shape: Tuple of (height, width) for resized frames (default: (84, 84))
        action_set: Which action set to use - "simple" or "complex" (default: "simple")
        max_episode_steps: Maximum steps per episode, None for no limit (default: None)
        skip_frames: Number of frames to skip between actions (default: 4)
    """
    level_id: str = "SuperMarioBros-1-1-v0"
    frame_stack: int = 4
    frame_shape: tuple[int, int] = (84, 84)
    action_set: Literal["simple", "complex"] = "simple"
    max_episode_steps: int | None = None
    skip_frames: int = 4


def make_mario_env(
    level: str = "SuperMarioBros-1-1-v0",
    render_mode: str | None = None,
    config: MarioArenaEnvConfig | None = None,
) -> gym.Env:
    """
    Create a Mario environment wrapped with standard RL preprocessing.

    This function creates a gym-super-mario-bros environment and applies
    the following wrappers:
    1. JoypadSpace - Convert to discrete action space
    2. FrameSkip - Skip frames to speed up training
    3. Grayscale - Convert RGB to grayscale
    4. Resize - Resize frames to standard size (84x84)
    5. FrameStack - Stack frames for temporal information

    Args:
        level: The level identifier (e.g., "SuperMarioBros-1-1-v0")
        render_mode: How to render the environment. Options:
            - None: No rendering (fastest, for training)
            - "human": Render to screen (for watching)
            - "rgb_array": Return frames as arrays
        config: Optional MarioArenaEnvConfig for custom settings.
                If None, uses default config with the specified level.

    Returns:
        A wrapped Gym environment ready for RL training with SB3.

    Example:
        >>> env = make_mario_env("SuperMarioBros-1-1-v0", render_mode=None)
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step(env.action_space.sample())

    Note:
        The returned environment has:
        - Observation space: Box(0, 255, (84, 84, 4), uint8) for 4 stacked grayscale frames
        - Action space: Discrete(n) where n depends on action_set
    """
    # TODO: Implement in Phase 2
    # - Import gym_super_mario_bros
    # - Apply JoypadSpace wrapper with SIMPLE_MOVEMENT or COMPLEX_MOVEMENT
    # - Apply frame skip wrapper
    # - Apply grayscale + resize wrappers
    # - Apply frame stacking wrapper
    # - Return wrapped environment
    raise NotImplementedError("Environment creation will be implemented in Phase 2")


def get_action_meanings(action_set: Literal["simple", "complex"] = "simple") -> list[str]:
    """
    Get human-readable action descriptions for the given action set.

    Args:
        action_set: Which action set to describe

    Returns:
        List of action descriptions (e.g., ["NOOP", "right", "right + A", ...])

    Example:
        >>> actions = get_action_meanings("simple")
        >>> print(f"Action 0: {actions[0]}")
        Action 0: NOOP
    """
    # TODO: Implement in Phase 2
    # Return action meanings from nes_py JoypadSpace
    raise NotImplementedError("Action meanings will be implemented in Phase 2")


def extract_info_from_env(info: dict) -> dict:
    """
    Extract and normalize useful information from the environment's info dict.

    The gym-super-mario-bros environment returns various metadata in the info dict
    after each step. This function extracts the most relevant fields for evaluation.

    Args:
        info: The info dictionary returned by env.step()

    Returns:
        Dictionary with normalized fields:
            - x_pos: Current x position in the level
            - y_pos: Current y position in the level
            - coins: Number of coins collected
            - flag_get: Whether Mario reached the flag (level complete)
            - life: Current life count
            - score: Game score
            - stage: Current stage number
            - status: Mario's status (small, tall, fireball)
            - time: Time remaining
            - world: Current world number

    Example:
        >>> _, _, _, info = env.step(action)
        >>> extracted = extract_info_from_env(info)
        >>> print(f"Mario is at x={extracted['x_pos']}")
    """
    # TODO: Implement in Phase 2
    # Extract relevant fields from info dict
    # Normalize and return
    raise NotImplementedError("Info extraction will be implemented in Phase 2")
