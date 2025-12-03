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
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import numpy as np


# Custom wrappers for preprocessing

class GymV21CompatibilityV0(gym.Wrapper):
    """
    Compatibility wrapper to convert old gym API (v0.21-) to new gym API (v0.26+).

    Old API: reset() returns obs, step() returns (obs, reward, done, info)
    New API: reset() returns (obs, info), step() returns (obs, reward, terminated, truncated, info)

    This wrapper ensures compatibility with gym 0.26 which expects the new API.
    """
    def reset(self, **kwargs):
        """Convert old reset() to new API."""
        obs = self.env.reset(**kwargs)
        return obs, {}  # Add empty info dict

    def step(self, action):
        """Convert old step() to new API."""
        obs, reward, done, info = self.env.step(action)
        # In old API, done meant either terminated or truncated
        # We'll treat it as terminated for simplicity
        return obs, reward, done, False, info


class SkipFrame(gym.Wrapper):
    """
    Wrapper to skip frames and return only every `skip`-th frame.

    This reduces the temporal resolution and speeds up training.
    The skipped frames still have actions applied and accumulate rewards.
    """
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action and sum rewards over skipped frames."""
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for _ in range(self._skip):
            result = self.env.step(action)
            if len(result) == 5:  # New API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # Old API: obs, reward, done, info
                obs, reward, done, info = result
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert RGB observations to grayscale.

    Reduces observation space from (H, W, 3) to (H, W) using luminosity formula.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # Remove color channel
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        """Convert RGB to grayscale using luminosity method."""
        # Use standard luminosity weights: R*0.299 + G*0.587 + B*0.114
        observation = np.dot(observation[...,:3], [0.299, 0.587, 0.114])
        return observation.astype(np.uint8)

    def reset(self, **kwargs):
        """Reset with compatibility for old gym API."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self.observation(obs), info
        else:
            return self.observation(result)

    def step(self, action):
        """Step with compatibility for old gym API."""
        result = self.env.step(action)
        if len(result) == 5:  # New API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return self.observation(obs), reward, done, info
        else:  # Old API: obs, reward, done, info
            obs, reward, done, info = result
            return self.observation(obs), reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    """
    Resize observations to a specified shape.

    This standardizes the observation size for neural network input.
    """
    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.shape = shape
        obs_shape = self.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        """Resize observation using OpenCV."""
        import cv2
        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        return observation

    def reset(self, **kwargs):
        """Reset with compatibility for old gym API."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self.observation(obs), info
        else:
            return self.observation(result)

    def step(self, action):
        """Step with compatibility for old gym API."""
        result = self.env.step(action)
        if len(result) == 5:  # New API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return self.observation(obs), reward, done, info
        else:  # Old API: obs, reward, done, info
            obs, reward, done, info = result
            return self.observation(obs), reward, done, info


class FrameStack(gym.Wrapper):
    """
    Stack the last N frames together.

    This provides temporal information to the agent by stacking
    multiple consecutive frames into a single observation.
    """
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = None

        # Update observation space to include stacked frames
        low = np.repeat(self.observation_space.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(self.observation_space.high[..., np.newaxis], num_stack, axis=-1)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        # Handle both old gym API (returns obs) and new API (returns obs, info)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return_info = True
        else:
            obs = result
            return_info = False

        # Stack the initial frame num_stack times
        self.frames = np.stack([obs for _ in range(self.num_stack)], axis=-1)

        if return_info:
            return self.frames, info
        else:
            return self.frames

    def step(self, action):
        """Take a step and update the frame stack."""
        result = self.env.step(action)
        if len(result) == 5:  # New API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old API: obs, reward, done, info
            obs, reward, done, info = result

        # Roll the frames and add the new observation
        self.frames = np.roll(self.frames, shift=-1, axis=-1)
        self.frames[..., -1] = obs
        return self.frames, reward, done, info


@dataclass
class MarioArenaEnvConfig:
    """
    Configuration for Mario Arena environment.

    Attributes:
        level_id: The gym-super-mario-bros level identifier (e.g., "SuperMarioBros-1-1-v0")
        frame_stack: Number of frames to stack for temporal information (default: 4)
        frame_shape: Tuple of (height, width) for resized frames (default: (84, 84))
        action_set: Which action set to use - "simple", "complex", or "right_only" (default: "simple")
        max_episode_steps: Maximum steps per episode, None for no limit (default: None)
        skip_frames: Number of frames to skip between actions (default: 4)
    """
    level_id: str = "SuperMarioBros-1-1-v0"
    frame_stack: int = 4
    frame_shape: tuple[int, int] = (84, 84)
    action_set: Literal["simple", "complex", "right_only"] = "simple"
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
    # Use default config if none provided
    if config is None:
        config = MarioArenaEnvConfig(level_id=level)
    else:
        # Override level if explicitly provided
        config.level_id = level

    # Create the base Mario environment
    # Note: gym-super-mario-bros with gym 0.26.2 doesn't support render_mode in make()
    # We'll handle rendering via env.render() calls instead
    env = gym_super_mario_bros.make(config.level_id)

    # Note: render_mode parameter is kept for API compatibility but not used with gym 0.26
    # Users should call env.render() manually if they want to see the game

    # CRITICAL: gym.make() adds a TimeLimit wrapper that expects new API (5-tuple),
    # but the base env uses old API (4-tuple). We need to unwrap to the base env,
    # apply compatibility wrapper, then re-wrap with TimeLimit.
    # For now, we'll just unwrap and skip TimeLimit since we handle episode length differently.
    if hasattr(env, 'env'):
        # Unwrap the TimeLimit wrapper
        env = env.env

    # Apply compatibility wrapper to base environment FIRST
    # This converts old gym API (4-tuple) to new gym API (5-tuple)
    env = GymV21CompatibilityV0(env)

    # 1. Apply JoypadSpace wrapper to convert to discrete action space
    # Do this BEFORE compatibility wrapper since JoypadSpace uses old API
    if config.action_set == "simple":
        action_space = SIMPLE_MOVEMENT
    elif config.action_set == "complex":
        action_space = COMPLEX_MOVEMENT
    elif config.action_set == "right_only":
        action_space = RIGHT_ONLY
    else:
        raise ValueError(f"Unknown action_set: {config.action_set}. Use 'simple', 'complex', or 'right_only'")

    env = JoypadSpace(env, action_space)

    # 2. Apply frame skipping
    if config.skip_frames > 1:
        env = SkipFrame(env, skip=config.skip_frames)

    # 3. Convert to grayscale
    env = GrayScaleObservation(env)

    # 4. Resize observations
    env = ResizeObservation(env, shape=config.frame_shape)

    # 5. Stack frames for temporal information
    if config.frame_stack > 1:
        env = FrameStack(env, num_stack=config.frame_stack)

    return env


def get_action_meanings(action_set: Literal["simple", "complex", "right_only"] = "simple") -> list[str]:
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
    # Map button combinations to readable strings
    action_map = {
        "simple": SIMPLE_MOVEMENT,
        "complex": COMPLEX_MOVEMENT,
        "right_only": RIGHT_ONLY,
    }

    if action_set not in action_map:
        raise ValueError(f"Unknown action_set: {action_set}")

    actions = action_map[action_set]

    # Actions are already lists of button names as strings
    # Just join them with " + "
    meanings = []
    for action in actions:
        if not action or action == ['NOOP']:
            meanings.append("NOOP")
        else:
            meanings.append(" + ".join(action))

    return meanings


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
    # The gym-super-mario-bros environment provides these fields in the info dict
    # Extract with defaults in case any field is missing
    extracted = {
        'x_pos': info.get('x_pos', 0),
        'y_pos': info.get('y_pos', 0),
        'coins': info.get('coins', 0),
        'flag_get': info.get('flag_get', False),
        'life': info.get('life', 2),
        'score': info.get('score', 0),
        'stage': info.get('stage', 1),
        'status': info.get('status', 'small'),
        'time': info.get('time', 400),
        'world': info.get('world', 1),
    }

    return extracted
