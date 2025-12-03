"""
Policy evaluation for Mario Arena.

This module provides functions for evaluating trained RL agents on Super Mario Bros
using standardized protocols. It tracks performance metrics and can render episodes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EpisodeResult:
    """
    Results from a single episode evaluation.

    Attributes:
        success: Whether the agent completed the level (reached flag)
        total_steps: Total steps taken in the episode
        max_x_pos: Maximum x-position reached
        final_x_pos: Final x-position when episode ended
        coins_collected: Number of coins collected
        score: Game score achieved
        time_remaining: Time remaining (if completed)
        died: Whether Mario died (vs timeout)
        world: World number
        stage: Stage number
    """
    success: bool
    total_steps: int
    max_x_pos: float
    final_x_pos: float
    coins_collected: int
    score: int
    time_remaining: int
    died: bool
    world: int
    stage: int


@dataclass
class EvaluationStats:
    """
    Aggregated statistics from multiple episode evaluations.

    Attributes:
        num_episodes: Total episodes evaluated
        success_rate: Fraction of episodes that completed the level
        avg_steps: Average steps per episode
        avg_max_x: Average maximum x-position reached
        best_episode: The best performing episode result
        all_episodes: List of all episode results
    """
    num_episodes: int
    success_rate: float
    avg_steps: float
    avg_max_x: float
    best_episode: EpisodeResult
    all_episodes: list[EpisodeResult]


def run_single_episode(
    model,
    env,
    render: bool = False,
    deterministic: bool = True,
) -> EpisodeResult:
    """
    Run a single episode with the given model and environment.

    Args:
        model: The trained RL model (e.g., PPO)
        env: The Mario environment
        render: Whether to render the episode
        deterministic: Whether to use deterministic actions (default: True for eval)

    Returns:
        EpisodeResult containing all metrics from the episode

    Example:
        >>> from mario_arena.agents.ppo_agent import load_ppo_model
        >>> from mario_arena.envs import make_mario_env
        >>>
        >>> model = load_ppo_model(Path("checkpoints/model.zip"))
        >>> env = make_mario_env("SuperMarioBros-1-1-v0", render_mode="human")
        >>> result = run_single_episode(model, env, render=True)
        >>> print(f"Success: {result.success}, Max X: {result.max_x_pos}")
    """
    # TODO: Implement in Phase 4
    # 1. Reset environment
    # 2. Loop until episode ends:
    #    - Get action from model
    #    - Step environment
    #    - Track max x-position, steps, etc.
    #    - Render if requested
    # 3. Extract final stats from info
    # 4. Return EpisodeResult
    raise NotImplementedError("Single episode evaluation will be implemented in Phase 4")


def run_full_level_eval(
    model_path: Path,
    level_id: str,
    episodes: int = 10,
    render: bool = False,
    seed: int | None = None,
) -> EvaluationStats:
    """
    Evaluate a model on a specific level over multiple episodes.

    Args:
        model_path: Path to the saved model checkpoint
        level_id: The level to evaluate on (e.g., "SuperMarioBros-1-1-v0")
        episodes: Number of episodes to run (default: 10)
        render: Whether to render episodes (default: False)
        seed: Random seed for reproducibility (default: None)

    Returns:
        EvaluationStats with aggregated results across all episodes

    Example:
        >>> stats = run_full_level_eval(
        ...     model_path=Path("checkpoints/mario_ppo.zip"),
        ...     level_id="SuperMarioBros-1-1-v0",
        ...     episodes=10,
        ...     render=False,
        ... )
        >>> print(f"Success rate: {stats.success_rate:.1%}")
        >>> print(f"Average steps: {stats.avg_steps:.0f}")
    """
    # TODO: Implement in Phase 4
    # 1. Load model from checkpoint
    # 2. Create environment
    # 3. Set seed if provided
    # 4. Run multiple episodes
    # 5. Aggregate statistics
    # 6. Return EvaluationStats
    raise NotImplementedError("Full level evaluation will be implemented in Phase 4")


def evaluate_and_print_results(
    model_path: Path,
    level_id: str,
    episodes: int = 10,
    render: bool = False,
) -> EvaluationStats:
    """
    Evaluate a model and print formatted results to console.

    This is a convenience function that runs evaluation and prints
    a nice summary of the results.

    Args:
        model_path: Path to the saved model checkpoint
        level_id: The level to evaluate on
        episodes: Number of episodes to run
        render: Whether to render episodes

    Returns:
        EvaluationStats with full results

    Example:
        >>> evaluate_and_print_results(
        ...     model_path=Path("checkpoints/mario_ppo.zip"),
        ...     level_id="SuperMarioBros-1-1-v0",
        ...     episodes=10,
        ... )
        Evaluating on SuperMarioBros-1-1-v0...
        ========================================
        Episodes: 10
        Success Rate: 70.0%
        Avg Steps: 342.5
        Avg Max X: 2456.3
        Best Episode: 3012 steps, x=3266
    """
    # TODO: Implement in Phase 4
    # 1. Run evaluation
    # 2. Print formatted results
    # 3. Return stats
    raise NotImplementedError("Formatted evaluation will be implemented in Phase 4")
