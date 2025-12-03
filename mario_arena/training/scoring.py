"""
Arena scoring system for Mario Arena.

This module implements the standardized scoring system used to rank
and compare different RL agents on Super Mario Bros. The scoring prioritizes:
1. Level/game completion
2. Progress (world, stage, x-position)
3. Efficiency (fewer steps is better)
"""

from dataclasses import dataclass


@dataclass
class ArenaScore:
    """
    Complete arena score breakdown.

    Attributes:
        total_score: Final computed arena score
        completion_bonus: Points awarded for completing level
        progress_score: Points for how far the agent got
        efficiency_penalty: Points deducted for taking too many steps
        breakdown: Dictionary with detailed scoring components
    """
    total_score: float
    completion_bonus: float
    progress_score: float
    efficiency_penalty: float
    breakdown: dict[str, float]


def parse_level_id(level_id: str) -> tuple[int, int]:
    """
    Parse world and stage numbers from a level ID.

    Args:
        level_id: Level identifier (e.g., "SuperMarioBros-1-1-v0")

    Returns:
        Tuple of (world, stage) as integers

    Example:
        >>> world, stage = parse_level_id("SuperMarioBros-1-1-v0")
        >>> print(f"World {world}, Stage {stage}")
        World 1, Stage 1

        >>> world, stage = parse_level_id("SuperMarioBros-3-4-v0")
        >>> print(f"World {world}, Stage {stage}")
        World 3, Stage 4

    Raises:
        ValueError: If level_id format is invalid
    """
    # TODO: Implement in Phase 4
    # Parse the level_id string to extract world and stage numbers
    # Handle different formats if needed
    raise NotImplementedError("Level ID parsing will be implemented in Phase 4")


def compute_arena_score(
    completed: bool,
    world: int,
    stage: int,
    max_x: float,
    total_steps: int,
    coins: int = 0,
    time_remaining: int = 0,
) -> ArenaScore:
    """
    Compute the arena score for a single episode.

    The scoring formula prioritizes:
    1. Completion: 1,000,000 points if level completed
    2. Progress:
       - World reached: 10,000 points per world
       - Stage reached: 1,000 points per stage
       - X-position: 1 point per unit of max x reached
    3. Efficiency: -0.1 points per step taken
    4. Bonuses:
       - Coins: 100 points per coin (minor bonus)
       - Time: 10 points per second remaining (if completed)

    Args:
        completed: Whether the agent reached the flag
        world: World number (1-8)
        stage: Stage number (1-4)
        max_x: Maximum x-position reached in the level
        total_steps: Total steps taken
        coins: Number of coins collected (default: 0)
        time_remaining: Time remaining if completed (default: 0)

    Returns:
        ArenaScore with total score and breakdown

    Example:
        >>> score = compute_arena_score(
        ...     completed=True,
        ...     world=1,
        ...     stage=1,
        ...     max_x=3266.0,
        ...     total_steps=342,
        ...     coins=15,
        ...     time_remaining=245,
        ... )
        >>> print(f"Total score: {score.total_score:.0f}")
        Total score: 1016631

    Note:
        This scoring system is designed so that:
        - Completing a level is always better than not completing it
        - Getting further is always better than not getting as far
        - Taking fewer steps is better (but weighted much less than progress)
    """
    # TODO: Implement in Phase 4
    # 1. Calculate completion bonus (1M if completed, 0 otherwise)
    # 2. Calculate progress score (world * 10k + stage * 1k + max_x)
    # 3. Calculate efficiency penalty (steps * 0.1)
    # 4. Calculate minor bonuses (coins, time)
    # 5. Sum up total score
    # 6. Return ArenaScore with breakdown
    raise NotImplementedError("Score computation will be implemented in Phase 4")


def aggregate_scores(episode_results: list) -> ArenaScore:
    """
    Aggregate scores from multiple episodes.

    Takes the best episode score as the representative score, but also
    provides average statistics in the breakdown.

    Args:
        episode_results: List of EpisodeResult objects from evaluation

    Returns:
        ArenaScore representing the best performance across episodes,
        with additional average statistics in the breakdown

    Example:
        >>> from mario_arena.training.eval_policy import run_full_level_eval
        >>> stats = run_full_level_eval(model_path, level_id, episodes=10)
        >>> score = aggregate_scores(stats.all_episodes)
        >>> print(f"Best score: {score.total_score:.0f}")
    """
    # TODO: Implement in Phase 4
    # 1. Compute score for each episode
    # 2. Find best score
    # 3. Compute average statistics
    # 4. Return ArenaScore with best score and averages in breakdown
    raise NotImplementedError("Score aggregation will be implemented in Phase 4")


def format_score_display(score: ArenaScore) -> str:
    """
    Format an ArenaScore for human-readable display.

    Args:
        score: The ArenaScore to format

    Returns:
        Formatted string suitable for printing

    Example:
        >>> score = compute_arena_score(True, 1, 1, 3266, 342, 15, 245)
        >>> print(format_score_display(score))
        ========================================
        ARENA SCORE: 1,016,631
        ========================================
        Completion Bonus:    1,000,000
        Progress Score:         11,266
        Efficiency Penalty:        -34
        Coin Bonus:              1,500
        Time Bonus:              2,450
        ========================================
    """
    # TODO: Implement in Phase 4
    # Create a nicely formatted string showing the score breakdown
    raise NotImplementedError("Score display formatting will be implemented in Phase 4")
