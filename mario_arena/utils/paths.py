"""
Path management utilities for Mario Arena.

This module provides helper functions for managing directories and file paths
for logs, checkpoints, and other training artifacts.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal


def get_project_root() -> Path:
    """
    Get the root directory of the mario-arena project.

    Returns:
        Path to the project root directory

    Example:
        >>> root = get_project_root()
        >>> print(root)
        C:\proj\mario-arena
    """
    # For now, assume we're in the mario_arena package
    # This returns the parent of the mario_arena package directory
    return Path(__file__).parent.parent.parent


def get_log_dir(
    base_dir: Path | str = "logs",
    experiment_name: str | None = None,
    create: bool = True,
) -> Path:
    """
    Get or create a directory for training logs.

    Args:
        base_dir: Base directory for logs (default: "logs")
        experiment_name: Optional name for this experiment.
                        If None, uses timestamp.
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to the log directory

    Example:
        >>> log_dir = get_log_dir(experiment_name="ppo_1-1_v1")
        >>> print(log_dir)
        C:\proj\mario-arena\logs\ppo_1-1_v1
    """
    # TODO: Implement in Phase 1 (simple version)
    # - Convert base_dir to Path
    # - If experiment_name is None, generate timestamped name
    # - Combine base_dir / experiment_name
    # - Create directory if create=True
    # - Return path
    raise NotImplementedError("Log directory management will be implemented in Phase 1")


def get_checkpoint_dir(
    base_dir: Path | str = "checkpoints",
    experiment_name: str | None = None,
    create: bool = True,
) -> Path:
    """
    Get or create a directory for model checkpoints.

    Args:
        base_dir: Base directory for checkpoints (default: "checkpoints")
        experiment_name: Optional name for this experiment.
                        If None, uses timestamp.
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path to the checkpoint directory

    Example:
        >>> checkpoint_dir = get_checkpoint_dir(experiment_name="ppo_1-1_v1")
        >>> print(checkpoint_dir)
        C:\proj\mario-arena\checkpoints\ppo_1-1_v1
    """
    # TODO: Implement in Phase 1 (simple version)
    # Similar to get_log_dir but for checkpoints
    raise NotImplementedError("Checkpoint directory management will be implemented in Phase 1")


def get_timestamped_name(prefix: str = "", suffix: str = "") -> str:
    """
    Generate a timestamped name for experiments.

    Args:
        prefix: Optional prefix before timestamp
        suffix: Optional suffix after timestamp

    Returns:
        String in format: [prefix_]YYYYMMDD_HHMMSS[_suffix]

    Example:
        >>> name = get_timestamped_name(prefix="ppo", suffix="1-1")
        >>> print(name)
        ppo_20250102_143052_1-1
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [p for p in [prefix, timestamp, suffix] if p]
    return "_".join(parts)


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Example:
        >>> ensure_dir("logs/experiment_1")
        PosixPath('logs/experiment_1')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_checkpoint(checkpoint_dir: Path | str, pattern: str = "*.zip") -> Path | None:
    """
    Find the most recent checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern for checkpoint files (default: "*.zip")

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found

    Example:
        >>> latest = get_latest_checkpoint("checkpoints/ppo_1-1_v1")
        >>> if latest:
        ...     print(f"Latest checkpoint: {latest.name}")
    """
    # TODO: Implement in Phase 3
    # - List all files matching pattern
    # - Sort by modification time
    # - Return most recent, or None if empty
    raise NotImplementedError("Latest checkpoint finding will be implemented in Phase 3")


def get_checkpoint_path(
    checkpoint_dir: Path | str,
    timestep: int,
    prefix: str = "mario_ppo",
) -> Path:
    """
    Generate a standardized checkpoint filename.

    Args:
        checkpoint_dir: Directory for the checkpoint
        timestep: Training timestep for this checkpoint
        prefix: Prefix for the filename (default: "mario_ppo")

    Returns:
        Full path for the checkpoint file

    Example:
        >>> path = get_checkpoint_path("checkpoints/", 100000)
        >>> print(path.name)
        mario_ppo_100000.zip
    """
    checkpoint_dir = Path(checkpoint_dir)
    filename = f"{prefix}_{timestep}.zip"
    return checkpoint_dir / filename


def list_checkpoints(checkpoint_dir: Path | str) -> list[Path]:
    """
    List all checkpoint files in a directory, sorted by timestep.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        List of checkpoint paths, sorted by timestep (ascending)

    Example:
        >>> checkpoints = list_checkpoints("checkpoints/ppo_1-1_v1")
        >>> for cp in checkpoints:
        ...     print(cp.name)
        mario_ppo_10000.zip
        mario_ppo_20000.zip
        mario_ppo_30000.zip
    """
    # TODO: Implement in Phase 3
    # - Find all .zip files
    # - Parse timesteps from filenames
    # - Sort by timestep
    # - Return sorted list
    raise NotImplementedError("Checkpoint listing will be implemented in Phase 3")
