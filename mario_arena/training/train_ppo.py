"""
PPO training entrypoint for Mario Arena.

This module provides the main training loop for PPO agents on Super Mario Bros.
It handles environment setup, model creation, training, and checkpointing.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for PPO training.

    Returns:
        Parsed arguments namespace with the following attributes:
            - total_timesteps: Total training timesteps
            - level_id: Mario level to train on
            - log_dir: Directory for TensorBoard logs
            - checkpoint_dir: Directory for model checkpoints
            - seed: Random seed
            - n_envs: Number of parallel environments
            - save_freq: Save checkpoint every N steps
            - eval_freq: Evaluate model every N steps
            - eval_episodes: Number of episodes for evaluation
            - render: Whether to render during training (slow)
            - device: Device to use (auto/cpu/cuda)
    """
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Super Mario Bros",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of timesteps to train",
    )
    parser.add_argument(
        "--level-id",
        type=str,
        default="SuperMarioBros-1-1-v0",
        help="Mario level to train on",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100_000,
        help="Save checkpoint every N timesteps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate model every N timesteps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training (very slow, for debugging only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )

    return parser.parse_args()


def setup_training_dirs(log_dir: Path, checkpoint_dir: Path) -> tuple[Path, Path]:
    """
    Create and prepare directories for training outputs.

    Args:
        log_dir: Directory for TensorBoard logs
        checkpoint_dir: Directory for model checkpoints

    Returns:
        Tuple of (log_dir, checkpoint_dir) as Path objects
    """
    # TODO: Implement in Phase 3
    # - Create directories if they don't exist
    # - Set up timestamped subdirectories for this run
    # - Return prepared paths
    raise NotImplementedError("Directory setup will be implemented in Phase 3")


def create_training_callback(
    checkpoint_dir: Path,
    save_freq: int,
    eval_env,
    eval_freq: int,
    eval_episodes: int,
):
    """
    Create callbacks for training (checkpointing, evaluation, etc.).

    Args:
        checkpoint_dir: Where to save model checkpoints
        save_freq: How often to save (in timesteps)
        eval_env: Environment for evaluation
        eval_freq: How often to evaluate (in timesteps)
        eval_episodes: Number of episodes per evaluation

    Returns:
        A callback or list of callbacks for PPO.learn()

    Example:
        >>> callback = create_training_callback(
        ...     checkpoint_dir=Path("checkpoints/"),
        ...     save_freq=10000,
        ...     eval_env=eval_env,
        ...     eval_freq=5000,
        ...     eval_episodes=5,
        ... )
    """
    # TODO: Implement in Phase 3
    # - Create CheckpointCallback for saving models
    # - Create EvalCallback for periodic evaluation
    # - Create TensorBoard callback if needed
    # - Return callback list
    raise NotImplementedError("Callback creation will be implemented in Phase 3")


def main() -> None:
    """
    Main training entrypoint.

    This function:
    1. Parses command-line arguments
    2. Sets up directories and logging
    3. Creates training and evaluation environments
    4. Creates PPO model
    5. Runs training loop with callbacks
    6. Saves final model

    Example:
        Run from command line:
        $ python -m mario_arena.training.train_ppo --total-timesteps 100000

        Or call from Python:
        >>> from mario_arena.training.train_ppo import main
        >>> main()  # Uses command-line args
    """
    # TODO: Implement in Phase 3
    # 1. Parse arguments
    # 2. Set random seeds (Python, NumPy, PyTorch)
    # 3. Create log and checkpoint directories
    # 4. Create vectorized training environment (DummyVecEnv or SubprocVecEnv)
    # 5. Create evaluation environment
    # 6. Create PPO model with make_ppo_model()
    # 7. Create callbacks (checkpoint, eval, tensorboard)
    # 8. Train model with model.learn()
    # 9. Save final model
    # 10. Close environments
    raise NotImplementedError("Training main loop will be implemented in Phase 3")


if __name__ == "__main__":
    main()
