"""
Main CLI entrypoint for Mario Arena.

This module provides the command-line interface for training, evaluating,
and scoring RL agents on Super Mario Bros.
"""

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with subcommands.

    Returns:
        ArgumentParser configured with all CLI subcommands
    """
    parser = argparse.ArgumentParser(
        prog="mario-arena",
        description="Mario Arena - RL training and evaluation for Super Mario Bros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a PPO agent on level 1-1
  python -m mario_arena.cli.main train --total-timesteps 100000 --level-id SuperMarioBros-1-1-v0

  # Evaluate and score a trained model
  python -m mario_arena.cli.main score --checkpoint checkpoints/mario_ppo_100000.zip --episodes 10

  # Watch a trained model play (with rendering)
  python -m mario_arena.cli.main score --checkpoint checkpoints/mario_ppo_100000.zip --render

  # Race multiple models (coming soon)
  python -m mario_arena.cli.main race --checkpoints model1.zip model2.zip model3.zip

For more information, see docs/OVERVIEW.md
        """,
    )

    # Add version
    parser.add_argument(
        "--version",
        action="version",
        version="Mario Arena v0.1.0",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,
    )

    # Train command (stub for now)
    train_parser = subparsers.add_parser(
        "train",
        help="Train a new RL agent",
        description="Train a PPO agent on Super Mario Bros",
    )
    train_parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1000000)",
    )
    train_parser.add_argument(
        "--level-id",
        type=str,
        default="SuperMarioBros-1-1-v0",
        help="Mario level to train on (default: SuperMarioBros-1-1-v0)",
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Directory for logs (default: logs/)",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/",
        help="Directory for checkpoints (default: checkpoints/)",
    )

    # Score/eval command (stub for now)
    score_parser = subparsers.add_parser(
        "score",
        help="Evaluate and score a trained model",
        description="Evaluate a trained model and compute arena score",
    )
    score_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.zip file)",
    )
    score_parser.add_argument(
        "--level-id",
        type=str,
        default="SuperMarioBros-1-1-v0",
        help="Mario level to evaluate on (default: SuperMarioBros-1-1-v0)",
    )
    score_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    score_parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes to screen",
    )

    # Race command (stub for future)
    race_parser = subparsers.add_parser(
        "race",
        help="Race multiple models head-to-head (coming soon)",
        description="Compare multiple trained models on the same evaluation protocol",
    )
    race_parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints to race",
    )
    race_parser.add_argument(
        "--level-id",
        type=str,
        default="SuperMarioBros-1-1-v0",
        help="Mario level for the race (default: SuperMarioBros-1-1-v0)",
    )
    race_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per model (default: 10)",
    )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """
    Handle the 'train' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("=== Mario Arena - Training ===")
    print(f"Level: {args.level_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Log directory: {args.log_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print()
    print("NOTE: Training implementation coming in Phase 3")
    print("The training loop will use PPO from Stable-Baselines3")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """
    Handle the 'score' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("=== Mario Arena - Evaluation & Scoring ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Level: {args.level_id}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print()
    print("NOTE: Evaluation implementation coming in Phase 4")
    print("This will run the model and compute an arena score")
    return 0


def cmd_race(args: argparse.Namespace) -> int:
    """
    Handle the 'race' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("=== Mario Arena - Multi-Model Race ===")
    print(f"Number of models: {len(args.checkpoints)}")
    print(f"Level: {args.level_id}")
    print(f"Episodes per model: {args.episodes}")
    print()
    print("NOTE: Racing implementation coming in Phase 5")
    print("This will compare multiple models head-to-head")
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entrypoint.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Route to appropriate command handler
    if args.command == "train":
        return cmd_train(args)
    elif args.command == "score":
        return cmd_score(args)
    elif args.command == "race":
        return cmd_race(args)
    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
