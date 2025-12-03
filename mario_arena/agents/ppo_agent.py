"""
PPO (Proximal Policy Optimization) agent implementation for Mario Arena.

This module provides wrappers around Stable-Baselines3's PPO implementation
optimized for training on Super Mario Bros.
"""

from pathlib import Path
from typing import Any
import gym


def make_ppo_model(
    env: gym.Env,
    log_dir: Path,
    learning_rate: float = 2.5e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,
    seed: int | None = None,
    device: str = "auto",
    tensorboard_log: Path | None = None,
    **kwargs: Any,
):
    """
    Create a PPO model configured for Super Mario Bros.

    This function creates a Stable-Baselines3 PPO model with hyperparameters
    tuned for Atari-like environments. The model uses a CNN policy to process
    stacked grayscale frames.

    Args:
        env: The Mario environment (should be vectorized for SB3)
        log_dir: Directory for saving model checkpoints
        learning_rate: Learning rate for optimizer (default: 2.5e-4)
        n_steps: Number of steps to collect per update (default: 2048)
        batch_size: Minibatch size for training (default: 64)
        n_epochs: Number of epochs for policy update (default: 10)
        gamma: Discount factor (default: 0.99)
        gae_lambda: GAE lambda parameter (default: 0.95)
        clip_range: PPO clipping parameter (default: 0.2)
        ent_coef: Entropy coefficient for exploration (default: 0.01)
        vf_coef: Value function coefficient (default: 0.5)
        max_grad_norm: Max gradient norm for clipping (default: 0.5)
        use_sde: Whether to use State-Dependent Exploration (default: False)
        seed: Random seed for reproducibility (default: None)
        device: Device to use - "auto", "cpu", or "cuda" (default: "auto")
        tensorboard_log: Path for TensorBoard logs (default: None)
        **kwargs: Additional arguments to pass to PPO constructor

    Returns:
        A configured PPO model ready for training.

    Example:
        >>> from stable_baselines3.common.vec_env import DummyVecEnv
        >>> from mario_arena.envs import make_mario_env
        >>>
        >>> env = DummyVecEnv([lambda: make_mario_env()])
        >>> model = make_ppo_model(env, log_dir=Path("logs/"))
        >>> model.learn(total_timesteps=100000)

    Note:
        Default hyperparameters are based on common settings for Atari games.
        You may need to tune them for optimal performance on specific Mario levels.
    """
    # TODO: Implement in Phase 3
    # - Import PPO from stable_baselines3
    # - Create PPO model with CnnPolicy
    # - Configure hyperparameters
    # - Set up TensorBoard logging
    # - Return configured model
    raise NotImplementedError("PPO model creation will be implemented in Phase 3")


def load_ppo_model(checkpoint_path: Path, env: gym.Env | None = None):
    """
    Load a trained PPO model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model (.zip file)
        env: Optional environment to set for the model.
                If None, uses the environment saved with the model.

    Returns:
        The loaded PPO model ready for evaluation or continued training.

    Example:
        >>> model = load_ppo_model(Path("checkpoints/mario_ppo_100000.zip"))
        >>> obs = env.reset()
        >>> action, _ = model.predict(obs, deterministic=True)

    Raises:
        FileNotFoundError: If checkpoint_path does not exist
    """
    # TODO: Implement in Phase 3
    # - Import PPO from stable_baselines3
    # - Load model from checkpoint
    # - Set environment if provided
    # - Return loaded model
    raise NotImplementedError("Model loading will be implemented in Phase 3")


def save_ppo_model(model, save_path: Path) -> None:
    """
    Save a PPO model to disk.

    Args:
        model: The PPO model to save
        save_path: Path where to save the model (.zip extension recommended)

    Example:
        >>> model.learn(total_timesteps=10000)
        >>> save_ppo_model(model, Path("checkpoints/mario_ppo_10000.zip"))
    """
    # TODO: Implement in Phase 3
    # - Create parent directory if needed
    # - Save model using SB3's save method
    raise NotImplementedError("Model saving will be implemented in Phase 3")


def get_default_hyperparameters() -> dict[str, Any]:
    """
    Get default PPO hyperparameters for Mario training.

    Returns:
        Dictionary of hyperparameter names and values.

    Example:
        >>> hyperparams = get_default_hyperparameters()
        >>> print(f"Default learning rate: {hyperparams['learning_rate']}")
    """
    return {
        "learning_rate": 2.5e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
