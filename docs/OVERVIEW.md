# Mario Arena - Technical Overview

## Introduction

Mario Arena is a reinforcement learning platform built on top of `gym-super-mario-bros` that provides:

1. A standardized environment wrapper optimized for RL training
2. PPO (Proximal Policy Optimization) agent implementation
3. Comprehensive evaluation and scoring system
4. CLI interface for training and evaluation
5. Future support for multi-agent racing

This document provides a technical overview of the architecture and design decisions.

## Architecture

### Project Structure

```
mario-arena/
├── mario_arena/              # Main Python package
│   ├── envs/                # Environment wrappers
│   │   └── mario_env.py     # Mario environment factory and preprocessing
│   ├── agents/              # RL agent implementations
│   │   └── ppo_agent.py     # PPO agent wrapper (Stable-Baselines3)
│   ├── training/            # Training and evaluation logic
│   │   ├── train_ppo.py     # Training entrypoint
│   │   ├── eval_policy.py   # Evaluation protocols
│   │   ├── scoring.py       # Arena scoring system
│   │   └── run_protocols.py # Full-game run protocols (Phase 5)
│   ├── cli/                 # Command-line interface
│   │   └── main.py          # CLI entrypoint
│   └── utils/               # Utilities
│       └── paths.py         # Path management helpers
├── docs/                    # Documentation
│   ├── OVERVIEW.md          # This file
│   └── SCORING.md           # Scoring system details
├── requirements.txt         # Python dependencies
└── README.md               # User-facing documentation
```

## Environment Wrapper (Phase 2)

### Design Goals

The environment wrapper (`mario_arena/envs/mario_env.py`) provides a clean interface to `gym-super-mario-bros` with standard RL preprocessing:

1. **Frame preprocessing**: Convert to grayscale, resize to 84×84
2. **Frame stacking**: Stack 4 frames for temporal information
3. **Frame skipping**: Skip frames to speed up training (default: 4)
4. **Action space**: Discretized action space (SIMPLE_MOVEMENT or COMPLEX_MOVEMENT)
5. **Configurability**: Easy to adjust settings via `MarioArenaEnvConfig`

### Example Usage

Once implemented in Phase 2, you'll be able to use the environment like this:

```python
from mario_arena.envs import make_mario_env

# Create environment with default settings
env = make_mario_env("SuperMarioBros-1-1-v0", render_mode=None)

# Reset and take random steps
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

### Configuration

```python
from mario_arena.envs import make_mario_env, MarioArenaEnvConfig

config = MarioArenaEnvConfig(
    level_id="SuperMarioBros-1-1-v0",
    frame_stack=4,
    frame_shape=(84, 84),
    action_set="simple",
    max_episode_steps=None,
    skip_frames=4,
)

env = make_mario_env(config=config, render_mode="human")
```

## PPO Agent (Phase 3)

### Implementation

The PPO agent wrapper (`mario_arena/agents/ppo_agent.py`) uses Stable-Baselines3's PPO implementation with a CNN policy.

### Hyperparameters

Default hyperparameters are tuned for Atari-like environments:

- **Learning rate**: 2.5e-4
- **Batch size**: 64
- **N steps**: 2048
- **N epochs**: 10
- **Gamma**: 0.99
- **GAE lambda**: 0.95
- **Clip range**: 0.2
- **Entropy coefficient**: 0.01

These can be adjusted via command-line arguments or programmatically.

## Training Pipeline (Phase 3)

### Training Flow

1. Parse command-line arguments
2. Set random seeds (Python, NumPy, PyTorch)
3. Create vectorized training environment (multiple parallel envs)
4. Create evaluation environment (for periodic evaluation)
5. Create PPO model with CNN policy
6. Set up callbacks:
   - Checkpoint callback (save models periodically)
   - Evaluation callback (evaluate during training)
   - TensorBoard logging
7. Train with `model.learn()`
8. Save final model

### Checkpointing

Models are saved periodically during training:
- Every N timesteps (configurable via `--save-freq`)
- Final model at the end of training

Checkpoints are saved as `.zip` files compatible with SB3's save/load system.

## Evaluation System (Phase 4)

### Evaluation Protocol

The evaluation system runs a trained model on a level for multiple episodes and collects:

- Success rate (% of episodes that completed the level)
- Average steps taken
- Average maximum x-position reached
- Coins collected
- Game score
- Time remaining (if completed)

### Episode Results

Each episode produces an `EpisodeResult` containing:
- `success`: Whether the level was completed
- `total_steps`: Steps taken
- `max_x_pos`: Furthest x-position reached
- `coins_collected`, `score`, `time_remaining`: Game stats
- `world`, `stage`: Level identifiers

### Aggregated Statistics

Multiple episodes are aggregated into `EvaluationStats`:
- Success rate across all episodes
- Average performance metrics
- Best episode performance

## Scoring System (Phase 4)

See [SCORING.md](SCORING.md) for detailed scoring formulas.

The arena score prioritizes:
1. **Completion** (1M points if completed)
2. **Progress** (world, stage, x-position)
3. **Efficiency** (penalty for excessive steps)

This ensures that completing a level is always better than not completing it, and getting further is always better than not getting as far.

## CLI Interface

### Commands

The CLI provides three main commands:

1. **train**: Train a new PPO agent
   ```bash
   python -m mario_arena.cli.main train \
       --total-timesteps 1000000 \
       --level-id SuperMarioBros-1-1-v0 \
       --log-dir logs/ \
       --checkpoint-dir checkpoints/
   ```

2. **score**: Evaluate and score a trained model
   ```bash
   python -m mario_arena.cli.main score \
       --checkpoint checkpoints/mario_ppo_100000.zip \
       --level-id SuperMarioBros-1-1-v0 \
       --episodes 10 \
       --render
   ```

3. **race**: Compare multiple models (Phase 5)
   ```bash
   python -m mario_arena.cli.main race \
       --checkpoints model1.zip model2.zip model3.zip \
       --level-id SuperMarioBros-1-1-v0
   ```

## Future Features (Phase 5)

### Full-Game Runs

Instead of evaluating on a single level, we'll support:
- Sequential level progression (1-1 → 1-2 → ... → 8-4)
- Life management (track deaths and game overs)
- Full-game completion tracking
- Warp zone handling

### Multi-Agent Racing

Compare multiple models head-to-head:
- Run the same evaluation protocol for each model
- Generate a leaderboard sorted by arena score
- Support different "tracks" (level sequences)
- Visualize comparative performance

## Design Principles

### Modularity

Each component is designed to be independently useful:
- Environment wrapper can be used standalone
- Agent implementation can work with any compatible environment
- Evaluation and scoring can work with any agent
- CLI is a thin wrapper around library functions

### Extensibility

The architecture supports future additions:
- New RL algorithms (DQN, A2C, etc.)
- Different reward shaping strategies
- Custom evaluation protocols
- Additional scoring systems

### Reproducibility

All operations support seeding:
- Environment seeding
- Agent seeding (PyTorch, NumPy)
- Evaluation seeding

This ensures that training and evaluation runs are reproducible.

## Dependencies

### Core Dependencies

- `gym-super-mario-bros`: Base Mario environment
- `nes-py`: NES emulator
- `gym==0.26.2`: Gymnasium interface
- `stable-baselines3`: PPO implementation
- `torch`: Neural network backend
- `tensorboard`: Training visualization

### Compatibility

- Python 3.11+
- Compatible with CUDA for GPU acceleration
- Cross-platform (Windows, Linux, macOS)

## Performance Considerations

### Training Speed

- Use vectorized environments (multiple parallel envs)
- Frame skipping reduces computation
- GPU acceleration via PyTorch
- Adjust `n_envs` based on CPU cores

### Memory Usage

- Frame stacking increases memory per step
- Vectorized envs multiply memory usage
- Consider reducing `n_envs` if memory-constrained

## Development Roadmap

- [x] Phase 1: Project skeleton ✓
- [ ] Phase 2: Environment wrapper
- [ ] Phase 3: PPO training
- [ ] Phase 4: Evaluation and scoring
- [ ] Phase 5: Full-game runs and racing
- [ ] Future: Web UI, leaderboards, additional algorithms

## References

- [gym-super-mario-bros documentation](https://github.com/Kautenja/gym-super-mario-bros)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
