# Mario Arena

A reinforcement learning training and evaluation platform for NES Super Mario Bros.

## Overview

Mario Arena provides a standardized framework for training, evaluating, and comparing RL agents on Super Mario Bros. The project wraps `gym-super-mario-bros` with a comprehensive arena layer that:

- **Trains** RL agents using PPO (Proximal Policy Optimization) via Stable-Baselines3
- **Evaluates** agents on standardized full-game protocols
- **Scores** performance with a unified arena scoring system for fair comparison
- **Renders** trained models so you can watch them play
- **Races** multiple agents head-to-head (planned feature)

## Features

- Clean wrapper around `gym-super-mario-bros` optimized for RL training
- PPO agent implementation with CNN policy
- Comprehensive evaluation and scoring system
- CLI-first interface for training, evaluation, and scoring
- Designed as a library for future UI integration
- Reproducible training with proper seeding
- TensorBoard logging for training metrics

## Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
cd C:\proj\mario-arena
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install PyTorch (choose the appropriate version for your system):
```bash
# For CPU-only:
pip install torch torchvision

# For CUDA (check pytorch.org for your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Train an agent

```bash
python -m mario_arena.cli.main train \
    --total-timesteps 100000 \
    --level-id SuperMarioBros-1-1-v0 \
    --log-dir logs/ \
    --checkpoint-dir checkpoints/
```

### Evaluate a trained model

```bash
python -m mario_arena.cli.main score \
    --checkpoint checkpoints/mario_ppo_100000.zip \
    --level-id SuperMarioBros-1-1-v0 \
    --episodes 10 \
    --render
```

### Get help

```bash
python -m mario_arena.cli.main --help
```

## Project Structure

```
mario-arena/
├── mario_arena/
│   ├── envs/          # Mario environment wrappers
│   ├── agents/        # RL agent implementations (PPO)
│   ├── training/      # Training, evaluation, and scoring logic
│   ├── cli/           # Command-line interface
│   └── utils/         # Helper utilities
├── docs/              # Documentation
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Documentation

- [OVERVIEW.md](docs/OVERVIEW.md) - Technical overview and architecture
- [SCORING.md](docs/SCORING.md) - Arena scoring system explained

## Scoring System

Mario Arena uses a comprehensive scoring system that prioritizes:

1. **Game completion** - Did the agent beat the level?
2. **Progress** - How far did it get (world, stage, x-position)?
3. **Efficiency** - How many steps did it take?

See [SCORING.md](docs/SCORING.md) for detailed scoring formulas.

## Roadmap

- [x] Phase 1: Project skeleton and packaging
- [ ] Phase 2: Mario environment wrapper
- [ ] Phase 3: PPO training implementation
- [ ] Phase 4: Evaluation and scoring system
- [ ] Phase 5: Full-game runs and multi-agent racing

## Requirements

- Python 3.11+
- gym-super-mario-bros
- nes-py
- gym 0.26.2
- stable-baselines3
- torch
- tensorboard

## Contributing

This project is designed to be extensible. Future additions may include:

- Additional RL algorithms (DQN, A2C, etc.)
- More sophisticated reward shaping
- Web UI for training visualization
- Leaderboard system
- Multi-agent competitive racing

## License

MIT (or specify your license)

## Acknowledgments

- OpenAI Gym and gym-super-mario-bros for the environment
- Stable-Baselines3 for the RL implementations
- The RL community for research and best practices
