# Clauding

A collection of AI/ML experiments, built iteratively with Claude Code.

## Projects

### Reinforcement Learning — CarRacing-v3

A progression of RL agents learning to drive in OpenAI Gymnasium's CarRacing-v3:

| Project | Algorithm | Notes |
|---------|-----------|-------|
| [001](001/) | DQN | Deep Q-Network baseline |
| [002](002/) | DDQN | Double DQN, improved stability |
| [003](003/) | SAC | Soft Actor-Critic, continuous actions |
| [004](004/) | SAC | 2D action space, performance fixes |
| [005](005/) | SAC | Magic Formula tire model, multi-car |
| [006](006/) | SAC | Custom 2D physics engine, domain randomization |

### Board Games — AlphaZero Style

- [go9x9](go9x9/) — AlphaZero-style 9x9 Go engine. ResNet trained via MCTS self-play. Includes GTP interface for play in Sabaki.
- [checkers](checkers/) — AlphaZero-style 8x8 Checkers engine. C++ MCTS core with Python training loop.

### Research

- [grokking](grokking/) — Grokking experiment on modular arithmetic. Transformer trained to generalize late after apparent convergence.

### Miscellaneous

- [whatever](whatever/) — Simulations, visualizations, and Metal/GPU experiments (reaction-diffusion, xenobots, physarum, etc.)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install torch numpy
```

Each project may have its own `requirements.txt`.
