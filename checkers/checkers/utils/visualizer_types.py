"""
Data structures for real-time training visualization.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class MetricsUpdate:
    """Metrics update from training iteration."""
    iteration: int
    total_loss: float
    policy_loss: float
    value_loss: float
    buffer_size: int
    time_selfplay: float = 0.0
    time_training: float = 0.0
    time_eval: float = 0.0


@dataclass
class GameStateUpdate:
    """Game state update from self-play."""
    game_array: np.ndarray  # 10x10 array representing board
    policy: Optional[np.ndarray] = None  # 150-dim policy vector
    move_count: int = 0
    player: int = 1  # Current player (1 or 2)


@dataclass
class EvaluationUpdate:
    """Evaluation results update."""
    iteration: int
    win_rate: float
    wins: int
    draws: int
    losses: int
    is_best: bool = False


@dataclass
class StatusUpdate:
    """General status update."""
    message: str
    iteration: int = 0
    phase: str = ""  # "selfplay", "training", "evaluation", "checkpoint"
