"""Environment configuration dataclass for RL training.

This module defines the EnvConfig class which bundles all parameters
needed to configure a Toribash environment. It includes:
    - Physics parameters (steps per turn, spawn position)
    - Damage thresholds (for dismemberment, damage calculation)
    - Reward weights (for RL training via compute_reward)
    - Opponent behavior (for single-agent training)

Usage:
    >>> from config.env_config import EnvConfig
    >>> config = EnvConfig(
    ...     max_turns=30,
    ...     opponent_type="random",
    ...     reward_damage_dealt=2.0,
    ... )
"""

from dataclasses import dataclass, field
from .body_config import BodyConfig, DEFAULT_BODY
from .constants import (
    STEPS_PER_TURN, MAX_TURNS, DAMAGE_IMPULSE_THRESHOLD,
    DISMEMBER_IMPULSE, SPAWN_OFFSET_X,
)


@dataclass
class EnvConfig:
    """Configuration for a Toribash RL environment.
    
    This dataclass contains all tunable parameters for the game,
    including physics simulation settings, damage thresholds, and
    reward function weights for reinforcement learning.
    
    Attributes:
        body_config: Ragdoll body definition (segments and joints).
        steps_per_turn: Physics steps per game turn (default: 30).
        max_turns: Maximum turns before match ends (default: 20).
        spawn_offset_x: Horizontal offset from center for each fighter.
        damage_impulse_threshold: Minimum impulse to register damage.
        dismember_impulse: Impulse needed to dismember a limb.
        
        # Reward weights (positive = agent gets rewarded, negative = penalized)
        reward_damage_dealt: Weight for damage the agent deals.
        reward_damage_taken: Weight for damage the agent takes (negative).
        reward_ground_touch: Penalty per non-exempt segment touching ground.
        reward_opponent_ground: Bonus when opponent touches ground.
        reward_dismember: Bonus per joint dismembered on opponent.
        reward_ko: Bonus when opponent is knocked out.
        reward_win: Bonus when agent wins the match.
        
        # Opponent behavior
        opponent_type: How player 1 is controlled ("hold", "random", "mirror").
    
    Example:
        Create a config for training against random opponents:
        
        >>> config = EnvConfig(
        ...     max_turns=30,
        ...     opponent_type="random",
        ...     reward_damage_dealt=2.0,  # Double reward for damage
        ...     reward_damage_taken=-1.0,  # Stronger penalty for being hit
        ... )
    """
    # -------------------------------------------------------------------------
    # Body Configuration
    # -------------------------------------------------------------------------
    body_config: BodyConfig = field(default_factory=lambda: DEFAULT_BODY)
    
    # -------------------------------------------------------------------------
    # Game Parameters
    # -------------------------------------------------------------------------
    steps_per_turn: int = STEPS_PER_TURN
    max_turns: int = MAX_TURNS
    spawn_offset_x: float = SPAWN_OFFSET_X

    # -------------------------------------------------------------------------
    # Damage Thresholds
    # -------------------------------------------------------------------------
    damage_impulse_threshold: float = DAMAGE_IMPULSE_THRESHOLD
    dismember_impulse: float = DISMEMBER_IMPULSE
    head_damage_multiplier: float = 2.0

    # -------------------------------------------------------------------------
    # Reward Weights for RL
    # -------------------------------------------------------------------------
    # Damage rewards (small — KO is primary, damage is secondary signal)
    # Original values: dealt=1.0, taken=-0.5
    reward_damage_dealt: float = 0.1
    reward_damage_taken: float = -0.05
    
    # Ground contact rewards
    reward_ground_touch: float = -0.2     # Penalty for own segment on ground
    reward_opponent_ground: float = 0.1   # Bonus when opponent on ground
    
    # Special event rewards
    reward_dismember: float = 5.0         # Bonus per opponent joint dismembered
    reward_ko: float = 10.0               # Bonus for KO'ing opponent
    reward_ko_penalty: float = -10.0      # Penalty for being KO'd
    reward_win: float = 20.0              # Bonus for winning match
    reward_loss: float = -20.0            # Penalty for losing match

    # -------------------------------------------------------------------------
    # Opponent Behavior
    # -------------------------------------------------------------------------
    # How player 1 (opponent) is controlled in single-agent training.
    # Options:
    #   "hold"    - All joints held rigid (easiest opponent)
    #   "random"  - Random joint states each turn
    #   "mirror"  - Copies the agent's last action
    opponent_type: str = "hold"

    # -------------------------------------------------------------------------
    # PPO Training Hyperparameters
    # -------------------------------------------------------------------------
    ppo_learning_rate: float = 5e-5
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_target_kl: float = 0.1
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_net_arch: list[int] = field(default_factory=lambda: [256, 256])
