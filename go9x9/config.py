"""
Configuration for 9x9 Go Training.
"""

import torch


class Config:
    """Hyperparameters for 9x9 Go training."""

    # Network architecture
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 10
    INPUT_PLANES = 17  # 8 history × 2 colors + 1 color-to-play
    POLICY_SIZE = 82   # 81 intersections + pass
    GLOBAL_POOL_FREQ = 3      # Insert global-pool-bias layer every N residual blocks
    OWNERSHIP_LOSS_WEIGHT = 0.5  # Weight on ownership auxiliary head BCE loss

    # MCTS configuration
    MCTS_SIMS_SELFPLAY = 400
    MCTS_SIMS_FAST = 50        # sims for non-training (fast) moves
    MCTS_SIMS_EVAL = 200
    MCTS_BATCH_SIZE = 16

    # Playout cap randomization (KataGo-style)
    P_FAST_MOVE = 0.75         # fraction of moves that are fast (not stored in buffer)

    # Exploration
    C_PUCT = 1.0
    DIRICHLET_ALPHA = 0.15  # ~10/avg_legal_moves for 9x9 Go (~50-60 legal moves)
    DIRICHLET_EPSILON = 0.25

    # Temperature
    TEMPERATURE_THRESHOLD = 30  # Longer games than checkers
    TEMPERATURE = 1.0

    # Training configuration
    GAMES_PER_ITERATION = 20
    NUM_WORKERS = 6
    BUFFER_SIZE = 50_000

    # Data Sampling
    RECENCY_TAU = 30

    BATCH_SIZE = 256
    MIN_SAMPLE_REUSE = 10
    MAX_SAMPLE_REUSE = 15
    LEARNING_RATE = 0.002
    LR_MIN = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    # Evaluation
    EVAL_FREQUENCY = 25
    EVAL_GAMES = 11
    PROMOTION_THRESHOLD = 0.55

    # System
    DEVICE = "mps"
    SELFPLAY_DEVICE = "mps"
    NUM_THREADS = 8
    SEED = 42

    # Logging and checkpointing
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    SAVE_FREQUENCY = 10
    LOG_FREQUENCY = 1

    # Game configuration
    MAX_GAME_LENGTH = 200
    KOMI = 7.5

    @classmethod
    def get_device(cls) -> torch.device:
        if cls.DEVICE == "cpu":
            torch.set_num_threads(cls.NUM_THREADS)
            return torch.device("cpu")
        elif cls.DEVICE == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif cls.DEVICE == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print(f"Warning: {cls.DEVICE} not available, falling back to CPU")
            torch.set_num_threads(cls.NUM_THREADS)
            return torch.device("cpu")

    @classmethod
    def get_selfplay_device(cls) -> torch.device:
        if cls.SELFPLAY_DEVICE == "cpu":
            torch.set_num_threads(cls.NUM_THREADS)
            return torch.device("cpu")
        elif cls.SELFPLAY_DEVICE == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif cls.SELFPLAY_DEVICE == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print(f"Warning: {cls.SELFPLAY_DEVICE} not available, falling back to CPU")
            torch.set_num_threads(cls.NUM_THREADS)
            return torch.device("cpu")

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("9x9 Go Configuration")
        print("=" * 60)

        print("\n[Network]")
        print(f"  Filters: {cls.NUM_FILTERS}")
        print(f"  Residual blocks: {cls.NUM_RES_BLOCKS}")
        print(f"  Input planes: {cls.INPUT_PLANES}")
        print(f"  Policy size: {cls.POLICY_SIZE}")

        print("\n[MCTS]")
        print(f"  Self-play simulations: {cls.MCTS_SIMS_SELFPLAY} (slow) / {cls.MCTS_SIMS_FAST} (fast)")
        print(f"  Fast move probability: {cls.P_FAST_MOVE} (avg sims: {cls.P_FAST_MOVE*cls.MCTS_SIMS_FAST + (1-cls.P_FAST_MOVE)*cls.MCTS_SIMS_SELFPLAY:.0f})")
        print(f"  Evaluation simulations: {cls.MCTS_SIMS_EVAL}")
        print(f"  MCTS batch size: {cls.MCTS_BATCH_SIZE}")
        print(f"  C_PUCT: {cls.C_PUCT}")
        print(f"  Dirichlet alpha: {cls.DIRICHLET_ALPHA}")
        print(f"  Temperature threshold: {cls.TEMPERATURE_THRESHOLD}")

        print("\n[Training]")
        print(f"  Games per iteration: {cls.GAMES_PER_ITERATION}")
        print(f"  Workers: {cls.NUM_WORKERS}")
        print(f"  Buffer size: {cls.BUFFER_SIZE:,}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Sample Reuse: {cls.MIN_SAMPLE_REUSE} to {cls.MAX_SAMPLE_REUSE}x")
        print(f"  Learning rate: {cls.LEARNING_RATE} → {cls.LR_MIN} (cosine)")

        print("\n[Evaluation]")
        print(f"  Frequency: every {cls.EVAL_FREQUENCY} iterations")
        print(f"  Games: {cls.EVAL_GAMES}")

        print("\n[System]")
        print(f"  Training device: {cls.DEVICE}")
        print(f"  Self-play device: {cls.SELFPLAY_DEVICE}")
        print(f"  Komi: {cls.KOMI}")

        print("=" * 60)
