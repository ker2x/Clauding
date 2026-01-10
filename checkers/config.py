"""
Configuration file for AlphaZero-style checkers training.

Centralizes all hyperparameters in one place.
"""

import torch


class Config:
    """Hyperparameters for checkers training."""

    # Network architecture
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 6
    POLICY_SIZE = 150

    # MCTS configuration
    # Unlike DQN/SAC which use the network to act directly, AlphaZero uses MCTS for policy improvement.
    # The network provides a prior P(s,a) and value V(s), which MCTS improves to a posterior policy π.
    MCTS_SIMS_SELFPLAY = 100  # Simulations per move. Reduced to 25 for faster self-play on Apple Silicon.
                              # Higher = better quality data but slower generation.
    MCTS_SIMS_EVAL = 100   # Reduced for fast evaluation.

    # Exploration in Tree Search (PUCT algorithm)
    C_PUCT = 1.0  # Exploration constant in the Upper Confidence Bound formula.
                  # Balances Q-value (exploitation) vs Prior/Visit_Count (exploration).

    # Root Noise (Dirichlet)
    # Added to the root node's prior probabilities to forcibly explore new moves during self-play.
    # prevents the agent from getting stuck in local optima early on.
    DIRICHLET_ALPHA = 0.3  # Concentration parameter. Lower (e.g. 0.03 for Go) = more concentrated noise.
                           # 0.3 is often used for Chess/Checkers.
    DIRICHLET_EPSILON = 0.25  # Mixing factor: Policy = (1-eps)*Network_Prob + eps*Noise.

    # Temperature (controlling greediness of move selection)
    # The final policy π is derived from visit counts N: π ~ N^(1/temp).
    TEMPERATURE_THRESHOLD = 15  # For the first 15 moves, use temperature > 0 (stochastic).
                                # After valid move 15, drop to temp=0 (deterministic/greedy) to stabilize training.
    TEMPERATURE = 1.0  # Initial temperature. 1.0 = select proportional to visit counts.
                       # Higher = more random, Lower = more greedy.
    MCTS_BATCH_SIZE = 64  # Batch size for evaluating leaf nodes on GPU during one search.

    # Training configuration
    # AlphaZero loops: Self-Play (Data Gen) -> Training (Network Update) -> Evaluation
    GAMES_PER_ITERATION = 5  # Number of full games to play against self per iteration to fill buffer.
    NUM_WORKERS = 8  # Parallel self-play workers for concurrent game generation.
    BUFFER_SIZE = 200_000  # Max number of positions (s, pi, z) in replay buffer.

    # Data Sampling
    # Unlike DQN which often uses uniform sampling, we might want to prioritize newer data 
    # as the policy shifts rapidly.
    RECENCY_TAU = 50  # Constant for exponential weighting of sample probability by age.

    BATCH_SIZE = 256  # Mini-batch size for network updates.
    TRAINING_STEPS_PER_ITERATION = 800  # Increased to 800 to balance with 219s self-play time.
                                        # (8x augmentation removed -> 8x more training steps on real data).
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0  # Gradient clipping max norm

    # Data augmentation
    AUGMENTATION = False  # DISABLED: Rotations break checkers parity (black squares -> white)
                          # and invalidate policy indices (dynamic action space).
    AUG_FACTOR = 8  # Number of augmented samples per position

    # Evaluation configuration
    # In AlphaZero, the new network plays against the previous best network.
    EVAL_FREQUENCY = 5  # Evaluate every 5 iterations to reduce overhead.
    EVAL_GAMES = 10  # Minimal games for evaluation (2 games = 1 as P1, 1 as P2).
    PROMOTION_THRESHOLD = 0.55  # Fraction of wins needed to become the new "best" model.
                                # >0.5 ensures we replace only if significantly stronger (reduces cycling).

    # System configuration
    DEVICE = "mps"  # 'cpu', 'cuda', or 'mps' - used for TRAINING (batch operations)
    SELFPLAY_DEVICE = "cpu"  # Device for self-play (use 'cpu' for better MCTS performance with small batches)
    NUM_THREADS = 8  # CPU threads for PyTorch
    SEED = 42  # Random seed

    # Logging and checkpointing
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    SAVE_FREQUENCY = 10  # Save checkpoint every N iterations
    LOG_FREQUENCY = 1  # Log metrics every N iterations

    # Visualization configuration
    VISUALIZE_TRAINING = False  # Enable real-time pygame visualization
    VISUALIZE_UPDATE_RATE = 5.0  # Seconds between GUI updates (to avoid overhead)
    VISUALIZE_HISTORY_LENGTH = 100  # Iterations to show in plots
    VISUALIZE_WINDOW_WIDTH = 1400  # Window width in pixels
    VISUALIZE_WINDOW_HEIGHT = 800  # Window height in pixels

    # Game configuration
    MAX_GAME_LENGTH = 200  # Maximum moves per game

    @classmethod
    def get_device(cls) -> torch.device:
        """Get PyTorch device for training based on configuration."""
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
        """Get PyTorch device for self-play based on configuration."""
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
        """Print all configuration settings."""
        print("=" * 60)
        print("Configuration")
        print("=" * 60)

        print("\n[Network]")
        print(f"  Filters: {cls.NUM_FILTERS}")
        print(f"  Residual blocks: {cls.NUM_RES_BLOCKS}")
        print(f"  Policy size: {cls.POLICY_SIZE}")

        print("\n[MCTS]")
        print(f"  Self-play simulations: {cls.MCTS_SIMS_SELFPLAY}")
        print(f"  Evaluation simulations: {cls.MCTS_SIMS_EVAL}")
        print(f"  C_PUCT: {cls.C_PUCT}")
        print(f"  Dirichlet alpha: {cls.DIRICHLET_ALPHA}")
        print(f"  Temperature threshold: {cls.TEMPERATURE_THRESHOLD}")
        print(f"  Batch size: {cls.MCTS_BATCH_SIZE}")

        print("\n[Training]")
        print(f"  Games per iteration: {cls.GAMES_PER_ITERATION}")
        print(f"  Workers: {cls.NUM_WORKERS}")
        print(f"  Buffer size: {cls.BUFFER_SIZE:,}")
        print(f"  Recency tau: {cls.RECENCY_TAU}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Training steps per iteration: {cls.TRAINING_STEPS_PER_ITERATION}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        print(f"  Gradient clipping: {cls.GRAD_CLIP}")

        print("\n[Data Augmentation]")
        print(f"  Enabled: {cls.AUGMENTATION}")
        print(f"  Factor: {cls.AUG_FACTOR}x")

        print("\n[Evaluation]")
        print(f"  Frequency: every {cls.EVAL_FREQUENCY} iterations")
        print(f"  Games: {cls.EVAL_GAMES}")
        print(f"  Promotion threshold: {cls.PROMOTION_THRESHOLD}")

        print("\n[System]")
        print(f"  Training device: {cls.DEVICE}")
        print(f"  Self-play device: {cls.SELFPLAY_DEVICE}")
        print(f"  PyTorch threads: {cls.NUM_THREADS} (internal compute)")
        print(f"  Seed: {cls.SEED}")

        print("\n[Paths]")
        print(f"  Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"  Logs: {cls.LOG_DIR}")

        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()

    device = Config.get_device()
    print(f"\nUsing device: {device}")

    # Estimate memory usage
    params = 2_682_071  # From network test
    bytes_per_param = 4  # float32
    network_mb = (params * bytes_per_param) / (1024**2)

    buffer_mb = (Config.BUFFER_SIZE * 8 * 10 * 10 * 4) / (1024**2)  # States only

    print(f"\nEstimated memory usage:")
    print(f"  Network: {network_mb:.1f} MB")
    print(f"  Replay buffer (states): {buffer_mb:.1f} MB")
    print(f"  Total (approx): {network_mb + buffer_mb:.1f} MB")
