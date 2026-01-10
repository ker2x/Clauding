"""
Configuration for 8x8 American Checkers Training.

Optimized hyperparameters for fixed action space learning.
"""

import torch


class Config:
    """Hyperparameters for 8x8 checkers training."""

    # Network architecture
    NUM_FILTERS = 128
    NUM_RES_BLOCKS = 6
    POLICY_SIZE = 128  # Fixed action space: 32 squares × 4 directions

    # MCTS configuration
    MCTS_SIMS_SELFPLAY = 75  # Simulations per move during self-play
    MCTS_SIMS_EVAL = 100  # Simulations per move during evaluation

    # Exploration in Tree Search (PUCT algorithm)
    C_PUCT = 1.0  # Exploration constant

    # Root Noise (Dirichlet)
    DIRICHLET_ALPHA = 0.5  # Concentration parameter for checkers
    DIRICHLET_EPSILON = 0.25  # Mixing factor

    # Temperature (controlling greediness of move selection)
    TEMPERATURE_THRESHOLD = 15  # Use temperature=1.0 for first 15 moves
    TEMPERATURE = 1.0  # Stochastic move selection

    # Training configuration
    GAMES_PER_ITERATION = 30  # Games to play per iteration
    NUM_WORKERS = 8  # Parallel self-play workers (not implemented yet, sequential for now)
    BUFFER_SIZE = 200_000  # Max positions in replay buffer (increased capacity)

    # Data Sampling
    RECENCY_TAU = 50  # Exponential weighting constant for recent data

    BATCH_SIZE = 256  # Mini-batch size for training
    TRAINING_STEPS_PER_ITERATION = 150  # Gradient steps per iteration
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0  # Gradient clipping max norm

    # Evaluation configuration
    EVAL_FREQUENCY = 5  # Evaluate every N iterations
    EVAL_GAMES = 20  # Games for evaluation
    PROMOTION_THRESHOLD = 0.55  # Win rate needed to become new best model

    # System configuration
    DEVICE = "mps"  # Training device (batch operations)
    SELFPLAY_DEVICE = "cpu"  # Self-play device (MCTS is faster on CPU)
    NUM_THREADS = 8  # CPU threads for PyTorch
    SEED = 42  # Default seed (only used if --seed is specified; otherwise random)

    # Logging and checkpointing
    CHECKPOINT_DIR = "checkpoints8x8"
    LOG_DIR = "logs8x8"
    SAVE_FREQUENCY = 10  # Save checkpoint every N iterations
    LOG_FREQUENCY = 1  # Log metrics every N iterations

    # Game configuration
    MAX_GAME_LENGTH = 200  # Maximum moves per game

    @classmethod
    def get_device(cls) -> torch.device:
        """Get PyTorch device for training."""
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
        """Get PyTorch device for self-play."""
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
        print("8x8 Checkers Configuration")
        print("=" * 60)

        print("\n[Network]")
        print(f"  Filters: {cls.NUM_FILTERS}")
        print(f"  Residual blocks: {cls.NUM_RES_BLOCKS}")
        print(f"  Policy size: {cls.POLICY_SIZE} (FIXED ACTION SPACE)")

        print("\n[MCTS]")
        print(f"  Self-play simulations: {cls.MCTS_SIMS_SELFPLAY}")
        print(f"  Evaluation simulations: {cls.MCTS_SIMS_EVAL}")
        print(f"  C_PUCT: {cls.C_PUCT}")
        print(f"  Dirichlet alpha: {cls.DIRICHLET_ALPHA}")
        print(f"  Temperature threshold: {cls.TEMPERATURE_THRESHOLD}")

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

        print("\n[Evaluation]")
        print(f"  Frequency: every {cls.EVAL_FREQUENCY} iterations")
        print(f"  Games: {cls.EVAL_GAMES}")
        print(f"  Promotion threshold: {cls.PROMOTION_THRESHOLD}")

        print("\n[System]")
        print(f"  Training device: {cls.DEVICE}")
        print(f"  Self-play device: {cls.SELFPLAY_DEVICE}")
        print(f"  PyTorch threads: {cls.NUM_THREADS}")
        print(f"  Seed: {cls.SEED}")

        print("\n[Paths]")
        print(f"  Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"  Logs: {cls.LOG_DIR}")

        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()

    device = Config.get_device()
    selfplay_device = Config.get_selfplay_device()
    print(f"\nTraining device: {device}")
    print(f"Self-play device: {selfplay_device}")

    # Estimate memory usage
    params = 2_316_737  # From network test
    bytes_per_param = 4  # float32
    network_mb = (params * bytes_per_param) / (1024**2)

    buffer_mb = (Config.BUFFER_SIZE * 8 * 8 * 8 * 4) / (1024**2)  # States only

    print(f"\nEstimated memory usage:")
    print(f"  Network: {network_mb:.1f} MB")
    print(f"  Replay buffer (states): {buffer_mb:.1f} MB")
    print(f"  Total (approx): {network_mb + buffer_mb:.1f} MB")
