from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TierConfig:
    num_ops: tuple[int, int]  # (min, max) number of operators
    operand_range: tuple[int, int]  # (min, max) operand value
    count: int  # number of expressions to generate


@dataclass
class Config:
    # Seed
    SEED: int = 42

    # Paths
    EXPRESSIONS_PATH: Path = Path("data/expressions.jsonl")
    TRACES_PATH: Path = Path("data/traces.jsonl")
    ERRORS_PATH: Path = Path("data/errors.jsonl")
    TRAIN_PATH: Path = Path("data/train.jsonl")
    VALID_PATH: Path = Path("data/valid.jsonl")

    # Expression generation tiers
    TIERS: list[TierConfig] = field(default_factory=lambda: [
        # Small numbers — build foundation
        TierConfig(num_ops=(1, 1), operand_range=(-20, 20),    count=500),  # Tier 1
        TierConfig(num_ops=(2, 2), operand_range=(-50, 50),    count=500),  # Tier 2
        TierConfig(num_ops=(3, 4), operand_range=(-100, 100),  count=500),  # Tier 3
        # Larger numbers — same structure, harder arithmetic
        TierConfig(num_ops=(1, 1), operand_range=(-999, 999),  count=500),  # Tier 4
        TierConfig(num_ops=(2, 3), operand_range=(-999, 999),  count=500),  # Tier 5
        TierConfig(num_ops=(3, 5), operand_range=(-9999, 9999), count=500), # Tier 6
    ])

    # Oracle / Teacher
    # Format: {url: concurrency} — number of parallel worker threads per endpoint
    TEACHER_URLS: dict[str, int] = field(default_factory=lambda: {
    #    "http://127.0.0.1:11434":       2,  # Mac Mini (local)
    #    "http://192.168.1.40:8000":    64,  # NVIDIA box
        "http://192.168.1.110:8000":    128,  # shitbox
    #    "http://192.168.1.43:11434":    2,  # Mac M1
    #    "http://192.168.1.17:11434":    2,  # AMD Box
    #    "http://142.171.48.138:28804":  64, # Vast.ai RTX 5070 (vLLM)
    #   "https://openrouter.ai/api":    1,
    })
    TEACHER_MODEL: str = "lfm2.5-thinking"
    TEACHER_TEMPERATURE: float = 0.6
    TEACHER_TIMEOUT: int = 120
    TEACHER_MAX_RETRIES: int = 3

    # MLX Training
    MLX_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
    MLX_ADAPTER_PATH: Path = Path("adapters")
    MLX_TRAIN_ITERS: int = 1000
    MLX_BATCH_SIZE: int = 2
    MLX_LEARNING_RATE: float = 1e-4
    MLX_MAX_SEQ_LEN: int = 1024
    VALID_SPLIT: float = 0.1
