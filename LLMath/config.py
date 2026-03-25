from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TierConfig:
    num_ops: tuple[int, int]  # (min, max) number of operators
    operand_range: tuple[int, int]  # (min, max) operand value
    paren_probability: float  # probability of adding parentheses
    count: int  # number of expressions to generate


@dataclass
class Config:
    # Seed
    SEED: int = 42

    # Paths
    EXPRESSIONS_PATH: Path = Path("data/expressions.jsonl")
    TRACES_PATH: Path = Path("data/traces.jsonl")
    ERRORS_PATH: Path = Path("data/errors.jsonl")
    DATASET_PATH: Path = Path("data/dataset.jsonl")
    CHECKPOINT_DIR: Path = Path("checkpoints")

    # Expression generation tiers
    TIERS: list[TierConfig] = field(default_factory=lambda: [
        TierConfig(num_ops=(1, 1), operand_range=(-20, 20), paren_probability=0.0, count=500),
        TierConfig(num_ops=(2, 2), operand_range=(-50, 50), paren_probability=0.3, count=500),
        TierConfig(num_ops=(3, 4), operand_range=(-100, 100), paren_probability=0.4, count=500),
    ])

    # Oracle / Teacher
    #TEACHER_URL: str = "http://192.168.1.17:11434"
    TEACHER_URL: str = "http://127.0.0.1:11434"
    TEACHER_MODEL: str = "lfm2.5-thinking"
    TEACHER_TEMPERATURE: float = 0.6
    TEACHER_TIMEOUT: int = 120
    TEACHER_MAX_RETRIES: int = 3

    # Training
    STUDENT_MODEL: str = "models/Qwen2.5-0.5B"
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 2
    GRAD_ACCUM_STEPS: int = 16
    NUM_EPOCHS: int = 3
    MAX_SEQ_LEN: int = 384
    EVAL_SPLIT: float = 0.1
    WARMUP_STEPS: int = 50
    CHECKPOINT_EVERY: int = 200
    WEIGHT_DECAY: float = 0.01
