from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    QUESTIONS_PATH: Path = Path("data/questions.jsonl")
    TRACES_PATH: Path = Path("data/traces.jsonl")
    EVALUATED_PATH: Path = Path("data/evaluated.jsonl")

    # Solver (vLLM endpoint)
    SOLVER_URL: str = "http://192.168.1.110:8000"
    #SOLVER_MODEL: str = "lfm2.5-thinking"
    SOLVER_MODEL: str = "Qwen/Qwen3-1.7B"
    SOLVER_TEMPERATURE: float = 0.1
    SOLVER_TOP_K: int = 50
    SOLVER_REPETITION_PENALTY: float = 1.05
    SOLVER_MAX_TOKENS: int = 6000
    SOLVER_CONCURRENCY: int = 32
    SOLVER_TIMEOUT: int = 240
    SOLVER_MAX_RETRIES: int = 3

    # Self-evaluator (reuses solver endpoint)
    #SELF_EVAL_MODEL: str = "lfm2.5-thinking"
    SELF_EVAL_MODEL: str = "Qwen/Qwen3-1.7B"
    SELF_EVAL_CONCURRENCY: int = 64
    SELF_EVAL_MAX_TOKENS: int = 6000

    # Smart evaluator (different model, same box)
    SMART_EVAL_URL: str = "http://192.168.1.40:8000"
    SMART_EVAL_MODEL: str = "qwen3.5"
    SMART_EVAL_CONCURRENCY: int = 32
    SMART_EVAL_MAX_TOKENS: int = 512
