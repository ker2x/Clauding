"""Stage 4: Fine-tune student model with mlx-lm LoRA."""

import subprocess

from config import Config


def train(config: Config, resume_from: str | None = None) -> None:
    """Run mlx-lm LoRA training."""
    config.MLX_ADAPTER_PATH.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mlx_lm", "lora",
        "--model", config.MLX_MODEL,
        "--train",
        "--data", str(config.TRAIN_PATH.parent),
        "--iters", str(config.MLX_TRAIN_ITERS),
        "--batch-size", str(config.MLX_BATCH_SIZE),
        "--learning-rate", str(config.MLX_LEARNING_RATE),
        "--max-seq-length", str(config.MLX_MAX_SEQ_LEN),
        "--adapter-path", str(config.MLX_ADAPTER_PATH),
        "--steps-per-report", "10",
        "--steps-per-eval", "500",
        "--save-every", "100",
    ]
    if resume_from:
        cmd += ["--resume-adapter-file", resume_from]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
