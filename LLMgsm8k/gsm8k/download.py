"""Stage 1: Download GSM8K parquet from HuggingFace, write as JSONL."""

import json

import pandas as pd

from config import Config
from gsm8k.extract import extract_ground_truth

HF_BASE = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main"
SPLITS = {
    "test": f"{HF_BASE}/test-00000-of-00001.parquet",
    "train": f"{HF_BASE}/train-00000-of-00001.parquet",
}


def download(config: Config) -> None:
    """Download both splits and write to a single questions JSONL."""
    config.QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    question_id = 0
    total = 0

    with open(config.QUESTIONS_PATH, "w") as f:
        for split_name, url in SPLITS.items():
            print(f"Downloading {split_name} split from {url}...")
            df = pd.read_parquet(url)
            print(f"  Got {len(df)} rows")

            for _, row in df.iterrows():
                ground_truth = extract_ground_truth(row["answer"])
                record = {
                    "id": question_id,
                    "split": split_name,
                    "question": row["question"],
                    "ground_truth_full": row["answer"],
                    "ground_truth_answer": ground_truth,
                }
                f.write(json.dumps(record) + "\n")
                question_id += 1

            total += len(df)

    print(f"Wrote {total} questions to {config.QUESTIONS_PATH}")
