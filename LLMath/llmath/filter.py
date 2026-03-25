"""Stage 3: Filter correct traces and format as training data."""

import json
from pathlib import Path

from config import Config


def filter_and_format(config: Config) -> None:
    """Read traces, keep correct ones, output training-format JSONL."""
    traces_path = config.TRACES_PATH
    dataset_path = config.DATASET_PATH

    total = 0
    correct = 0
    tier_stats: dict[int, dict[str, int]] = {}
    training_examples = []

    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total += 1
            tier = data["tier"]

            if tier not in tier_stats:
                tier_stats[tier] = {"total": 0, "correct": 0}
            tier_stats[tier]["total"] += 1

            if data["correct"]:
                correct += 1
                tier_stats[tier]["correct"] += 1

                example = {
                    "prompt": f"What is {data['expression']}?",
                    "reasoning": data["model_response"],
                    "answer": str(data["answer"]),
                }
                training_examples.append(example)

    # Write dataset
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")

    # Print stats
    print(f"\nFilter results:")
    print(f"  Total traces: {total}")
    print(f"  Correct: {correct} ({correct / total * 100:.1f}%)" if total > 0 else "  No traces")
    print(f"  Dataset size: {len(training_examples)}")
    print()
    for tier in sorted(tier_stats.keys()):
        s = tier_stats[tier]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  Tier {tier}: {s['correct']}/{s['total']} correct ({acc:.1f}%)")

    print(f"\nSaved to {dataset_path}")
