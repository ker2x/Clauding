"""Stage 3: Filter correct traces and format as mlx-lm training data."""

import json
import random

from config import Config
from llmath.oracle import extract_answer


def filter_and_format(config: Config) -> None:
    """Read traces, keep correct ones, output train/valid JSONL for mlx-lm."""
    traces_path = config.TRACES_PATH

    STRONG_METHODS = {"the_answer_is", "boxed"}

    total = 0
    correct = 0
    accepted = 0
    rejected_weak = 0
    rejected_circular = 0
    tier_stats: dict[int, dict[str, int]] = {}
    examples = []

    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total += 1
            tier = data["tier"]

            if tier not in tier_stats:
                tier_stats[tier] = {"total": 0, "correct": 0, "accepted": 0}
            tier_stats[tier]["total"] += 1

            if data["correct"]:
                correct += 1
                tier_stats[tier]["correct"] += 1

                method = data.get("extraction_method")
                if method is None:
                    _, method = extract_answer(data["model_response"])
                if method not in STRONG_METHODS:
                    rejected_weak += 1
                    continue

                reasoning = data["model_response"]

                # Reject looping responses: more than one </think> means the model
                # closed its thinking block multiple times — circular reasoning.
                if reasoning.count("</think>") > 1:
                    rejected_circular += 1
                    continue

                accepted += 1
                tier_stats[tier]["accepted"] += 1

                if "</think>" in reasoning and "<think>" not in reasoning:
                    reasoning = "<think>\n" + reasoning

                examples.append({
                    "messages": [
                        {"role": "user", "content": data.get("prompt", f"What is {data['expression']}?")},
                        {"role": "assistant", "content": reasoning},
                    ]
                })

    # Shuffle and split train/valid
    random.seed(config.SEED)
    random.shuffle(examples)
    n_valid = max(1, int(len(examples) * config.VALID_SPLIT))
    valid_examples = examples[:n_valid]
    train_examples = examples[n_valid:]

    # Write splits
    config.TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TRAIN_PATH, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(config.VALID_PATH, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    # Print stats
    print(f"\nFilter results:")
    print(f"  Total traces: {total}")
    print(f"  Correct: {correct} ({correct / total * 100:.1f}%)" if total > 0 else "  No traces")
    print(f"  Rejected (weak extraction): {rejected_weak}")
    print(f"  Rejected (circular reasoning): {rejected_circular}")
    print(f"  Accepted: {accepted}")
    print(f"  Train: {len(train_examples)}, Valid: {len(valid_examples)}")
    print()
    for tier in sorted(tier_stats.keys()):
        s = tier_stats[tier]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        a = s.get("accepted", 0)
        print(f"  Tier {tier}: {s['correct']}/{s['total']} correct ({acc:.1f}%), {a} accepted")

    print(f"\nSaved to {config.TRAIN_PATH} + {config.VALID_PATH}")
