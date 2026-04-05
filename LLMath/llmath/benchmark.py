"""Benchmark a model on arithmetic expressions (pre- or post-training)."""

import json
import re
from pathlib import Path

from mlx_lm import load, generate

from config import Config


def extract_numeric_answer(text: str) -> str | None:
    """Extract numeric answer from model output."""
    # Look before first </think> tag (model may repeat after)
    if "</think>" in text:
        text = text.split("</think>")[0].strip()

    # "The answer is X"
    match = re.search(r"[Tt]he answer is\s*[:\s]*(-?\d+)", text)
    if match:
        return match.group(1)

    # \boxed{X}
    match = re.search(r"\\boxed\{(-?\d+)\}", text)
    if match:
        return match.group(1)

    # "= X" at end of line
    match = re.search(r"=\s*(-?\d+)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1)

    # Last number
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return numbers[-1]

    return None


def benchmark(config: Config, model_path: str | None = None, max_per_tier: int = 0) -> None:
    """Run benchmark on expressions.

    Args:
        config: Pipeline config.
        model_path: Path to fine-tuned model or adapter, or None for base model.
        max_per_tier: Max expressions per tier (0 = all).
    """
    base = model_path or config.MLX_MODEL
    adapter_path = str(config.MLX_ADAPTER_PATH) if config.MLX_ADAPTER_PATH.exists() else None

    print(f"Loading model: {base}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")

    model, tokenizer = load(base, adapter_path=adapter_path)

    # Load expressions
    expressions = []
    with open(config.EXPRESSIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                expressions.append(json.loads(line))

    # Optionally limit per tier
    if max_per_tier > 0:
        tier_counts: dict[int, int] = {}
        filtered = []
        for e in expressions:
            t = e["tier"]
            tier_counts[t] = tier_counts.get(t, 0) + 1
            if tier_counts[t] <= max_per_tier:
                filtered.append(e)
        expressions = filtered

    print(f"Evaluating {len(expressions)} expressions\n")

    tier_stats: dict[int, dict[str, int]] = {}
    total_correct = 0
    errors = []

    for i, expr_data in enumerate(expressions):
        expr = expr_data["expression"]
        answer = str(expr_data["answer"])
        tier = expr_data["tier"]

        if tier not in tier_stats:
            tier_stats[tier] = {"correct": 0, "total": 0}
        tier_stats[tier]["total"] += 1

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": expr}],
            tokenize=False,
            add_generation_prompt=True,
        ) + "<think>\n"

        response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
        model_answer = extract_numeric_answer(response)

        correct = model_answer == answer
        if correct:
            total_correct += 1
            tier_stats[tier]["correct"] += 1
        else:
            errors.append({
                "expression": expr,
                "expected": answer,
                "got": model_answer,
                "tier": tier,
                "generated": response,
            })

        status = "OK" if correct else f"WRONG (got {model_answer}, expected {answer})"
        if (i + 1) % 50 == 0 or not correct:
            print(f"[{i + 1}/{len(expressions)}] {expr} = {answer} → {status}")
        if model_answer is None:
            print(f"  → raw output: {response[:200]}")

    # Save errors
    errors_path = config.MLX_ADAPTER_PATH / "benchmark_errors.jsonl"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(errors_path, "w") as f:
        for err in errors:
            f.write(json.dumps(err) + "\n")
    print(f"\nLogged {len(errors)} errors to {errors_path}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Model: {base}" + (f" + {adapter_path}" if adapter_path else ""))
    print(f"Total: {total_correct}/{len(expressions)} ({total_correct / len(expressions) * 100:.1f}%)")
    print()
    for tier in sorted(tier_stats):
        s = tier_stats[tier]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  Tier {tier}: {s['correct']}/{s['total']} ({acc:.1f}%)")
