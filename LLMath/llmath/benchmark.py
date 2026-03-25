"""Benchmark a model on arithmetic expressions (pre- or post-training)."""

import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config


def extract_numeric_answer(text: str) -> str | None:
    """Extract numeric answer from model output."""
    # After </think> tag
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

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
        model_path: Path to fine-tuned checkpoint, or None for base model.
        max_per_tier: Max expressions per tier (0 = all).
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = model_path or config.STUDENT_MODEL

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    with torch.no_grad():
        for i, expr_data in enumerate(expressions):
            expr = expr_data["expression"]
            answer = str(expr_data["answer"])
            tier = expr_data["tier"]

            if tier not in tier_stats:
                tier_stats[tier] = {"correct": 0, "total": 0}
            tier_stats[tier]["total"] += 1

            prompt_str = f"<|im_start|>user\nWhat is {expr}?<|im_end|>\n<|im_start|>assistant\n<think>\n"
            input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=eos_token_id,
            )

            generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            model_answer = extract_numeric_answer(generated)

            correct = model_answer == answer
            if correct:
                total_correct += 1
                tier_stats[tier]["correct"] += 1

            status = "OK" if correct else f"WRONG (got {model_answer}, expected {answer})"
            if (i + 1) % 50 == 0 or not correct:
                print(f"[{i + 1}/{len(expressions)}] {expr} = {answer} → {status}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"Total: {total_correct}/{len(expressions)} ({total_correct / len(expressions) * 100:.1f}%)")
    print()
    for tier in sorted(tier_stats):
        s = tier_stats[tier]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  Tier {tier}: {s['correct']}/{s['total']} ({acc:.1f}%)")
