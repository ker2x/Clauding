"""Stage 1: Expression generator with progressive curriculum."""

import json
import random
from pathlib import Path

from config import Config, TierConfig

OPERATORS = ["+", "-", "*", "/"]

PROMPT_TEMPLATES = [
    "What is {expr}?",
    "Calculate {expr}",
    "Solve: {expr}",
    "{expr} = ?",
    "What's the result of {expr}?",
    "Evaluate {expr}",
    "Compute {expr}",
    "What does {expr} equal?",
    "Find the value of {expr}",
    "{expr}",
]


def random_expression(rng: random.Random, tier: TierConfig) -> tuple[str, int | None]:
    """Generate a random arithmetic expression and evaluate it.

    Returns (expression_string, answer) where answer is None if invalid
    (non-integer result or division by zero).
    """
    num_ops = rng.randint(*tier.num_ops)
    expr, value = _random_tree(rng, tier, budget=num_ops)
    if value is None:
        return expr, None
    return expr, value


def _random_tree(rng: random.Random, tier: TierConfig, budget: int) -> tuple[str, int | None]:
    """Recursively build a random expression tree.

    budget: number of operators remaining to place.
    Returns (expression_string, value) where value is None if invalid.
    """
    if budget == 0:
        n = rng.randint(*tier.operand_range)
        return _format_operand(n), n

    # Split budget between left and right subtrees
    left_budget = rng.randint(0, budget - 1)
    right_budget = budget - 1 - left_budget

    left_expr, left_val = _random_tree(rng, tier, left_budget)
    right_expr, right_val = _random_tree(rng, tier, right_budget)

    if left_val is None or right_val is None:
        return "", None

    op = rng.choice(OPERATORS)

    if op == "+":
        value = left_val + right_val
    elif op == "-":
        value = left_val - right_val
    elif op == "*":
        value = left_val * right_val
    elif op == "/":
        if right_val == 0 or left_val % right_val != 0:
            return "", None
        value = left_val // right_val
    else:
        return "", None

    # Parenthesize non-leaf subexpressions to make precedence explicit
    left_str = f"({left_expr})" if left_budget > 0 else left_expr
    right_str = f"({right_expr})" if right_budget > 0 else right_expr

    return f"{left_str} {op} {right_str}", value


def _format_operand(n: int) -> str:
    if n < 0:
        return f"({n})"
    return str(n)


def generate_expressions(config: Config, seed: int | None = None, existing: set[str] | None = None) -> list[dict]:
    """Generate expressions for all tiers."""
    rng = random.Random(seed if seed is not None else config.SEED)
    results = []

    for tier_idx, tier in enumerate(config.TIERS, start=1):
        count = 0
        seen = set(existing) if existing else set()
        attempts = 0
        max_attempts = tier.count * 20

        while count < tier.count and attempts < max_attempts:
            attempts += 1
            expr, answer = random_expression(rng, tier)

            if answer is None:
                continue
            if expr in seen:
                continue

            seen.add(expr)
            prompt = rng.choice(PROMPT_TEMPLATES).format(expr=expr)
            results.append({
                "expression": expr,
                "prompt": prompt,
                "answer": answer,
                "tier": tier_idx,
            })
            count += 1

        print(f"Tier {tier_idx}: generated {count}/{tier.count} expressions")

    return results


def save_expressions(expressions: list[dict], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for expr in expressions:
            f.write(json.dumps(expr) + "\n")
    action = "Appended" if append else "Saved"
    print(f"{action} {len(expressions)} expressions to {path}")
