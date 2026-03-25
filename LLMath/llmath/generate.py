"""Stage 1: Expression generator with progressive curriculum."""

import json
import random
from pathlib import Path

from config import Config, TierConfig

OPERATORS = ["+", "-", "*", "/"]


def random_expression(rng: random.Random, tier: TierConfig) -> tuple[str, int | None]:
    """Generate a random arithmetic expression and evaluate it.

    Returns (expression_string, answer) where answer is None if invalid
    (non-integer result or division by zero).
    """
    num_ops = rng.randint(*tier.num_ops)
    operands = [rng.randint(*tier.operand_range) for _ in range(num_ops + 1)]
    ops = [rng.choice(OPERATORS) for _ in range(num_ops)]

    # Build expression tree as nested pairs, then render to infix
    # Simple approach: linear chain with optional parenthesized sub-expressions
    tokens = []
    i = 0
    while i < len(operands):
        if i > 0:
            tokens.append(ops[i - 1])

        # Decide whether to parenthesize a sub-expression (current + next operand)
        if (
            tier.paren_probability > 0
            and i + 1 < len(operands)
            and i < len(ops)
            and rng.random() < tier.paren_probability
        ):
            a, b = operands[i], operands[i + 1]
            op = ops[i]
            sub = f"({_format_operand(a)} {op} {_format_operand(b)})"
            tokens.append(sub)
            i += 2
        else:
            tokens.append(_format_operand(operands[i]))
            i += 1

    expr = " ".join(tokens)

    # Evaluate
    try:
        result = eval(expr)  # noqa: S307
    except ZeroDivisionError:
        return expr, None

    if not isinstance(result, int) and not (isinstance(result, float) and result == int(result)):
        return expr, None

    return expr, int(result)


def _format_operand(n: int) -> str:
    if n < 0:
        return f"({n})"
    return str(n)


def generate_expressions(config: Config) -> list[dict]:
    """Generate expressions for all tiers."""
    rng = random.Random(config.SEED)
    results = []

    for tier_idx, tier in enumerate(config.TIERS, start=1):
        count = 0
        seen = set()
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
            results.append({
                "expression": expr,
                "answer": answer,
                "tier": tier_idx,
            })
            count += 1

        print(f"Tier {tier_idx}: generated {count}/{tier.count} expressions")

    return results


def save_expressions(expressions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for expr in expressions:
            f.write(json.dumps(expr) + "\n")
    print(f"Saved {len(expressions)} expressions to {path}")
