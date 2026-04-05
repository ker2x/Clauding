"""Answer extraction utilities for GSM8K responses."""

import re


def extract_ground_truth(answer_str: str) -> int | None:
    """Extract final numeric answer from GSM8K format (#### N)."""
    match = re.search(r"####\s*([\d,]+)", answer_str)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def extract_model_answer(response: str) -> tuple[int | None, str]:
    """Extract the final numeric answer from the model's response.

    Returns (answer, method) where method indicates how the answer was found.
    Priority: boxed > the_answer_is > gsm8k_format > equals_eol > fallback.
    """
    # Try $\boxed{X}$ or \boxed{X} (official GSM8K eval format)
    match = re.search(r"\$?\\boxed\{(-?[\d,]+)\}\$?", response)
    if match:
        return int(match.group(1).replace(",", "")), "boxed"

    # Try "The answer is X" / "the final answer is X"
    match = re.search(r"[Tt]he (?:final )?answer is[:\s]*\$?(-?[\d,]+)", response)
    if match:
        return int(match.group(1).replace(",", "")), "the_answer_is"

    # Try #### X (model mimicking GSM8K format)
    match = re.search(r"####\s*(-?[\d,]+)", response)
    if match:
        return int(match.group(1).replace(",", "")), "gsm8k_format"

    # Try "= X" at end of lines
    match = re.search(r"=\s*(-?[\d,]+)\s*$", response, re.MULTILINE)
    if match:
        return int(match.group(1).replace(",", "")), "equals_eol"

    # Fall back to last number in response
    numbers = re.findall(r"(-?\d+)", response)
    if numbers:
        return int(numbers[-1]), "fallback"

    return None, "none"


def extract_eval_rating(response: str) -> str:
    """Extract CORRECT/INCORRECT/UNSURE from an evaluator response.

    Strips <think> blocks first so we only look at the final answer.
    """
    # Take only text after the last </think> tag (model may emit multiple think blocks)
    parts = response.split("</think>")
    text = parts[-1].strip().upper()

    # Find the first occurrence of any rating keyword
    import re
    match = re.search(r"\b(INCORRECT|CORRECT|UNSURE)\b", text)
    if match:
        return match.group(1)
    return "UNSURE"
