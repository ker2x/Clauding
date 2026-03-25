"""Stage 2: Teacher model client with resumability."""

import json
import re
import time
from pathlib import Path

import requests

from config import Config

SYSTEM_PROMPT = "You are a math calculator. You evaluate arithmetic expressions and return the integer result. Nothing else."

USER_TEMPLATE = """Calculate the result of this arithmetic expression. Only do math, nothing else.

{expression} = ?

Compute step by step, then write "The answer is <number>" with the final integer result."""


def extract_answer(response: str) -> int | None:
    """Extract the final numeric answer from the model's response."""
    # Try "The answer is X" pattern
    match = re.search(r"[Tt]he answer is\s*[:\s]*(-?\d+)", response)
    if match:
        return int(match.group(1))

    # Try \boxed{X} (LaTeX)
    match = re.search(r"\\boxed\{(-?\d+)\}", response)
    if match:
        return int(match.group(1))

    # Try "= X" at end of lines
    match = re.search(r"=\s*(-?\d+)\s*$", response, re.MULTILINE)
    if match:
        return int(match.group(1))

    # Fall back to last number in response
    numbers = re.findall(r"(-?\d+)", response)
    if numbers:
        return int(numbers[-1])

    return None


def load_completed(path: Path) -> set[str]:
    """Load already-processed expressions from traces file."""
    completed = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        completed.add(data["expression"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def query_teacher(expression: str, config: Config) -> str | None:
    """Send expression to teacher model, return response text."""
    url = f"{config.TEACHER_URL}/v1/chat/completions"
    payload = {
        "model": config.TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(expression=expression)},
        ],
        "temperature": config.TEACHER_TEMPERATURE,
        "max_tokens": 2048,
    }

    for attempt in range(config.TEACHER_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=config.TEACHER_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < config.TEACHER_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return None


def run_oracle(config: Config) -> None:
    """Run teacher on all expressions, resumable."""
    expressions_path = config.EXPRESSIONS_PATH
    traces_path = config.TRACES_PATH

    # Load expressions
    expressions = []
    with open(expressions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                expressions.append(json.loads(line))

    # Load already completed
    completed = load_completed(traces_path)
    remaining = [e for e in expressions if e["expression"] not in completed]
    print(f"Total: {len(expressions)}, already done: {len(completed)}, remaining: {len(remaining)}")

    traces_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = config.ERRORS_PATH

    with open(traces_path, "a") as f, open(errors_path, "a") as ef:
        for i, expr_data in enumerate(remaining):
            expr = expr_data["expression"]
            answer = expr_data["answer"]
            tier = expr_data["tier"]

            print(f"[{i + 1}/{len(remaining)}] {expr} = {answer}")
            response = query_teacher(expr, config)

            if not response:
                print(f"  EMPTY RESPONSE — skipping")
                continue

            model_answer = extract_answer(response)
            correct = model_answer == answer

            result = {
                "expression": expr,
                "answer": answer,
                "tier": tier,
                "model_response": response,
                "model_answer": model_answer,
                "correct": correct,
            }

            f.write(json.dumps(result) + "\n")
            f.flush()

            if not correct:
                ef.write(json.dumps(result) + "\n")
                ef.flush()

            if correct:
                print(f"  CORRECT")
            else:
                print(f"  WRONG (got {model_answer}, expected {answer})")
                print(f"  Response: {response}")
