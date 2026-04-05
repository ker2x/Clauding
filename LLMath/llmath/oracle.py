"""Stage 2: Teacher model client with resumability and multi-GPU distribution."""

import json
import os
import queue
import re
import threading
import time
from pathlib import Path

import requests

from config import Config

def _load_env():
    """Load .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

SYSTEM_PROMPT = "Solve arithmetic expressions. Show your work step by step, then state the final answer."

USER_TEMPLATE = "{prompt}"


def extract_answer(response: str) -> tuple[int | None, str]:
    """Extract the final numeric answer from the model's response.

    Returns (answer, method) where method indicates how the answer was found.
    """
    # Try "The answer is X" pattern
    match = re.search(r"[Tt]he answer is\s*[:\s]*(-?\d+)", response)
    if match:
        return int(match.group(1)), "the_answer_is"

    # Try \boxed{X} (LaTeX)
    match = re.search(r"\\boxed\{(-?\d+)\}", response)
    if match:
        return int(match.group(1)), "boxed"

    # Try "= X" at end of lines
    match = re.search(r"=\s*(-?\d+)\s*$", response, re.MULTILINE)
    if match:
        return int(match.group(1)), "equals_eol"

    # Fall back to last number in response
    numbers = re.findall(r"(-?\d+)", response)
    if numbers:
        return int(numbers[-1]), "fallback"

    return None, "none"


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


def query_teacher(expression: str, base_url: str, config: Config) -> str | None:
    """Send expression to teacher model, return response text."""
    url = f"{base_url}/v1/chat/completions"

    # Use OpenRouter model name and auth for openrouter URLs
    is_openrouter = "openrouter.ai" in base_url
    model = "liquid/lfm-2.5-1.2b-thinking:free" if is_openrouter else config.TEACHER_MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(prompt=expression)},
        ],
        "temperature": config.TEACHER_TEMPERATURE,
        "max_tokens": 4096,
    }

    headers = {}
    if is_openrouter:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(config.TEACHER_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=config.TEACHER_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"  [{base_url}] Attempt {attempt + 1} failed: {e}")
            if attempt < config.TEACHER_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return None


def run_oracle(config: Config) -> None:
    """Run teacher on all expressions, resumable, distributed across TEACHER_URLS."""
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
    urls = [url for url, n in config.TEACHER_URLS.items() for _ in range(n)]
    print(f"Total: {len(expressions)}, already done: {len(completed)}, remaining: {len(remaining)}")
    print(f"Using {len(urls)} workers across {len(config.TEACHER_URLS)} endpoints")

    traces_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = config.ERRORS_PATH

    # Shared work queue — each worker thread pulls the next task when idle
    work_queue = queue.Queue()
    for expr_data in remaining:
        work_queue.put(expr_data)

    write_lock = threading.Lock()
    counter = {"done": 0, "correct": 0, "wrong": 0}

    def worker(base_url, traces_file, errors_file):
        while True:
            try:
                expr_data = work_queue.get_nowait()
            except queue.Empty:
                return

            expr = expr_data["expression"]
            prompt = expr_data.get("prompt", f"What is {expr}?")
            answer = expr_data["answer"]
            tier = expr_data["tier"]

            response = query_teacher(prompt, base_url, config)

            with write_lock:
                counter["done"] += 1
                idx = counter["done"]

                if not response:
                    print(f"[{idx}/{len(remaining)}] {expr} — EMPTY RESPONSE [{base_url}]")
                    work_queue.task_done()
                    continue

                model_answer, extraction_method = extract_answer(response)
                correct = model_answer == answer

                result = {
                    "expression": expr,
                    "prompt": prompt,
                    "answer": str(answer),
                    "tier": tier,
                    "model_response": response,
                    "model_answer": model_answer,
                    "correct": correct,
                    "extraction_method": extraction_method,
                }

                traces_file.write(json.dumps(result) + "\n")
                traces_file.flush()

                if correct:
                    counter["correct"] += 1
                    print(f"[{idx}/{len(remaining)}] {expr} = {answer} — CORRECT [{base_url}]")
                else:
                    counter["wrong"] += 1
                    errors_file.write(json.dumps(result) + "\n")
                    errors_file.flush()
                    print(f"[{idx}/{len(remaining)}] {expr} — WRONG (got {model_answer}, expected {answer}) [{base_url}]")

            work_queue.task_done()

    with open(traces_path, "a") as f, open(errors_path, "a") as ef:
        threads = []
        for url in urls:
            t = threading.Thread(target=worker, args=(url, f, ef))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    print(f"\nDone: {counter['correct']} correct, {counter['wrong']} wrong out of {counter['done']}")
