"""Stage 2: Send GSM8K questions to vLLM, save traces with thinking."""

import asyncio
import json
import time

import aiohttp

from config import Config
from gsm8k.extract import extract_model_answer

PROMPT_TEMPLATE = """\
Solve the following math problem efficiently and clearly. The last line \
of your response should be of the following format: 'Therefore, the final \
answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) \
where ANSWER is just the final number or expression that solves the \
problem. Think step by step before answering.

{question}"""


def load_completed(path) -> set[int]:
    """Load IDs already in traces file."""
    completed = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


async def solve_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    question: dict,
    config: Config,
) -> dict | None:
    """Query vLLM for a single question."""
    url = f"{config.SOLVER_URL}/v1/chat/completions"
    payload = {
        "model": config.SOLVER_MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(question=question["question"])},
        ],
        "temperature": config.SOLVER_TEMPERATURE,
        "top_k": config.SOLVER_TOP_K,
        "repetition_penalty": config.SOLVER_REPETITION_PENALTY,
        "max_tokens": config.SOLVER_MAX_TOKENS,
    }

    for attempt in range(config.SOLVER_MAX_RETRIES):
        try:
            async with sem:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=config.SOLVER_TIMEOUT)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            message = data["choices"][0]["message"]
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or ""

            # Combine thinking + content for full trace
            if reasoning:
                full_response = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                full_response = content

            extracted, method = extract_model_answer(content)

            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            return {
                **question,
                "model_response": full_response,
                "model_content": content,
                "extracted_answer": extracted,
                "extraction_method": method,
                "completion_tokens": completion_tokens,
                "correct": extracted == question["ground_truth_answer"],
            }

        except Exception as e:
            if attempt < config.SOLVER_MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"  [FAIL] id={question['id']}: {e}")
                return None


async def solve(config: Config, split: str = "both") -> None:
    """Run solver on all questions with async concurrency."""
    # Load questions
    questions = []
    with open(config.QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                if split == "both" or q["split"] == split:
                    questions.append(q)

    # Resumability
    completed = load_completed(config.TRACES_PATH)
    remaining = [q for q in questions if q["id"] not in completed]
    print(f"Total: {len(questions)}, done: {len(completed)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    config.TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(config.SOLVER_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=config.SOLVER_CONCURRENCY)

    done = 0
    correct = 0
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [solve_one(session, sem, q, config) for q in remaining]

        with open(config.TRACES_PATH, "a") as f:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is None:
                    continue

                f.write(json.dumps(result) + "\n")
                f.flush()

                done += 1
                if result["correct"]:
                    correct += 1

                if done % 50 == 0 or done == len(remaining):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(
                        f"[{done}/{len(remaining)}] "
                        f"acc={correct}/{done} ({100*correct/done:.1f}%) "
                        f"rate={rate:.1f}q/s"
                    )

    print(f"\nDone: {correct} correct out of {done} ({100*correct/done:.1f}%)")
