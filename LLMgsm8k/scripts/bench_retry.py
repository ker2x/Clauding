#!/usr/bin/env python3
"""Benchmark accuracy with self-correction retries."""

import argparse
import asyncio
import json
import sys
import time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import aiohttp
from config import Config
from gsm8k.solve import PROMPT_TEMPLATE
from gsm8k.extract import extract_model_answer

RETRY_MESSAGE = "That answer is incorrect. Please try again, carefully rethinking the problem step by step."


async def solve_with_retries(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    question: dict,
    config: Config,
    max_attempts: int,
) -> dict:
    """Query vLLM with initial attempt + retries (max_attempts total)."""
    url = f"{config.SOLVER_URL}/v1/chat/completions"
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(question=question["question"])},
    ]

    attempts = []
    correct_at = None  # which attempt first got it right (1-indexed), None if never

    MODEL_CONTEXT_LIMIT = 35000

    for attempt in range(1, max_attempts + 1):
        # Estimate prompt tokens (~4 chars per token) and cap max_tokens to fit context
        prompt_chars = sum(len(m["content"]) for m in messages)
        estimated_prompt_tokens = prompt_chars // 3 + 200  # conservative estimate
        max_tokens = min(config.SOLVER_MAX_TOKENS, MODEL_CONTEXT_LIMIT - estimated_prompt_tokens)
        if max_tokens < 256:
            break

        payload = {
            "model": config.SOLVER_MODEL,
            "messages": messages,
            "temperature": config.SOLVER_TEMPERATURE,
            "top_k": config.SOLVER_TOP_K,
            "repetition_penalty": config.SOLVER_REPETITION_PENALTY,
            "max_tokens": max_tokens,
        }

        content = None
        for retry in range(3):
            try:
                async with sem:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.SOLVER_TIMEOUT),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

                msg = data["choices"][0]["message"]
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning_content", "") or ""
                if reasoning:
                    full = f"<think>\n{reasoning}\n</think>\n{content}"
                else:
                    full = content
                break
            except Exception as e:
                if retry < 2:
                    await asyncio.sleep(2 ** retry)
                else:
                    full = None
                    content = None

        if content is None:
            attempts.append({"attempt": attempt, "extracted": None, "correct": False})
            break

        extracted, method = extract_model_answer(content)
        is_correct = extracted == question["ground_truth_answer"]

        attempts.append({
            "attempt": attempt,
            "extracted": extracted,
            "method": method,
            "correct": is_correct,
        })

        if is_correct:
            if correct_at is None:
                correct_at = attempt
            break

        # Add assistant response + retry message for next attempt
        messages.append({"role": "assistant", "content": full})
        messages.append({"role": "user", "content": RETRY_MESSAGE})

    return {
        "id": question["id"],
        "ground_truth_answer": question["ground_truth_answer"],
        "attempts": attempts,
        "correct_at": correct_at,
        "total_attempts": len(attempts),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries", type=int, default=2, help="Number of retries after initial attempt (default: 2)")
    parser.add_argument("--split", type=str, default="test", help="test, train, or both")
    parser.add_argument("--output", type=str, default="data/bench_retry.jsonl")
    args = parser.parse_args()

    config = Config()

    questions = []
    with open(config.QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                if args.split == "both" or q["split"] == args.split:
                    questions.append(q)

    max_attempts = 1 + args.retries  # initial + retries

    print(f"Questions: {len(questions)} ({args.split})")
    print(f"Retries: {args.retries} (initial + {args.retries} = {max_attempts} total attempts)")
    print(f"Endpoint: {config.SOLVER_URL} / {config.SOLVER_MODEL}")
    print()

    sem = asyncio.Semaphore(config.SOLVER_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=config.SOLVER_CONCURRENCY)

    results = []
    done = 0
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            solve_with_retries(session, sem, q, config, max_attempts)
            for q in questions
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done += 1

            if done % 100 == 0 or done == len(questions):
                elapsed = time.time() - t0
                # Accuracy at each attempt level
                acc_by_attempt = {}
                for i in range(1, max_attempts + 1):
                    solved = sum(1 for r in results if r["correct_at"] is not None and r["correct_at"] <= i)
                    acc_by_attempt[i] = solved
                labels = {1: "initial"}
                acc_str = " | ".join(
                    f"{labels.get(i, 'retry'+str(i-1))}={acc_by_attempt[i]}"
                    for i in sorted(acc_by_attempt)
                )
                print(f"[{done}/{len(questions)}] {acc_str} rate={done/elapsed:.1f}q/s")

    # Write per-question results
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    print(f"\n=== Retry Benchmark Summary ===")
    print(f"Total: {len(results)}")

    # Initial accuracy (no retries)
    initial_correct = sum(1 for r in results if r["correct_at"] == 1)
    initial_pct = 100 * initial_correct / len(results)
    print(f"\nInitial accuracy: {initial_correct}/{len(results)} ({initial_pct:.1f}%)")

    if args.retries > 0:
        print(f"\n{'':>8}  {'cumulative_acc':>15}  {'newly_solved':>12}")
        prev = initial_correct
        for i in range(2, max_attempts + 1):
            solved = sum(1 for r in results if r["correct_at"] is not None and r["correct_at"] <= i)
            new = solved - prev
            pct = 100 * solved / len(results)
            print(f"{'Retry '+str(i-1):>8}  {solved:>8}/{len(results)} ({pct:>5.1f}%)  {'+'+str(new):>12}")
            prev = solved

    never = sum(1 for r in results if r["correct_at"] is None)
    print(f"\nNever correct: {never}/{len(results)} ({100*never/len(results):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
