#!/usr/bin/env python3
"""Benchmark accuracy across different context sizes (max_tokens)."""

import argparse
import asyncio
import json
import sys
import time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import Config
from gsm8k.solve import solve_one

import aiohttp


async def bench_one_size(questions, config, max_tokens):
    """Run all questions at a given max_tokens, return accuracy stats."""
    sem = asyncio.Semaphore(config.SOLVER_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=config.SOLVER_CONCURRENCY)

    # Override max_tokens for this run
    config_copy = Config()
    config_copy.__dict__.update(config.__dict__)
    config_copy.SOLVER_MAX_TOKENS = max_tokens

    done = 0
    correct = 0
    methods = {}
    total_completion_tokens = 0
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [solve_one(session, sem, q, config_copy) for q in questions]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                done += 1
                continue

            done += 1
            if result["correct"]:
                correct += 1

            total_completion_tokens += result.get("completion_tokens", 0)

            m = result["extraction_method"]
            methods[m] = methods.get(m, 0) + 1

            if done % 200 == 0:
                elapsed = time.time() - t0
                avg_tok = total_completion_tokens / done
                print(
                    f"  [{done}/{len(questions)}] "
                    f"acc={correct}/{done} ({100*correct/done:.1f}%) "
                    f"avg_tokens={avg_tok:.0f} "
                    f"rate={done/elapsed:.1f}q/s"
                )

    elapsed = time.time() - t0
    answered = done - (len(questions) - len([1 for _ in range(done)]))  # count non-None
    return {
        "max_tokens": max_tokens,
        "total": len(questions),
        "answered": done,
        "correct": correct,
        "accuracy": round(correct / done, 4) if done else 0,
        "rate": round(done / elapsed, 2),
        "avg_completion_tokens": round(total_completion_tokens / done) if done else 0,
        "extraction_methods": methods,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,2000,3000,4000,5000,6000,7000,8000",
                        help="Comma-separated list of max_tokens values")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to use: test, train, or both (default: test)")
    parser.add_argument("--output", type=str, default="data/bench_context.jsonl",
                        help="Output file for results")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    config = Config()

    # Load questions
    questions = []
    with open(config.QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                if args.split == "both" or q["split"] == args.split:
                    questions.append(q)

    print(f"Questions: {len(questions)} ({args.split})")
    print(f"Context sizes to test: {sizes}")
    print(f"Endpoint: {config.SOLVER_URL} / {config.SOLVER_MODEL}")
    print()

    # Load existing results to skip already-tested sizes
    from pathlib import Path
    existing = []
    existing_sizes = set()
    if Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.append(r)
                    existing_sizes.add(r["max_tokens"])
        print(f"Loaded {len(existing)} existing results, sizes: {sorted(existing_sizes)}")

    new_sizes = [s for s in sizes if s not in existing_sizes]
    if not new_sizes:
        print("All sizes already tested.")
        results = existing
    else:
        print(f"New sizes to test: {new_sizes}\n")
        results = list(existing)
        for max_tokens in new_sizes:
            print(f"=== max_tokens={max_tokens} ===")
            result = await bench_one_size(questions, config, max_tokens)
            results.append(result)
            print(
                f"  Result: {result['correct']}/{result['answered']} "
                f"({100*result['accuracy']:.1f}%) @ {result['rate']}q/s"
            )
            print()

            # Save after each size so progress isn't lost on crash
            results.sort(key=lambda r: r["max_tokens"])
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    # Summary table
    print("=== Summary ===")
    print(f"{'max_tokens':>10}  {'accuracy':>8}  {'correct':>8}  {'total':>6}  {'avg_tokens':>10}  {'rate':>6}")
    for r in results:
        avg_tok = r.get('avg_completion_tokens', r.get('avg_response_chars', 'n/a'))
        print(
            f"{r['max_tokens']:>10}  "
            f"{100*r['accuracy']:>7.1f}%  "
            f"{r['correct']:>8}  "
            f"{r['answered']:>6}  "
            f"{avg_tok:>10}  "
            f"{r['rate']:>5.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
