"""Stage 3: Evaluate model answers using self-eval and smart-eval."""

import asyncio
import json
import time

import aiohttp

from config import Config
from gsm8k.extract import extract_eval_rating

EVAL_PROMPT = """\
Do these two answers match?

Answer A: {ground_truth_answer}
Answer B: {extracted_answer}

Reply with one word: CORRECT if they match, INCORRECT if they don't, or UNSURE if you can't tell."""


def load_evaluated(path) -> dict[int, dict]:
    """Load already-evaluated records keyed by ID."""
    records = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        records[data["id"]] = data
                    except (json.JSONDecodeError, KeyError):
                        continue
    return records


async def eval_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    trace: dict,
    url: str,
    model: str,
    max_tokens: int,
    timeout: int = 120,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Send one evaluation request. Returns (rating, raw_response)."""
    prompt = EVAL_PROMPT.format(
        ground_truth_answer=trace["ground_truth_answer"],
        extracted_answer=trace["extracted_answer"],
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            async with sem:
                async with session.post(
                    f"{url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            content = data["choices"][0]["message"].get("content", "") or ""
            rating = extract_eval_rating(content)
            return rating, content

        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"  [EVAL FAIL] id={trace['id']}: {e}")
                return "UNSURE", f"ERROR: {e}"


async def run_eval_pass(
    traces: list[dict],
    url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    pass_name: str,
) -> dict[int, tuple[str, str]]:
    """Run one evaluation pass. Returns {id: (rating, raw_response)}."""
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    results = {}

    done = 0
    counts = {"CORRECT": 0, "INCORRECT": 0, "UNSURE": 0}
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector) as session:

        async def _eval_with_trace(trace):
            rating, raw = await eval_one(session, sem, trace, url, model, max_tokens)
            return trace, rating, raw

        tasks = [_eval_with_trace(t) for t in traces]

        for coro in asyncio.as_completed(tasks):
            trace, rating, raw = await coro
            results[trace["id"]] = (rating, raw)

            done += 1
            counts[rating] = counts.get(rating, 0) + 1

            if done % 100 == 0 or done == len(traces):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  [{pass_name}] [{done}/{len(traces)}] "
                    f"C={counts['CORRECT']} I={counts['INCORRECT']} U={counts['UNSURE']} "
                    f"rate={rate:.1f}q/s"
                )

    return results


async def evaluate(config: Config, run_self: bool = True, run_smart: bool = True) -> None:
    """Run evaluation passes and write results."""
    # Load traces
    traces = []
    with open(config.TRACES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    # Load existing evaluations for resumability
    evaluated = load_evaluated(config.EVALUATED_PATH)

    # Figure out what still needs work
    if run_self:
        need_self = [t for t in traces if t["id"] not in evaluated or "self_eval" not in evaluated[t["id"]]]
    else:
        need_self = []

    if run_smart:
        need_smart = [t for t in traces if t["id"] not in evaluated or "smart_eval" not in evaluated[t["id"]]]
    else:
        need_smart = []

    print(f"Traces: {len(traces)}, already evaluated: {len(evaluated)}")
    print(f"Need self-eval: {len(need_self)}, need smart-eval: {len(need_smart)}")

    def _write_evaluated(evaluated, traces, config):
        """Write all evaluated records to disk."""
        for trace in traces:
            tid = trace["id"]
            if tid not in evaluated:
                evaluated[tid] = {**trace}
        config.EVALUATED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.EVALUATED_PATH, "w") as f:
            for tid in sorted(evaluated.keys()):
                f.write(json.dumps(evaluated[tid]) + "\n")

    # Run self-eval
    if need_self:
        print(f"\nRunning self-eval ({config.SELF_EVAL_MODEL})...")
        self_results = await run_eval_pass(
            need_self,
            config.SOLVER_URL,
            config.SELF_EVAL_MODEL,
            config.SELF_EVAL_CONCURRENCY,
            config.SELF_EVAL_MAX_TOKENS,
            "self",
        )
        for tid, (rating, raw) in self_results.items():
            record = evaluated.get(tid, {**{t["id"]: t for t in traces}[tid]})
            record["self_eval"] = {"rating": rating, "raw_response": raw}
            evaluated[tid] = record
        _write_evaluated(evaluated, traces, config)
        print(f"Saved self-eval results ({len(self_results)} records)")

    # Run smart-eval
    if need_smart:
        print(f"\nRunning smart-eval ({config.SMART_EVAL_MODEL})...")
        smart_results = await run_eval_pass(
            need_smart,
            config.SMART_EVAL_URL,
            config.SMART_EVAL_MODEL,
            config.SMART_EVAL_CONCURRENCY,
            config.SMART_EVAL_MAX_TOKENS,
            "smart",
        )
        for tid, (rating, raw) in smart_results.items():
            record = evaluated.get(tid, {**{t["id"]: t for t in traces}[tid]})
            record["smart_eval"] = {"rating": rating, "raw_response": raw}
            evaluated[tid] = record
        _write_evaluated(evaluated, traces, config)
        print(f"Saved smart-eval results ({len(smart_results)} records)")

    print(f"\nDone. {len(evaluated)} evaluated records in {config.EVALUATED_PATH}")


def rescore(config: Config) -> None:
    """Re-extract ratings from saved raw responses without re-querying."""
    evaluated = load_evaluated(config.EVALUATED_PATH)
    updated = 0

    for tid, record in evaluated.items():
        for eval_key in ("self_eval", "smart_eval"):
            if eval_key in record and "raw_response" in record[eval_key]:
                old = record[eval_key]["rating"]
                new = extract_eval_rating(record[eval_key]["raw_response"])
                if old != new:
                    record[eval_key]["rating"] = new
                    updated += 1

    config.EVALUATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVALUATED_PATH, "w") as f:
        for tid in sorted(evaluated.keys()):
            f.write(json.dumps(evaluated[tid]) + "\n")

    print(f"Rescored {len(evaluated)} records, {updated} ratings changed")
