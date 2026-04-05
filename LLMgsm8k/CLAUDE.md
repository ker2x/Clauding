# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GSM8K evaluation pipeline: downloads the GSM8K math benchmark, queries a vLLM-hosted model, then runs two independent LLM-as-judge evaluation passes (self-eval + smart-eval) alongside regex-based answer extraction. Three-way comparison surfaces extraction failures, judge errors, and suspect ground truth labels. Includes benchmarking scripts for context size and self-correction experiments.

## Commands

```bash
# Main pipeline
python scripts/download.py              # Stage 1: fetch parquet from HuggingFace → data/questions.jsonl
python scripts/solve.py                 # Stage 2: query vLLM, write data/traces.jsonl
python scripts/evaluate.py              # Stage 3: run both eval passes
python scripts/evaluate.py --self       # self-eval only
python scripts/evaluate.py --smart      # smart-eval only
python scripts/evaluate.py --rescore    # re-extract ratings from saved raw responses (no network)
python scripts/report.py                # Stage 4: print accuracy/disagreement report

# Benchmarks
python scripts/bench_context.py                          # sweep max_tokens 1k-8k on test split
python scripts/bench_context.py --sizes 8000,16000,32000 # custom sizes (skips already-completed)
python scripts/bench_context.py --split both             # use all 8792 questions
python scripts/bench_retry.py                            # self-correction: initial + 2 retries
python scripts/bench_retry.py --retries 4                # more retries

# Plots
python scripts/plot_context.py           # accuracy + extraction rates vs context size
python scripts/plot_retry.py             # cumulative accuracy vs retry attempts
```

## Architecture

Four sequential stages, each reading the previous stage's JSONL output:

```
questions.jsonl → traces.jsonl → evaluated.jsonl → report (stdout)
  (download)       (solve)         (evaluate)       (report)
```

- **Config** (`config.py`): single dataclass, all settings (endpoints, concurrency, paths, models). Includes `SOLVER_TOP_K` and `SOLVER_REPETITION_PENALTY` for LFM2.5-Thinking recommended params.
- **Core logic** lives in `gsm8k/` modules; `scripts/` are thin CLI wrappers that `sys.path.insert` the project root
- **Extraction** (`gsm8k/extract.py`): three functions — `extract_ground_truth` (parses `#### N`), `extract_model_answer` (5-level priority: boxed → the_answer_is → gsm8k → equals_eol → fallback), `extract_eval_rating` (strips `</think>` blocks, finds first CORRECT/INCORRECT/UNSURE)
- **Benchmarks** are standalone scripts that reuse `solve_one` from `gsm8k/solve.py` directly

## Key Design Decisions

- **Blind evaluation**: judges see only two numbers ("Answer A" / "Answer B"), no question, no reasoning, no regex result — prevents confirmation bias
- **Thinking model quirks**: LFM2.5-Thinking emits `<think>` blocks inside `content` (not `reasoning_content`). The eval rating extractor splits on the *last* `</think>` and only searches after it. The solver stores both `model_response` (full trace with thinking) and `model_content` (content field as-is)
- **Resumability**: every stage loads completed IDs from its output file and skips them. Evaluate stage writes to disk after each pass (self/smart), not at the end. Bench_context saves after each size completes.
- **Concurrency**: asyncio + aiohttp with `Semaphore`. vLLM works best with concurrent requests. Concurrency values are in config.
- **All data is JSONL**: parquet is only used to read the HuggingFace dataset, never for storage
- **Token tracking**: `completion_tokens` from vLLM API response is stored in traces for accurate token counting (not char-based estimates)
- **Model context limit**: LFM2.5-Thinking has 35K context. The retry benchmark dynamically caps `max_tokens` to stay within this limit as conversation history grows.
