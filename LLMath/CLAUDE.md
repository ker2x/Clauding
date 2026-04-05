# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Knowledge distillation pipeline: a teacher model (lfm2.5-thinking via vLLM) generates reasoning traces for arithmetic problems, which are filtered and used to fine-tune a student model (Qwen2.5-1.5B) with mlx-lm LoRA on Apple Silicon.

## Running the Pipeline

```bash
python scripts/generate.py                        # Stage 1: expressions → data/expressions.jsonl
python scripts/oracle.py                           # Stage 2: teacher traces → data/traces.jsonl (resumable)
python scripts/filter.py                           # Stage 3: filter correct → data/train.jsonl + data/valid.jsonl
python scripts/train.py                            # Stage 4: fine-tune student → adapters/
python scripts/train.py --resume adapters/0001000_adapters.safetensors  # Resume from checkpoint
python scripts/benchmark.py --max-per-tier 20      # Evaluate model
python scripts/chat.py                             # Interactive REPL
```

Stages are independent after their input file exists. The oracle is resumable — it skips already-processed expressions by matching on the expression string.

Generate more data: `python scripts/generate.py --append --seed 123 --count 1000`

## Architecture

All config lives in `config.py` (single `Config` dataclass). Library code in `llmath/`, thin CLI wrappers in `scripts/` that just import and call the corresponding `llmath/` module.

Data flows as JSONL files: `expressions.jsonl` → `traces.jsonl` + `errors.jsonl` → `train.jsonl` + `valid.jsonl`.

Training data is ChatML format: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. The assistant content is the full model response including `<think>` reasoning.

Each expression carries a `prompt` field (varied phrasing like "Calculate X", "Solve: X", "What is X?") set at generation time. This flows through the oracle (so the teacher sees the varied phrasing) into the training data. Old expressions without a `prompt` field fall back to `"What is {expr}?"`.

## Training

Training uses mlx-lm LoRA via subprocess — `llmath/train.py` is a thin wrapper around `mlx_lm lora`. mlx-lm handles optimizer, checkpointing, eval, and memory management natively. Adapters are saved to `adapters/` and loaded alongside the base model for inference (no merging needed).

Target device: Mac Mini M4 with 16GB unified memory.

## Distributed Oracle

The oracle distributes inference across multiple endpoints (local machines, Vast.ai GPUs, OpenRouter). `TEACHER_URLS` is a `{url: concurrency}` dict — each URL gets N worker threads. All threads pull from a shared `queue.Queue` so fast machines aren't bottlenecked by slow ones.

OpenRouter URLs are auto-detected (`"openrouter.ai" in url`) — the oracle uses the correct model name and reads `OPENROUTER_API_KEY` from `.env`.

## Conventions

- Expressions are generated as random binary trees (not linear chains), producing nested structures like `(a + b) * (c - d)`.
- Negative operands are wrapped in parens: `(-19)` to avoid parsing ambiguity.
- Teacher answer extraction has a priority chain: "The answer is X" → `\boxed{X}` → "= X" at EOL → last number in response. The `extraction_method` is stored in traces.
- **Filter only accepts strong extraction methods** (`the_answer_is`, `boxed`). Weak matches (equals at EOL, last number fallback) are rejected to avoid coincidental correct answers polluting training data.
- Responses with multiple `</think>` tags are rejected (circular reasoning).
- Wrong answers from the teacher are logged to `data/errors.jsonl`.
- Part of the `Clauding` monorepo but self-contained — all paths are relative to this directory.
