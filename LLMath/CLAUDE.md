# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Knowledge distillation pipeline: a teacher model (lfm2.5-thinking via Ollama) generates reasoning traces for arithmetic problems, which are filtered and used to fine-tune a student model (Qwen2.5-0.5B) with pure PyTorch on MPS.

## Running the Pipeline

```bash
python scripts/generate.py   # Stage 1: expressions → data/expressions.jsonl
python scripts/oracle.py     # Stage 2: teacher traces → data/traces.jsonl (resumable, safe to Ctrl+C)
python scripts/filter.py     # Stage 3: filter correct → data/dataset.jsonl
python scripts/train.py      # Stage 4: fine-tune student → checkpoints/
```

Stages are independent after their input file exists. The oracle is resumable — it skips already-processed expressions by matching on the expression string.

## Architecture

All config lives in `config.py` (single `Config` dataclass). Library code in `llmath/`, thin CLI wrappers in `scripts/` that just import and call the corresponding `llmath/` module.

Data flows as JSONL files: `expressions.jsonl` → `traces.jsonl` + `errors.jsonl` → `dataset.jsonl`.

## Conventions

- Pure PyTorch training loop (no HuggingFace Trainer) — `transformers` is used only for loading model weights and tokenizer.
- Training uses Qwen's `<|im_start|>` chat template with `<think>` tags wrapping reasoning traces. Loss is masked on prompt tokens.
- Negative operands are wrapped in parens: `(-19)` to avoid parsing ambiguity.
- Teacher answer extraction has a priority chain: "The answer is X" → `\boxed{X}` → "= X" at EOL → last number in response.
- Wrong answers are logged separately to `data/errors.jsonl` for review.
- Part of the `Clauding` monorepo but self-contained — all paths are relative to this directory.
