# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Monorepo of AI/ML experiments, each in its own directory with its own `CLAUDE.md`. Always read the project-specific `CLAUDE.md` before working in a subdirectory.

## Environment

- Target hardware: Mac Mini M4 with 16GB unified memory
- Shared venv at `.venv/` (Python 3.13) — most projects use `../.venv/bin/python`
- LLMath and LLMgsm8k use their own Python (not the shared venv) and have separate `requirements.txt`
- Each project may have additional dependencies in its own `requirements.txt`

## Projects

| Directory | What | Framework |
|-----------|------|-----------|
| `001`–`006` | RL agents for CarRacing-v3 (DQN → SAC progression) | PyTorch |
| `go9x9` | AlphaZero 9x9 Go engine with GTP interface | PyTorch |
| `checkers` | AlphaZero 8x8 Checkers (bitboard engine, optional C++ MCTS) | PyTorch |
| `grokking` | Grokking on modular arithmetic | PyTorch |
| `LLMArena` | Chain-of-debate TUI via Ollama | Ollama/Textual |
| `LLMath` | Knowledge distillation: teacher→student fine-tuning for math | mlx-lm (LoRA) |
| `LLMgsm8k` | GSM8K benchmark evaluation pipeline | asyncio/aiohttp |
| `whatever` | Simulations, Metal/GPU experiments, visualizations | Mixed (Metal, JAX, WebGPU) |
| `funk` | Stack-based programming language (Jasmin/Forth-inspired VM + assembler) | Python (stdlib only) |
| `senpai` | Statically-typed OOP language compiling to LLVM IR | Python (stdlib only) + clang |

## Conventions

- **Config pattern**: each project uses a single `Config` dataclass in `config.py` for all hyperparameters
- **Code structure**: library code in a package directory (e.g. `llmath/`, `going/`, `gsm8k/`), thin CLI wrappers in `scripts/`
- **Data format**: JSONL for all data pipelines (LLMath, LLMgsm8k)
- **Tests**: standalone scripts with `assert` checks (no pytest/unittest), run directly with Python
- **Self-contained projects**: each project's paths are relative to its own directory

## Common Commands

```bash
# Go 9x9
cd go9x9 && ../.venv/bin/python scripts/train.py --iterations 100
../.venv/bin/python going/tests/test_rules.py
../.venv/bin/python going/tests/test_integration.py

# Checkers
cd checkers && ../.venv/bin/python scripts8x8/train.py --iterations 100
../.venv/bin/python checkers8x8/tests/test_integration.py

# LLMath (full pipeline)
cd LLMath
python scripts/generate.py
python scripts/oracle.py
python scripts/filter.py
python scripts/train.py
python scripts/benchmark.py --max-per-tier 20

# LLMgsm8k (full pipeline)
cd LLMgsm8k
python scripts/download.py
python scripts/solve.py
python scripts/evaluate.py
python scripts/report.py

# Grokking
cd grokking && ../.venv/bin/python main.py

# Funk
cd funk && ../.venv/bin/python scripts/run.py examples/hello.funk
../.venv/bin/python tests/test_vm.py

# Senpai
cd senpai && ../.venv/bin/python scripts/run.py examples/hello.sen
../.venv/bin/python tests/test_compiler.py
```

## Large Files

Data files and model weights are gitignored per-project. Key `.gitignore` entries:
- `LLMath/`: `data/`, `adapters/`, `models/`, `checkpoints/`
- `LLMgsm8k/`: `data/`, `data-lfm/`
- `go9x9/`, `checkers/`: `checkpoints*/`, `logs*/`
