# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

Uses shared venv at `../.venv` (Python 3.13):

```bash
../.venv/bin/python scripts/main.py
../.venv/bin/python scripts/main.py --config examples/chain.yaml
../.venv/bin/python scripts/main.py --topic "How should I structure a microservices architecture?"
../.venv/bin/python scripts/main.py --model-a mistral --model-b llama3.2 --host http://localhost:11434
```

Install dependencies: `../.venv/bin/pip install ollama textual pyyaml`

## Architecture

**Chain-of-Debate:** TUI application where a moderator orchestrates two "thinkers" via Ollama to answer user questions. The moderator dynamically assigns roles to each thinker per question, then synthesizes their responses into a unified answer. This replaces chain-of-thought with chain-of-debate.

**Data flow:** `scripts/main.py` → loads `Config` → creates `Participant` objects via factories in `arena/moderator.py` → passes them to `DebateApp` → on mount, creates `ChainDebate` engine with `_handle_event` callback → engine runs as Textual worker, streams tokens via `on_event` → TUI routes events to `DebatePanel` widgets.

**Question cycle (4 steps per question):**
1. **Frame** — Moderator decomposes the question and assigns each thinker a specific role/perspective
2. **Think A** — Thinker A responds from assigned role (isolated — doesn't see Thinker B)
3. **Think B** — Thinker B responds from assigned role (isolated — doesn't see Thinker A)
4. **Synthesize** — Moderator reads both responses and produces a unified answer for the user

**Key design: UI-agnostic engine.** `ChainDebate` knows nothing about the TUI. It emits `TurnEvent` objects through an async callback.

**Thinker isolation.** Within a question cycle, thinkers cannot see each other's responses (ensured via `exclude_after` index in `_build_messages_for`). Across questions, full history is visible so the conversation builds on itself.

**Dynamic role assignment.** The moderator's "frame" step assigns roles tailored to the question (e.g., "security researcher", "pragmatist", "domain expert"). Thinkers have generic system prompts and adopt whatever role the moderator assigns.

**Per-participant message perspective.** `_build_messages_for(participant)` transforms shared history so the current speaker sees own messages as `role: "assistant"` and others' as `role: "user"`.

**Thinking mode handling.** Some models (Qwen3) use a `thinking` field instead of `content`. `Participant.respond()` detects this and wraps thinking tokens in `<think>...</think>` tags. When thinking is enabled (`THINK = True`), `num_predict` is removed from Ollama options so thinking tokens don't starve the actual response. Thinking traces are stripped from history (via regex) to prevent context bloat, but preserved in log files.

## Config precedence

Defaults in `config.py` → overridden by YAML (`--config`) → overridden by CLI args. YAML uses nested keys (`participant_a.model`) that flatten to class attrs (`PARTICIPANT_A_MODEL`).

## TUI keybindings

- `p` — pause/resume
- `q` — quit
- `Escape` — focus input field
- `Enter` — send question

## TUI layout

The moderator panel takes the majority of screen space (~70%), with the two thinker panels compressed to the top ~30%. The moderator panel is the primary reading experience — the user talks to the moderator.

## Logging

Session transcripts auto-saved to `logs/chain_YYYYMMDD_HHMMSS.md` as markdown. Includes full text with thinking traces from all turns.

## Prompt design lessons

- Explicit token budgets in system prompts cause thinking models to waste massive token counts on meta-reasoning about budgets — use vague guidance ("keep it focused") and let `num_predict` handle the hard cap
- Thinker isolation is critical — without it, the second thinker just agrees with the first
- Dynamic role assignment produces better results than fixed analytical personalities

## Original debate mode

The original adversarial debate format is preserved on the `original-debate` branch.
