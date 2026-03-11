# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

Uses shared venv at `../.venv` (Python 3.13):

```bash
../.venv/bin/python scripts/main.py
../.venv/bin/python scripts/main.py --config examples/debate.yaml
../.venv/bin/python scripts/main.py --topic "Tabs vs spaces" --rounds 3 --moderator
../.venv/bin/python scripts/main.py --model-a mistral --model-b llama3.2 --host http://localhost:11434
```

Install dependencies: `../.venv/bin/pip install ollama textual pyyaml`

## Architecture

TUI application where two LLMs debate a topic via Ollama, with optional moderator and human intervention.

**Data flow:** `scripts/main.py` → loads `Config` → creates `Participant` objects via factories in `arena/moderator.py` → passes them to `DebateApp` → on mount, creates `Debate` engine with `_handle_event` callback → debate runs as Textual worker, streams tokens via `on_event` → TUI routes events to `DebatePanel` widgets.

**Key design: UI-agnostic debate engine.** `Debate` knows nothing about the TUI. It emits `TurnEvent` objects (speaker, token, round info) through an async callback. The TUI consumes these and routes to panels. This means the engine could drive a web UI or CLI without changes.

**Per-participant message perspective.** `_build_messages_for(participant)` transforms shared history so the current speaker sees own messages as `role: "assistant"` and others' as `role: "user"`. This prevents the model from treating the opponent's argument as its own continuation.

**Thinking mode handling.** Some models (Qwen3) use a `thinking` field instead of `content`. `Participant.respond()` detects this and wraps thinking tokens in `<think>...</think>` tags. When thinking is enabled (`THINK = True`), `num_predict` is removed from Ollama options so thinking tokens don't starve the actual response. Thinking traces are stripped from debate history (via regex) to prevent context bloat, but preserved in log files.

**Per-turn reinforcement.** Before each debater's turn, `debate.py` injects a `[Round X of Y]` message with role reinforcement (stay on your side, focus on opponent). The moderator's latest commentary is also included as background context the debater may optionally weave in.

**Human interjections.** Queue-based, drained between turns. Framed as `[Human Audience Member]` with explicit instructions that debaters are not required to answer directly — they should incorporate it only if relevant to their position. This prevents models from getting confused about who is speaking.

## Config precedence

Defaults in `config.py` → overridden by YAML (`--config`) → overridden by CLI args. YAML uses nested keys (`participant_a.model`) that flatten to class attrs (`PARTICIPANT_A_MODEL`).

## TUI keybindings

- `p` — pause/resume debate
- `c` — add one more round (dynamically extends debate)
- `q` — quit
- `Escape` — focus input field
- `Enter` — send human interjection (queued, injected between turns)

## TUI layout

The moderator panel takes the majority of screen space (~70%), with the two debater panels compressed to the top ~30%. This reflects that the moderator's analytical commentary is the primary reading experience.

## Logging

Debate transcripts auto-saved to `logs/debate_YYYYMMDD_HHMMSS_topic.md` as markdown. Includes full text with thinking traces from all turns.

## Moderator behavior

- Activates after each A+B exchange (once per round)
- On the final round, receives an extra prompt requesting a comprehensive closing summary
- Acts as an analyst for the audience — allowed to take sides and judge which argument was stronger
- System prompt instructs third-person only (never addresses debaters directly)
- Poses follow-up questions that debaters receive as background context (not direct prompts)

## Q&A mode

After all debate rounds complete, the app enters Q&A mode. The flow depends on whether a moderator is enabled:

**With moderator (mediated):** Audience question → moderator rephrases for debaters (`qa_role="rephrase"`) → debater A responds (`qa_role="respond"`) → debater B responds → moderator summarizes both answers (`qa_role="summarize"`). Four turns per question.

**Without moderator:** Audience question → both debaters respond directly (`qa_role="respond_no_mod"`). Two turns per question.

The `_run_turn` method uses `qa_role: str | None` (not a boolean) to select the appropriate prompt for each Q&A sub-step. Audience questions are stored in history as clean `[AUDIENCE QUESTION]: "..."` entries — per-turn instructions handle role-specific framing.

Pressing `c` during Q&A exits back to debate rounds. Pressing `q` quits.

## Prompt design lessons

- LLMs default to polite/agreeable — adversarial framing needed for real debate ("dismantle logic", "never concede")
- Banning "polite filler" causes models to substitute intellectual-sounding jargon instead — also ban "jargon and abstract buzzwords"
- Explicit token budgets in system prompts cause thinking models to waste massive token counts on meta-reasoning about budgets — use vague guidance ("keep it focused") and let `num_predict` handle the hard cap
- Role confusion between moderator questions and opponent arguments is common — the per-turn reinforcement now explicitly distinguishes them
