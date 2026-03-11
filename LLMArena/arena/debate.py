"""Debate engine: turn management, history, streaming coordination."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from arena.participants import Participant


@dataclass
class TurnEvent:
    """Emitted during a debate turn."""
    speaker: str
    token: str | None      # None for turn-start/end signals
    turn_complete: bool
    round_num: int
    is_moderator: bool
    is_human: bool = False
    is_error: bool = False


@dataclass
class HistoryEntry:
    """A message in the debate history with speaker metadata."""
    speaker: str       # participant name, "user" for topic/human
    content: str
    is_user: bool      # True for topic prompt and human interjections


class Debate:
    def __init__(
        self,
        participant_a: Participant,
        participant_b: Participant,
        moderator: Participant | None,
        topic: str,
        num_rounds: int,
        on_event: Callable[[TurnEvent], Awaitable[None]],
        log_dir: str = "logs",
    ):
        self.a = participant_a
        self.b = participant_b
        self.moderator = moderator
        self.topic = topic
        self.num_rounds = num_rounds
        self.on_event = on_event
        self.history: list[HistoryEntry] = []
        self._paused = asyncio.Event()
        self._paused.set()  # starts unpaused
        self._cancelled = False
        self._in_qa = False
        self._rounds_added = asyncio.Event()
        self._human_interjections: asyncio.Queue[str] = asyncio.Queue()
        self._log_dir = Path(log_dir)
        self._log_file: Path | None = None

    def _build_messages_for(self, participant: Participant) -> list[dict]:
        """Build chat messages from the perspective of a participant.

        Own previous messages become 'assistant', everything else becomes 'user'.
        This gives the model a proper alternating conversation structure.
        """
        messages = []
        for entry in self.history:
            if entry.is_user:
                messages.append({"role": "user", "content": entry.content})
            elif entry.speaker == participant.name:
                messages.append({"role": "assistant", "content": entry.content})
            else:
                # Other participants' messages shown as user messages
                messages.append({
                    "role": "user",
                    "content": f"[{entry.speaker}]: {entry.content}",
                })
        return messages

    def _init_log(self):
        """Create the log file for this debate."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self.topic[:50].replace(" ", "_").replace("/", "-")
        self._log_file = self._log_dir / f"debate_{timestamp}_{slug}.md"
        with open(self._log_file, "w") as f:
            f.write(f"# Debate: {self.topic}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Rounds:** {self.num_rounds}\n")
            f.write(f"**{self.a.name}:** {self.a.model}\n")
            f.write(f"**{self.b.name}:** {self.b.model}\n")
            if self.moderator:
                f.write(f"**{self.moderator.name}:** {self.moderator.model}\n")
            f.write("\n---\n\n")

    def _log_turn(self, speaker: str, text: str, round_num: int):
        """Append a completed turn to the log file."""
        if not self._log_file:
            return
        with open(self._log_file, "a") as f:
            f.write(f"## Round {round_num} — {speaker}\n\n")
            f.write(text)
            f.write("\n\n---\n\n")

    async def run(self):
        """Main debate loop."""
        self._init_log()

        self.history.append(HistoryEntry(
            speaker="user",
            content=f"Debate topic: {self.topic}\n\nPlease present your opening argument.",
            is_user=True,
        ))

        self._current_round = 0
        while not self._cancelled:
            if self._current_round >= self.num_rounds:
                # Enter Q&A mode
                await self._qa_loop()
                if self._cancelled:
                    break
                # If we exit Q&A, more rounds were added — continue debating
                continue

            self._current_round += 1
            round_num = self._current_round

            await self._run_turn(self.a, round_num)

            if self._cancelled:
                break

            await self._run_turn(self.b, round_num)

            if self.moderator and not self._cancelled:
                is_final = (round_num == self.num_rounds)
                await self._run_turn(self.moderator, round_num,
                                     is_moderator=True, is_final_round=is_final)

        # Signal debate complete
        msg = "Debate ended."
        if self._log_file:
            msg += f" Log saved to {self._log_file}"
        await self.on_event(TurnEvent(
            speaker="System", token=msg,
            turn_complete=True, round_num=self._current_round,
            is_moderator=False,
        ))

    def add_rounds(self, n: int = 1):
        """Add extra rounds to the debate (can be called while running)."""
        self.num_rounds += n
        self._rounds_added.set()

    async def _run_turn(self, participant: Participant, round_num: int,
                        is_moderator: bool = False, is_final_round: bool = False,
                        qa_role: str | None = None):
        await self._paused.wait()
        if self._cancelled:
            return

        # Signal turn start
        await self.on_event(TurnEvent(
            speaker=participant.name, token=None, turn_complete=False,
            round_num=round_num, is_moderator=is_moderator,
        ))

        messages = self._build_messages_for(participant)

        # Add context-appropriate instructions
        if qa_role == "rephrase":
            messages.append({
                "role": "user",
                "content": (
                    "[Q&A SESSION] An audience member has asked a question "
                    "(see above). Rephrase or refine this question for the "
                    "debaters. Provide context from the debate if relevant, "
                    "then clearly state the question you want both debaters "
                    "to address."
                ),
            })
        elif qa_role == "respond":
            messages.append({
                "role": "user",
                "content": (
                    f"[Q&A SESSION] You are {participant.name}. The moderator "
                    f"has posed a question to you (see above). Answer it "
                    f"directly, staying consistent with the position you "
                    f"argued during the debate."
                ),
            })
        elif qa_role == "summarize":
            messages.append({
                "role": "user",
                "content": (
                    "[Q&A SESSION] Both debaters have responded to the "
                    "audience question. Summarize their answers, highlight "
                    "key differences, and provide your own perspective. "
                    "Address the audience directly."
                ),
            })
        elif qa_role == "respond_no_mod":
            messages.append({
                "role": "user",
                "content": (
                    f"[Q&A SESSION] You are {participant.name}. "
                    f"The debate is over. Answer the audience member's question "
                    f"directly. You may be more conversational now, but stay "
                    f"consistent with the position you argued during the debate."
                ),
            })
        elif not is_moderator:
            # Find the last moderator message if any, to reframe as context
            moderator_note = ""
            if self.moderator:
                for entry in reversed(self.history):
                    if entry.speaker == self.moderator.name:
                        moderator_note = (
                            f"\n\nThe moderator has raised the following point "
                            f"for the audience's consideration — you may weave "
                            f"this into your argument if relevant, but your "
                            f"primary focus should remain on countering your "
                            f"opponent:\n\"{entry.content}\""
                        )
                        break
            opponent = self.b if participant is self.a else self.a
            messages.append({
                "role": "user",
                "content": (
                    f"[Round {round_num} of {self.num_rounds}] "
                    f"INSTRUCTIONS: You are {participant.name}. "
                    f"Your opponent is {opponent.name}. "
                    f"Stay consistent with YOUR assigned debating position "
                    f"as described in your system prompt. "
                    f"Do NOT argue FOR the opponent's side. "
                    f"Focus on countering your opponent's latest arguments."
                    f"{moderator_note}"
                ),
            })
        elif is_final_round:
            # On final round, tell moderator to write a closing summary
            messages.append({
                "role": "user",
                "content": (
                    "This is the FINAL round of the debate. "
                    "Please provide a comprehensive closing summary: "
                    "recap the strongest arguments from each side, "
                    "identify points of agreement and disagreement, "
                    "and give your assessment of the overall debate."
                ),
            })

        full_response: list[str] = []
        try:
            async for token in participant.respond(messages):
                if self._cancelled:
                    break
                await self._paused.wait()
                full_response.append(token)
                await self.on_event(TurnEvent(
                    speaker=participant.name, token=token, turn_complete=False,
                    round_num=round_num, is_moderator=is_moderator,
                ))
        except Exception as e:
            error_msg = f"Error: {e}"
            await self.on_event(TurnEvent(
                speaker=participant.name, token=error_msg, turn_complete=False,
                round_num=round_num, is_moderator=is_moderator, is_error=True,
            ))
            full_response.append(error_msg)

        text = "".join(full_response)
        # Strip thinking traces from history so they don't bloat context
        # for subsequent turns. Keep full text (with thinking) in the log.
        clean_text = re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL)
        self.history.append(HistoryEntry(
            speaker=participant.name,
            content=clean_text,
            is_user=False,
        ))
        self._log_turn(participant.name, text, round_num)

        await self.on_event(TurnEvent(
            speaker=participant.name, token=None, turn_complete=True,
            round_num=round_num, is_moderator=is_moderator,
        ))

    async def _qa_loop(self):
        """Q&A mode after debate rounds complete. Waits for audience questions."""
        await self.on_event(TurnEvent(
            speaker="System",
            token=(
                f"Debate complete ({self._current_round} rounds). "
                f"Entering Q&A — type a question for the debaters. "
                f"Press [c] to add more debate rounds, [q] to quit."
            ),
            turn_complete=True, round_num=self._current_round,
            is_moderator=False,
        ))

        self._in_qa = True
        while not self._cancelled and self._current_round >= self.num_rounds:
            # Wait for a question or more rounds
            self._rounds_added.clear()
            while self._human_interjections.empty():
                # Check if rounds were added (exit Q&A)
                if self._current_round < self.num_rounds or self._cancelled:
                    self._in_qa = False
                    return
                await asyncio.sleep(0.1)

            if self._cancelled:
                break

            msg = self._human_interjections.get_nowait()

            # Add question to history
            self.history.append(HistoryEntry(
                speaker="Audience",
                content=f'[AUDIENCE QUESTION]: "{msg}"',
                is_user=True,
            ))
            self._log_turn("Audience Q&A", msg, self._current_round)
            await self.on_event(TurnEvent(
                speaker="Human", token=msg, turn_complete=True,
                round_num=self._current_round, is_moderator=False, is_human=True,
            ))

            if self.moderator:
                # Moderator-mediated flow
                # 1. Moderator rephrases the question
                await self._run_turn(self.moderator, self._current_round,
                                     is_moderator=True, qa_role="rephrase")
                if self._cancelled:
                    break
                # 2. Debater A responds
                await self._run_turn(self.a, self._current_round,
                                     qa_role="respond")
                if self._cancelled:
                    break
                # 3. Debater B responds
                await self._run_turn(self.b, self._current_round,
                                     qa_role="respond")
                if self._cancelled:
                    break
                # 4. Moderator summarizes
                await self._run_turn(self.moderator, self._current_round,
                                     is_moderator=True, qa_role="summarize")
            else:
                # No moderator — debaters respond directly
                await self._run_turn(self.a, self._current_round,
                                     qa_role="respond_no_mod")
                if self._cancelled:
                    break
                await self._run_turn(self.b, self._current_round,
                                     qa_role="respond_no_mod")

        self._in_qa = False

    def inject_human_message(self, text: str):
        self._human_interjections.put_nowait(text)

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def cancel(self):
        self._cancelled = True
        self._paused.set()  # unblock if paused
        self._rounds_added.set()  # unblock if waiting for rounds

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()
