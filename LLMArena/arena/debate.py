"""Chain-of-Debate engine: question cycles, thinker isolation, streaming."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from arena.participants import Participant


@dataclass
class TurnEvent:
    """Emitted during a turn."""
    speaker: str
    token: str | None      # None for turn-start/end signals
    turn_complete: bool
    question_num: int
    is_moderator: bool
    is_human: bool = False
    is_error: bool = False


@dataclass
class HistoryEntry:
    """A message in the conversation history with speaker metadata."""
    speaker: str       # participant name, "user" for questions
    content: str
    is_user: bool      # True for user questions


class ChainDebate:
    def __init__(
        self,
        participant_a: Participant,
        participant_b: Participant,
        moderator: Participant,
        on_event: Callable[[TurnEvent], Awaitable[None]],
        topic: str = "",
        log_dir: str = "logs",
    ):
        self.a = participant_a
        self.b = participant_b
        self.moderator = moderator
        self.on_event = on_event
        self.topic = topic
        self.history: list[HistoryEntry] = []
        self._paused = asyncio.Event()
        self._paused.set()  # starts unpaused
        self._cancelled = False
        self._human_interjections: asyncio.Queue[str] = asyncio.Queue()
        self._log_dir = Path(log_dir)
        self._log_file: Path | None = None
        self._question_num = 0

    def _build_messages_for(
        self, participant: Participant,
        exclude_after: int | None = None,
    ) -> list[dict]:
        """Build chat messages from the perspective of a participant.

        Own previous messages become 'assistant', everything else becomes 'user'.
        If exclude_after is set, only include history up to that index (for
        thinker isolation within a question cycle).
        """
        entries = self.history[:exclude_after] if exclude_after is not None else self.history
        messages = []
        for entry in entries:
            if entry.is_user:
                messages.append({"role": "user", "content": entry.content})
            elif entry.speaker == participant.name:
                messages.append({"role": "assistant", "content": entry.content})
            else:
                messages.append({
                    "role": "user",
                    "content": f"[{entry.speaker}]: {entry.content}",
                })
        return messages

    def _init_log(self):
        """Create the log file for this session."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self._log_dir / f"chain_{timestamp}.md"
        with open(self._log_file, "w") as f:
            f.write("# Chain-of-Debate Session\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**{self.a.name}:** {self.a.model}\n")
            f.write(f"**{self.b.name}:** {self.b.model}\n")
            f.write(f"**{self.moderator.name}:** {self.moderator.model}\n")
            f.write("\n---\n\n")

    def _log_turn(self, speaker: str, text: str, question_num: int):
        """Append a completed turn to the log file."""
        if not self._log_file:
            return
        with open(self._log_file, "a") as f:
            f.write(f"## Q{question_num} — {speaker}\n\n")
            f.write(text)
            f.write("\n\n---\n\n")

    async def run(self):
        """Main loop: wait for questions, run question cycles."""
        self._init_log()

        # If a topic was provided, use it as the first question
        if self.topic:
            self._human_interjections.put_nowait(self.topic)

        if self._human_interjections.empty():
            await self.on_event(TurnEvent(
                speaker="System",
                token="Ask a question below to begin.",
                turn_complete=True, question_num=0, is_moderator=False,
            ))

        while not self._cancelled:
            # Wait for a question
            while self._human_interjections.empty():
                if self._cancelled:
                    break
                await asyncio.sleep(0.1)

            if self._cancelled:
                break

            question = self._human_interjections.get_nowait()
            self._question_num += 1

            # Show the question as a human event (unless it was the pre-seeded topic)
            if self._question_num > 1 or not self.topic:
                await self.on_event(TurnEvent(
                    speaker="Human", token=question, turn_complete=True,
                    question_num=self._question_num, is_moderator=False,
                    is_human=True,
                ))

            await self._run_question_cycle(question, self._question_num)

            if not self._cancelled:
                await self.on_event(TurnEvent(
                    speaker="System",
                    token="Ready for next question.",
                    turn_complete=True, question_num=self._question_num,
                    is_moderator=False,
                ))

        # Signal session complete
        msg = "Session ended."
        if self._log_file:
            msg += f" Log saved to {self._log_file}"
        await self.on_event(TurnEvent(
            speaker="System", token=msg,
            turn_complete=True, question_num=self._question_num,
            is_moderator=False,
        ))

    async def _run_question_cycle(self, question: str, question_num: int):
        """Run one full question cycle: frame → think A + think B → synthesize."""

        # 1. Add question to history
        self.history.append(HistoryEntry(
            speaker="user", content=question, is_user=True,
        ))

        # 2. Moderator frames the question and assigns roles
        await self._run_turn(self.moderator, question_num,
                             is_moderator=True, role="frame")
        if self._cancelled:
            return

        # 3. Snapshot history index BEFORE thinkers respond (for isolation)
        isolation_index = len(self.history)

        # 4. Thinker A responds (sees history up to isolation_index)
        await self._run_turn(self.a, question_num, role="think",
                             isolation_index=isolation_index)
        if self._cancelled:
            return

        # 5. Thinker B responds (sees history up to isolation_index — NOT Thinker A's response)
        await self._run_turn(self.b, question_num, role="think",
                             isolation_index=isolation_index)
        if self._cancelled:
            return

        # 6. Moderator synthesizes both responses
        await self._run_turn(self.moderator, question_num,
                             is_moderator=True, role="synthesize")

    async def _run_turn(self, participant: Participant, question_num: int,
                        is_moderator: bool = False, role: str = "",
                        isolation_index: int | None = None):
        await self._paused.wait()
        if self._cancelled:
            return

        # Signal turn start
        await self.on_event(TurnEvent(
            speaker=participant.name, token=None, turn_complete=False,
            question_num=question_num, is_moderator=is_moderator,
        ))

        # Build messages with optional isolation
        if isolation_index is not None:
            messages = self._build_messages_for(participant, exclude_after=isolation_index)
        else:
            messages = self._build_messages_for(participant)

        # Add role-specific instructions
        if role == "frame":
            messages.append({
                "role": "user",
                "content": (
                    f"[INSTRUCTION] A user asked the question above. "
                    f"Your job right now: assign roles to the two thinkers.\n\n"
                    f"1. Briefly identify the key aspects of the question.\n"
                    f"2. Assign {self.a.name} a specific role.\n"
                    f"3. Assign {self.b.name} a DELIBERATELY CONTRASTING role.\n\n"
                    f"The two roles MUST create tension or cover different ground. "
                    f"Do NOT assign two roles that will produce similar analysis. "
                    f"Good pairs: optimist vs skeptic, theorist vs practitioner, "
                    f"builder vs critic, short-term vs long-term, user vs engineer. "
                    f"If one role argues X is feasible, the other should stress why it might fail.\n\n"
                    f"Format your response EXACTLY like this:\n"
                    f"[brief question analysis]\n\n"
                    f"**{self.a.name}**, your role: [role name]. [1-2 sentence description of what to focus on].\n\n"
                    f"**{self.b.name}**, your role: [role name]. [1-2 sentence description of what to focus on].\n\n"
                    f"IMPORTANT: Stop after assigning roles. Do NOT answer the question yourself. "
                    f"Do NOT write a synthesis. The thinkers will respond separately after you."
                ),
            })
        elif role == "think":
            messages.append({
                "role": "user",
                "content": (
                    f"[INSTRUCTION] You are {participant.name}. The moderator above "
                    f"assigned {participant.name} a specific role — find it and adopt it. "
                    f"Ignore the role assigned to the other thinker. "
                    f"Respond to the user's question from your assigned perspective only. "
                    f"Be thorough and honest — you don't need to disagree with anyone."
                ),
            })
        elif role == "synthesize":
            # Find the latest user question for this cycle
            latest_question = ""
            for entry in reversed(self.history):
                if entry.is_user and entry.speaker == "user":
                    latest_question = entry.content
                    break
            messages.append({
                "role": "user",
                "content": (
                    f"[INSTRUCTION] Your two thinkers have responded above. "
                    f"Now answer this specific question from the user: \"{latest_question}\"\n\n"
                    f"Rules for your answer:\n"
                    f"1. The user does NOT know the thinkers exist. Never mention them. "
                    f"Write as if this is entirely your own analysis.\n"
                    f"2. Do NOT copy-paste or closely paraphrase the thinkers. "
                    f"Use their points as INPUT, then produce your OWN reasoning in your OWN words.\n"
                    f"3. Go BEYOND what the thinkers said: identify what they both missed, "
                    f"resolve contradictions between them, draw connections they didn't make, "
                    f"and add your own insights.\n"
                    f"4. Where the thinkers disagreed, take a clear position and explain why.\n"
                    f"5. Address the user in second person. Be detailed and substantive."
                ),
            })

        # Synthesis needs thorough output, not the default "be concise" suffix
        system_suffix = None  # default concise
        if role == "synthesize":
            system_suffix = ""  # no constraint — let it be thorough

        full_response: list[str] = []
        try:
            async for token in participant.respond(messages, system_suffix=system_suffix):
                if self._cancelled:
                    break
                await self._paused.wait()
                full_response.append(token)
                await self.on_event(TurnEvent(
                    speaker=participant.name, token=token, turn_complete=False,
                    question_num=question_num, is_moderator=is_moderator,
                ))
        except Exception as e:
            error_msg = f"Error: {e}"
            await self.on_event(TurnEvent(
                speaker=participant.name, token=error_msg, turn_complete=False,
                question_num=question_num, is_moderator=is_moderator, is_error=True,
            ))
            full_response.append(error_msg)

        text = "".join(full_response)
        # Strip thinking traces from history (keep in log)
        clean_text = re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL)
        self.history.append(HistoryEntry(
            speaker=participant.name, content=clean_text, is_user=False,
        ))
        self._log_turn(participant.name, text, question_num)

        await self.on_event(TurnEvent(
            speaker=participant.name, token=None, turn_complete=True,
            question_num=question_num, is_moderator=is_moderator,
        ))

    def inject_human_message(self, text: str):
        self._human_interjections.put_nowait(text)

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def cancel(self):
        self._cancelled = True
        self._paused.set()  # unblock if paused

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()
