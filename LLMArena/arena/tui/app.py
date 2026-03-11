"""Textual TUI app for the Chain-of-Debate Arena."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from arena.debate import ChainDebate, TurnEvent
from arena.participants import Participant
from arena.tui.widgets import DebatePanel


class DebateApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "Chain-of-Debate Arena"
    BINDINGS = [
        Binding("p", "toggle_pause", "Pause/Resume", priority=True),
        Binding("q", "quit_debate", "Quit", priority=True),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self, participant_a: Participant, participant_b: Participant,
                 moderator: Participant, config):
        super().__init__()
        self.participant_a = participant_a
        self.participant_b = participant_b
        self.moderator_participant = moderator
        self.config = config
        self.debate: ChainDebate | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        topic_text = (
            f"[bold]Topic:[/bold] {self.config.TOPIC}"
            if self.config.TOPIC
            else "[bold]Ask a question below to begin...[/bold]"
        )
        yield Static(topic_text, id="topic-bar", markup=True)
        with Horizontal(id="debate-area"):
            yield DebatePanel(self.config.PARTICIPANT_A_NAME, id="panel-a")
            yield DebatePanel(self.config.PARTICIPANT_B_NAME, id="panel-b")
        yield DebatePanel(self.config.MODERATOR_NAME, id="moderator-panel")
        yield Static("Waiting for question...", id="status-label", markup=True)
        yield Input(placeholder="Ask a question (Enter to send)...", id="input-area")
        yield Footer()

    def on_mount(self):
        self.debate = ChainDebate(
            participant_a=self.participant_a,
            participant_b=self.participant_b,
            moderator=self.moderator_participant,
            on_event=self._handle_event,
            topic=self.config.TOPIC,
        )
        self._run_debate()
        # Focus input so user can type immediately
        self.query_one("#input-area", Input).focus()

    @property
    def _status(self) -> Static:
        return self.query_one("#status-label", Static)

    def _get_panel(self, event: TurnEvent) -> DebatePanel | None:
        if event.is_moderator:
            try:
                return self.query_one("#moderator-panel", DebatePanel)
            except Exception:
                return None
        if event.speaker == self.config.PARTICIPANT_A_NAME:
            return self.query_one("#panel-a", DebatePanel)
        if event.speaker == self.config.PARTICIPANT_B_NAME:
            return self.query_one("#panel-b", DebatePanel)
        return None

    async def _handle_event(self, event: TurnEvent):
        """Route events to the correct panel."""
        # Human message — show in moderator panel
        if event.is_human:
            panel = self.query_one("#moderator-panel", DebatePanel)
            await panel.add_human_message(event.token or "")
            # Update topic bar with the question
            self.query_one("#topic-bar", Static).update(
                f"[bold]Question #{event.question_num}:[/bold] {event.token}"
            )
            return

        # System message
        if event.speaker == "System":
            self._status.update(f"[bold green]{event.token}[/bold green]")
            return

        panel = self._get_panel(event)
        if not panel:
            return

        # Turn start
        if event.token is None and not event.turn_complete:
            await panel.new_turn(
                event.speaker,
                round_info=f"Q#{event.question_num}",
            )
            paused = " [bold red]PAUSED[/bold red]" if self.debate and self.debate.is_paused else ""
            self._status.update(
                f"Question #{event.question_num} | "
                f"[bold]{event.speaker}[/bold] thinking...{paused}"
            )
        # Streaming token
        elif event.token is not None:
            await panel.append_token(event.token)
        # Turn complete
        elif event.turn_complete:
            await panel.finish_turn()

    def _run_debate(self):
        """Launch the engine as a Textual worker."""
        async def _worker():
            await self.debate.run()
        self.run_worker(_worker, exclusive=True)

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if text and self.debate:
            self.debate.inject_human_message(text)
            event.input.value = ""

    def action_toggle_pause(self):
        if not self.debate:
            return
        if self.debate.is_paused:
            self.debate.resume()
            self._status.update("[bold green]Resumed[/bold green]")
        else:
            self.debate.pause()
            self._status.update("[bold red]PAUSED[/bold red] — press [bold]p[/bold] to resume")

    def action_quit_debate(self):
        if self.debate:
            self.debate.cancel()
        self.exit()

    def action_focus_input(self):
        self.query_one("#input-area", Input).focus()
