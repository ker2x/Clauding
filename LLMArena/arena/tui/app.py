"""Textual TUI app for the LLM Debate Arena."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from arena.debate import Debate, TurnEvent
from arena.participants import Participant
from arena.tui.widgets import DebatePanel


class DebateApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "LLM Debate Arena"
    BINDINGS = [
        Binding("p", "toggle_pause", "Pause/Resume", priority=True),
        Binding("c", "add_round", "+1 Round", priority=True),
        Binding("q", "quit_debate", "Quit", priority=True),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self, participant_a: Participant, participant_b: Participant,
                 moderator: Participant | None, config):
        super().__init__()
        self.participant_a = participant_a
        self.participant_b = participant_b
        self.moderator_participant = moderator
        self.config = config
        self.debate: Debate | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"[bold]Topic:[/bold] {self.config.TOPIC}",
            id="topic-bar", markup=True,
        )
        with Horizontal(id="debate-area"):
            yield DebatePanel(self.config.PARTICIPANT_A_NAME, id="panel-a")
            yield DebatePanel(self.config.PARTICIPANT_B_NAME, id="panel-b")
        if self.config.MODERATOR_ENABLED:
            yield DebatePanel(self.config.MODERATOR_NAME, id="moderator-panel")
        yield Static("Round -/- | Starting...", id="status-label", markup=True)
        yield Input(placeholder="Type to interject (Enter to send)...", id="input-area")
        yield Footer()

    def on_mount(self):
        self.debate = Debate(
            participant_a=self.participant_a,
            participant_b=self.participant_b,
            moderator=self.moderator_participant,
            topic=self.config.TOPIC,
            num_rounds=self.config.NUM_ROUNDS,
            on_event=self._handle_event,
        )
        self._run_debate()

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
        """Route debate events to the correct panel."""
        # Human interjection — show in both panels
        if event.is_human:
            for pid in ("#panel-a", "#panel-b"):
                panel = self.query_one(pid, DebatePanel)
                await panel.add_human_message(event.token or "")
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
            await panel.new_turn(event.speaker)
            if self.debate and getattr(self.debate, '_in_qa', False):
                self._status.update(
                    f"[bold cyan]Q&A[/bold cyan] | "
                    f"[bold]{event.speaker}[/bold] responding..."
                )
            else:
                total = self.debate.num_rounds if self.debate else self.config.NUM_ROUNDS
                paused = " [bold red]PAUSED[/bold red]" if self.debate and self.debate.is_paused else ""
                self._status.update(
                    f"Round {event.round_num}/{total} | "
                    f"[bold]{event.speaker}[/bold] speaking...{paused}"
                )
        # Streaming token
        elif event.token is not None:
            panel.append_token(event.token)
        # Turn complete
        elif event.turn_complete:
            await panel.finish_turn()

    def _run_debate(self):
        """Launch the debate as a Textual worker."""
        async def _worker():
            await self.debate.run()
        self.run_worker(_worker, exclusive=True)

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if text and self.debate:
            self.debate.inject_human_message(text)
            event.input.value = ""
            # Show immediately in both panels
            for pid in ("#panel-a", "#panel-b"):
                panel = self.query_one(pid, DebatePanel)
                await panel.add_human_message(text)

    def action_toggle_pause(self):
        if not self.debate:
            return
        if self.debate.is_paused:
            self.debate.resume()
            self._status.update(self._status.renderable.__str__().replace("PAUSED", "resumed"))
        else:
            self.debate.pause()
            self._status.update("[bold red]PAUSED[/bold red] — press [bold]p[/bold] to resume")

    def action_add_round(self):
        if not self.debate:
            return
        self.debate.add_rounds(1)
        self._status.update(
            f"[bold cyan]+1 round[/bold cyan] — now {self.debate.num_rounds} total"
        )

    def action_quit_debate(self):
        if self.debate:
            self.debate.cancel()
        self.exit()

    def action_focus_input(self):
        self.query_one("#input-area", Input).focus()
