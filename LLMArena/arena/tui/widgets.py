"""Custom Textual widgets for the debate TUI."""

from textual.containers import VerticalScroll
from textual.widgets import Static


class DebatePanel(VerticalScroll):
    """Scrollable panel that accumulates debate text per turn."""

    def __init__(self, speaker_name: str, **kwargs):
        super().__init__(**kwargs)
        self.speaker_name = speaker_name
        self._current_label: Static | None = None
        self._buffer = ""
        self._in_thinking = False

    async def new_turn(self, speaker: str | None = None, round_info: str | None = None):
        """Start a new turn — create a fresh label for streaming."""
        name = speaker or self.speaker_name
        ri = f"  [dim]{round_info}[/dim]" if round_info else ""
        header = Static(f"[bold]{name}[/bold]{ri}", classes="turn-header", markup=True)
        await self.mount(header)
        self._current_label = Static("", classes="turn-text", markup=False)
        await self.mount(self._current_label)
        self._buffer = ""
        self._in_thinking = False

    async def _switch_label(self, css_class: str):
        """Switch to a new label with a different style class."""
        self._current_label = Static("", classes=css_class, markup=False)
        await self.mount(self._current_label)
        self._buffer = ""

    async def append_token(self, token: str):
        """Append a streamed token to the current turn.

        Detects think tags and switches label style.
        """
        # Check for thinking start tag
        if "<think>" in token:
            parts = token.split("<think>", 1)
            if parts[0]:
                self._buffer += parts[0]
                if self._current_label:
                    self._current_label.update(self._buffer)
            # Switch to thinking style
            self._in_thinking = True
            await self._switch_label("turn-thinking")
            remainder = parts[1]
            if remainder:
                self._buffer += remainder
                if self._current_label:
                    self._current_label.update(self._buffer)
            self.scroll_end(animate=False)
            return

        # Check for thinking end tag
        if "</think>" in token:
            parts = token.split("</think>", 1)
            if parts[0]:
                self._buffer += parts[0]
                if self._current_label:
                    self._current_label.update(self._buffer)
            # Switch back to normal style
            self._in_thinking = False
            await self._switch_label("turn-text")
            remainder = parts[1].lstrip("\n")
            if remainder:
                self._buffer += remainder
                if self._current_label:
                    self._current_label.update(self._buffer)
            self.scroll_end(animate=False)
            return

        # Normal token
        self._buffer += token
        if self._current_label:
            self._current_label.update(self._buffer)
        self.scroll_end(animate=False)

    async def finish_turn(self):
        """End the current turn with a separator."""
        await self.mount(Static("", classes="separator"))
        self._current_label = None
        self._buffer = ""
        self._in_thinking = False
        self.scroll_end(animate=False)

    async def add_human_message(self, text: str):
        """Show a human interjection."""
        await self.mount(Static("[bold cyan]Human[/bold cyan]", classes="turn-header", markup=True))
        await self.mount(Static(text, classes="human-text", markup=False))
        await self.mount(Static("", classes="separator"))
        self.scroll_end(animate=False)
