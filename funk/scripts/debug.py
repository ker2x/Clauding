#!/usr/bin/env python3
"""TUI debugger for Funk programs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static
from rich.text import Text

from funklang.assembler import assemble_debug
from funklang.vm import VM, FunkError


class SourcePanel(Static):
    pass


class StackPanel(Static):
    pass


class CallStackPanel(Static):
    pass


class LocalsPanel(Static):
    pass


class OutputPanel(Static):
    pass


class StatusBar(Static):
    pass


class FunkDebugger(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #source {
        width: 2fr;
        border: solid $accent;
        overflow-y: auto;
        padding: 0 1;
    }
    #sidebar {
        width: 1fr;
    }
    #stack {
        height: 1fr;
        border: solid $success;
        overflow-y: auto;
        padding: 0 1;
    }
    #callstack {
        height: auto;
        max-height: 20%;
        border: solid $warning;
        padding: 0 1;
    }
    #locals {
        height: auto;
        max-height: 30%;
        border: solid #b48ead;
        overflow-y: auto;
        padding: 0 1;
    }
    #output {
        height: auto;
        max-height: 25%;
        min-height: 3;
        border: solid $secondary;
        overflow-y: auto;
        padding: 0 1;
    }
    #status {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("s", "step", "Step"),
        Binding("space", "step", "Step", show=False),
        Binding("r", "run_all", "Run"),
        Binding("o", "step_over", "Over"),
        Binding("t", "reset", "Reset"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, source: str, filename: str):
        super().__init__()
        self.source = source
        self.filename = filename
        self.source_lines = source.splitlines()
        self.program, self.line_map = assemble_debug(source)
        self.output_lines: list[str] = []
        self.vm = self._make_vm()
        self.step_count = 0

    def _make_vm(self) -> VM:
        return VM(list(self.program), on_print=self._on_print)

    def _on_print(self, val):
        self.output_lines.append(str(val))

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield SourcePanel(id="source")
            with Vertical(id="sidebar"):
                yield StackPanel(id="stack")
                yield CallStackPanel(id="callstack")
                yield LocalsPanel(id="locals")
        yield OutputPanel(id="output")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Funk Debugger — {self.filename}"
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._refresh_source()
        self._refresh_stack()
        self._refresh_callstack()
        self._refresh_locals()
        self._refresh_output()
        self._refresh_status()

    def _refresh_source(self) -> None:
        current_line = self.line_map.get(self.vm.ip)
        text = Text()
        for i, line in enumerate(self.source_lines, 1):
            lineno = f"{i:4d}  "
            if i == current_line:
                text.append("► ", style="bold green")
                text.append(lineno, style="bold green")
                text.append(line, style="bold green reverse")
            else:
                text.append("  ", style="dim")
                text.append(lineno, style="dim cyan")
                text.append(line)
            text.append("\n")
        panel = self.query_one("#source", SourcePanel)
        panel.update(text)
        panel.border_title = "Source"

    def _refresh_stack(self) -> None:
        text = Text()
        stack = self.vm.data_stack
        if not stack:
            text.append("  (empty)", style="dim italic")
        else:
            for i in range(len(stack) - 1, -1, -1):
                marker = "TOS → " if i == len(stack) - 1 else "       "
                style = "bold" if i == len(stack) - 1 else ""
                text.append(marker, style="green" if i == len(stack) - 1 else "dim")
                text.append(f"[{i}] ", style="dim cyan")
                text.append(f"{stack[i]!r}\n", style=style)
        panel = self.query_one("#stack", StackPanel)
        panel.update(text)
        panel.border_title = f"Data Stack ({len(stack)})"

    def _refresh_callstack(self) -> None:
        text = Text()
        for i in range(len(self.vm.call_stack) - 1, -1, -1):
            frame = self.vm.call_stack[i]
            if i == 0:
                text.append(f"  [{i}] ", style="dim cyan")
                text.append("(top-level)\n", style="dim")
            else:
                text.append(f"  [{i}] ", style="dim cyan")
                text.append(f"return → instr {frame.return_addr}\n")
        panel = self.query_one("#callstack", CallStackPanel)
        panel.update(text)
        panel.border_title = f"Call Stack ({len(self.vm.call_stack)})"

    def _refresh_locals(self) -> None:
        text = Text()
        if self.vm.call_stack:
            frame = self.vm.call_stack[-1]
            if not frame.locals:
                text.append("  (none)", style="dim italic")
            else:
                for name, val in sorted(frame.locals.items()):
                    text.append(f"  {name} ", style="dim cyan")
                    text.append(f"= {val!r}\n")
        panel = self.query_one("#locals", LocalsPanel)
        panel.update(text)
        depth = len(self.vm.call_stack) - 1
        panel.border_title = f"Locals (frame {depth})"

    def _refresh_output(self) -> None:
        text = Text()
        if not self.output_lines:
            text.append("  (no output yet)", style="dim italic")
        else:
            for line in self.output_lines:
                text.append(f"  {line}\n")
        panel = self.query_one("#output", OutputPanel)
        panel.update(text)
        panel.border_title = "Output"

    def _refresh_status(self) -> None:
        bar = self.query_one("#status", StatusBar)
        if self.vm.done:
            bar.update(f" ■ HALTED  |  steps: {self.step_count}  |  ip: {self.vm.ip}")
        else:
            instr = self.vm.program[self.vm.ip]
            op_str = instr.opcode.name
            if instr.operand is not None:
                op_str += f" {instr.operand!r}"
            bar.update(f" ► READY   |  steps: {self.step_count}  |  ip: {self.vm.ip}  |  next: {op_str}")

    def action_step(self) -> None:
        if self.vm.done:
            return
        try:
            self.vm.step()
            self.step_count += 1
        except FunkError as e:
            self.output_lines.append(f"ERROR: {e}")
        self._refresh_all()

    def action_run_all(self) -> None:
        if self.vm.done:
            return
        try:
            while not self.vm.done:
                self.vm.step()
                self.step_count += 1
        except FunkError as e:
            self.output_lines.append(f"ERROR: {e}")
        self._refresh_all()

    def action_step_over(self) -> None:
        """Step over a CALL: run until we return to the same call depth."""
        if self.vm.done:
            return
        depth = len(self.vm.call_stack)
        try:
            self.vm.step()
            self.step_count += 1
            while not self.vm.done and len(self.vm.call_stack) > depth:
                self.vm.step()
                self.step_count += 1
        except FunkError as e:
            self.output_lines.append(f"ERROR: {e}")
        self._refresh_all()

    def action_reset(self) -> None:
        self.vm = self._make_vm()
        self.output_lines.clear()
        self.step_count = 0
        self._refresh_all()


def main():
    if len(sys.argv) < 2:
        print("Usage: debug.py <program.funk>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(1)

    app = FunkDebugger(source, os.path.basename(path))
    app.run()


if __name__ == "__main__":
    main()
