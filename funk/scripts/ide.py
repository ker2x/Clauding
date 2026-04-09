#!/usr/bin/env python3
"""TUI IDE for Funk programs — editor + integrated debugger."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, TextArea
from textual.reactive import reactive
from rich.text import Text

from funklang.assembler import assemble_debug
from funklang.vm import VM, FunkError


# ── Panels ──


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


# ── App ──

MODE_EDIT = "edit"
MODE_DEBUG = "debug"


class FunkIDE(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #editor {
        width: 2fr;
        border: solid $accent;
    }
    #editor.debug-active {
        border: solid $secondary;
    }
    #source-view {
        width: 2fr;
        border: solid $success;
        overflow-y: auto;
        padding: 0 1;
        display: none;
    }
    #source-view.debug-active {
        display: block;
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
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("ctrl+r", "run", "Run", priority=True),
        Binding("ctrl+d", "debug", "Debug", priority=True),
        Binding("escape", "edit_mode", "Edit", priority=True),
        Binding("ctrl+n", "step", "Step", priority=True),
        Binding("ctrl+o", "step_over", "Over", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    mode = reactive(MODE_EDIT)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        if os.path.exists(filepath):
            with open(filepath) as f:
                self.source = f.read()
        else:
            self.source = "; New Funk program\nPUSH_INT 42\nPRINT\nHALT\n"
        self.vm: VM | None = None
        self.program = []
        self.line_map: dict[int, int] = {}
        self.source_lines: list[str] = []
        self.output_lines: list[str] = []
        self.step_count = 0
        self.dirty = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield TextArea(self.source, language="asm", id="editor")
            yield Static(id="source-view")
            with Vertical(id="sidebar"):
                yield StackPanel(id="stack")
                yield CallStackPanel(id="callstack")
                yield LocalsPanel(id="locals")
        yield OutputPanel(id="output")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._update_title()
        self._refresh_sidebar_empty()
        self._refresh_output()
        self._refresh_status()

    def on_text_area_changed(self) -> None:
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = os.path.basename(self.filepath)
        marker = " ●" if self.dirty else ""
        self.title = f"Funk IDE — {name}{marker}"

    # ── Actions ──

    def action_save(self) -> None:
        editor = self.query_one("#editor", TextArea)
        self.source = editor.text
        with open(self.filepath, "w") as f:
            f.write(self.source)
        self.dirty = False
        self._update_title()
        self._set_status("Saved.")

    def action_run(self) -> None:
        """Assemble and run to completion."""
        editor = self.query_one("#editor", TextArea)
        self.source = editor.text
        self.output_lines.clear()
        try:
            program, _ = assemble_debug(self.source)
        except FunkError as e:
            self.output_lines.append(f"ASM ERROR: {e}")
            self._refresh_output()
            self._set_status("Assembly failed.")
            return
        vm = VM(program, on_print=self._on_print)
        try:
            vm.run()
        except FunkError as e:
            self.output_lines.append(f"RUNTIME ERROR: {e}")
        self.vm = vm
        self.mode = MODE_EDIT
        self._refresh_output()
        self._refresh_sidebar_vm()
        self._set_status(f"Finished — {len(self.output_lines)} output line(s).")

    def action_debug(self) -> None:
        """Assemble and enter debug mode."""
        editor = self.query_one("#editor", TextArea)
        self.source = editor.text
        self.source_lines = self.source.splitlines()
        self.output_lines.clear()
        self.step_count = 0
        try:
            self.program, self.line_map = assemble_debug(self.source)
        except FunkError as e:
            self.output_lines.append(f"ASM ERROR: {e}")
            self._refresh_output()
            self._set_status("Assembly failed.")
            return
        self.vm = VM(list(self.program), on_print=self._on_print)
        self.mode = MODE_DEBUG
        self._enter_debug_view()

    def action_edit_mode(self) -> None:
        if self.mode == MODE_DEBUG:
            self.mode = MODE_EDIT
            self._exit_debug_view()

    def action_step(self) -> None:
        if self.mode != MODE_DEBUG or self.vm is None or self.vm.done:
            return
        self._do_step()

    def action_step_over(self) -> None:
        if self.mode != MODE_DEBUG or self.vm is None or self.vm.done:
            return
        depth = len(self.vm.call_stack)
        self._do_step()
        while self.vm and not self.vm.done and len(self.vm.call_stack) > depth:
            self._do_step()

    # ── Debug helpers ──

    def _do_step(self) -> None:
        try:
            self.vm.step()
            self.step_count += 1
        except FunkError as e:
            self.output_lines.append(f"RUNTIME ERROR: {e}")
        self._refresh_debug_view()

    def _enter_debug_view(self) -> None:
        editor = self.query_one("#editor", TextArea)
        editor.display = False
        source_view = self.query_one("#source-view", Static)
        source_view.add_class("debug-active")
        self._refresh_debug_view()

    def _exit_debug_view(self) -> None:
        editor = self.query_one("#editor", TextArea)
        editor.display = True
        editor.focus()
        source_view = self.query_one("#source-view", Static)
        source_view.remove_class("debug-active")
        self._refresh_sidebar_vm()
        self._refresh_output()
        self._refresh_status()

    def _refresh_debug_view(self) -> None:
        self._refresh_source_view()
        self._refresh_sidebar_vm()
        self._refresh_output()
        self._refresh_debug_status()

    # ── Source view (debug mode) ──

    def _refresh_source_view(self) -> None:
        current_line = self.line_map.get(self.vm.ip) if self.vm and not self.vm.done else None
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
        self.query_one("#source-view", Static).update(text)

    # ── Sidebar panels ──

    def _refresh_sidebar_empty(self) -> None:
        p = self.query_one("#stack", StackPanel)
        p.update(Text("  (no program loaded)", style="dim italic"))
        p.border_title = "Data Stack"
        p = self.query_one("#callstack", CallStackPanel)
        p.update("")
        p.border_title = "Call Stack"
        p = self.query_one("#locals", LocalsPanel)
        p.update("")
        p.border_title = "Locals"

    def _refresh_sidebar_vm(self) -> None:
        if self.vm is None:
            self._refresh_sidebar_empty()
            return
        # Stack
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
        p = self.query_one("#stack", StackPanel)
        p.update(text)
        p.border_title = f"Data Stack ({len(stack)})"

        # Call stack
        text = Text()
        for i in range(len(self.vm.call_stack) - 1, -1, -1):
            frame = self.vm.call_stack[i]
            if i == 0:
                text.append(f"  [{i}] ", style="dim cyan")
                text.append("(top-level)\n", style="dim")
            else:
                text.append(f"  [{i}] ", style="dim cyan")
                text.append(f"return → instr {frame.return_addr}\n")
        p = self.query_one("#callstack", CallStackPanel)
        p.update(text)
        p.border_title = f"Call Stack ({len(self.vm.call_stack)})"

        # Locals
        text = Text()
        frame = self.vm.call_stack[-1]
        if not frame.locals:
            text.append("  (none)", style="dim italic")
        else:
            for name, val in sorted(frame.locals.items()):
                text.append(f"  {name} ", style="dim cyan")
                text.append(f"= {val!r}\n")
        p = self.query_one("#locals", LocalsPanel)
        p.update(text)
        depth = len(self.vm.call_stack) - 1
        p.border_title = f"Locals (frame {depth})"

    # ── Output ──

    def _refresh_output(self) -> None:
        text = Text()
        if not self.output_lines:
            text.append("  (no output)", style="dim italic")
        else:
            for line in self.output_lines:
                text.append(f"  {line}\n")
        p = self.query_one("#output", OutputPanel)
        p.update(text)
        p.border_title = "Output"

    # ── Status bar ──

    def _set_status(self, msg: str) -> None:
        self.query_one("#status", StatusBar).update(f" {msg}")

    def _refresh_status(self) -> None:
        if self.mode == MODE_EDIT:
            self._set_status("EDIT  |  ^R: Run  |  ^D: Debug  |  ^S: Save")
        else:
            self._refresh_debug_status()

    def _refresh_debug_status(self) -> None:
        if self.vm is None:
            return
        if self.vm.done:
            self._set_status(f"DEBUG ■ HALTED  |  steps: {self.step_count}  |  Esc: back to editor")
        else:
            instr = self.vm.program[self.vm.ip]
            op_str = instr.opcode.name
            if instr.operand is not None:
                op_str += f" {instr.operand!r}"
            self._set_status(f"DEBUG ►  |  steps: {self.step_count}  |  ip: {self.vm.ip}  |  next: {op_str}  |  ^N: Step  ^O: Over")

    def _on_print(self, val) -> None:
        self.output_lines.append(str(val))


def main():
    if len(sys.argv) < 2:
        print("Usage: ide.py <program.funk>", file=sys.stderr)
        sys.exit(1)
    app = FunkIDE(sys.argv[1])
    app.run()


if __name__ == "__main__":
    main()
