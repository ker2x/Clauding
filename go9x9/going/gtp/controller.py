"""
GTP I/O controller for stdin/stdout communication.

Reads GTP commands from stdin, dispatches to engine, writes responses to stdout.
Compatible with Sabaki, gogui-twogtp, KataGo, etc.
"""

import sys
from .engine import GTPEngine


class GTPController:
    """stdin/stdout GTP I/O loop."""

    def __init__(self, engine: GTPEngine, cmd_queue=None):
        self.engine = engine
        self.cmd_queue = cmd_queue

    def _read_line(self):
        """Read next line from queue (if set) or stdin."""
        if self.cmd_queue is not None:
            return self.cmd_queue.get()  # None signals EOF
        line = sys.stdin.readline()
        return line if line else None

    def run(self):
        """Main loop: read commands, dispatch, write responses."""
        while True:
            try:
                line = self._read_line()
                if line is None:
                    break  # EOF

                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines and comments


                # Parse optional command ID
                parts = line.split()
                cmd_id = None
                if parts[0].isdigit():
                    cmd_id = parts[0]

                # Handle command
                success, response = self.engine.handle_command(line)

                # Format GTP response
                if success:
                    prefix = f"={cmd_id}" if cmd_id else "="
                else:
                    prefix = f"?{cmd_id}" if cmd_id else "?"

                if response:
                    output = f"{prefix} {response}\n\n"
                else:
                    output = f"{prefix}\n\n"

                sys.stdout.write(output)
                sys.stdout.flush()

                # Check for quit
                cmd = parts[-1] if not parts[0].isdigit() else parts[1] if len(parts) > 1 else ""
                if cmd.lower() == "quit":
                    break

            except KeyboardInterrupt:
                break
            except Exception as e:
                sys.stderr.write(f"GTP error: {e}\n")
                sys.stderr.flush()
                sys.stdout.write(f"? internal error: {e}\n\n")
                sys.stdout.flush()
