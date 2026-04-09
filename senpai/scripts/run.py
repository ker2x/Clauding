#!/usr/bin/env python3
"""Run a Senpai program: compile and execute."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senpai.compiler import run_file, compile_file, CompileError


def main():
    if len(sys.argv) < 2:
        print("Usage: run.py <file.sen> [--ir]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    emit_ir = "--ir" in sys.argv

    try:
        if emit_ir:
            ir = compile_file(path, emit_ir=True)
            print(ir)
        else:
            code = run_file(path)
            sys.exit(code)
    except CompileError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
