#!/usr/bin/env python3
"""Compile a Kouhai program to a binary or LLVM IR."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kouhai.compiler import compile_file, CompileError


def main():
    if len(sys.argv) < 2:
        print("Usage: compile.py <file.kou> [-o output] [--ir]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    emit_ir = "--ir" in sys.argv

    # Parse -o flag
    output = None
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 >= len(sys.argv):
            print("Error: -o requires an argument", file=sys.stderr)
            sys.exit(1)
        output = sys.argv[idx + 1]

    # Default output name: input stem (no extension)
    if output is None and not emit_ir:
        output = str(Path(path).stem)

    try:
        if emit_ir:
            ir = compile_file(path, emit_ir=True)
            if output:
                Path(output).write_text(ir)
                print(f"Wrote {output}")
            else:
                print(ir)
        else:
            compile_file(path, output_path=output)
            print(f"Wrote {output}")
    except CompileError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
