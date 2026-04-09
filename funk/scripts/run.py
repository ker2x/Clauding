#!/usr/bin/env python3
"""Assemble and run a .funk program."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from funklang.assembler import assemble_file
from funklang.vm import VM, FunkError


def main():
    if len(sys.argv) < 2:
        print("Usage: run.py <program.funk>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    try:
        program = assemble_file(path)
        vm = VM(program)
        # Push argv: args first (in order), then argc on top
        args = sys.argv[2:]
        for arg in args:
            vm._push(arg)
        vm._push(len(args))
        vm.run()
        if vm.data_stack:
            print(f"[stack: {vm.data_stack}]")
    except FunkError as e:
        print(f"funk error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
