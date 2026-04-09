# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Funk** is a homoiconic stack-based programming language. Inspired by Jasmin (JVM assembler), Forth (stack-based), and Lisp (homoiconicity). Implementation in Python.

**Current state: Phase 1** — text assembler + stack-based VM. Programs are written in a line-oriented assembly format, assembled into in-memory instructions, and executed by the VM.

Future phases: S-expressions & homoiconicity, FFI via ctypes, REPL, optional LLVM backend.

## Environment & Setup

> [!IMPORTANT]
> Always use the python interpreter located in `../.venv/bin/python` for running all scripts and commands in this project.
> Example: `../.venv/bin/python scripts/run.py examples/hello.funk`

No additional dependencies beyond Python 3.10+ standard library.

## Commands

```bash
# Run a program
../.venv/bin/python scripts/run.py examples/hello.funk
../.venv/bin/python scripts/run.py examples/fib.funk
../.venv/bin/python scripts/run.py examples/factorial.funk

# Run tests
../.venv/bin/python tests/test_vm.py
```

## Architecture

```
.funk source → Assembler (two-pass) → list[Instruction] → VM (fetch-decode-execute)
```

**`funklang/opcodes.py`** — `Opcode` IntEnum and `Instruction` namedtuple. Defines the full instruction set and classifies opcodes by operand type (LABEL_OPCODES, SLOT_OPCODES, LITERAL_OPCODES).

**`funklang/assembler.py`** — Two-pass assembler. Pass 1 collects label→address mappings. Pass 2 parses instructions and resolves label references. Public API: `assemble(source: str)` and `assemble_file(path: str)`, both return `list[Instruction]`.

**`funklang/vm.py`** — Stack-based virtual machine. Data stack (shared across calls, Forth-style), call stack of Frames (return address + 256 local variable slots). `FunkError` for all runtime errors.

## Instruction Set

| Opcode | Hex | Operand | Description |
|--------|-----|---------|-------------|
| PUSH_INT | 0x01 | int | Push integer |
| PUSH_FLOAT | 0x02 | float | Push float |
| PUSH_STR | 0x03 | "string" | Push string |
| POP | 0x04 | — | Discard TOS |
| DUP | 0x05 | — | Duplicate TOS |
| SWAP | 0x06 | — | Swap top two |
| OVER | 0x07 | — | Copy second-from-top to TOS |
| ADD | 0x10 | — | a + b |
| SUB | 0x11 | — | a - b (second - top) |
| MUL | 0x12 | — | a * b |
| DIV | 0x13 | — | a / b (int//int, else float) |
| MOD | 0x14 | — | a % b |
| EQ | 0x20 | — | 1 if equal, else 0 |
| LT | 0x21 | — | 1 if a < b, else 0 |
| GT | 0x22 | — | 1 if a > b, else 0 |
| NOT | 0x23 | — | 1 if TOS falsy, else 0 |
| AND | 0x24 | — | Logical AND → 0 or 1 |
| OR | 0x25 | — | Logical OR → 0 or 1 |
| JMP | 0x30 | label | Unconditional jump |
| JMP_IF | 0x31 | label | Jump if TOS truthy (pops) |
| JMP_IF_NOT | 0x32 | label | Jump if TOS falsy (pops) |
| CALL | 0x40 | label | Push frame, jump to label |
| RET | 0x41 | — | Pop frame, return |
| LOAD | 0x50 | slot# | Push local variable |
| STORE | 0x51 | slot# | Pop into local variable |
| PRINT | 0x60 | — | Pop and print TOS |
| HALT | 0xFF | — | Stop execution |

**Operand order for binary ops:** second-from-top OP top. `PUSH_INT 10; PUSH_INT 3; SUB` → 7.

## Assembly Syntax

```asm
; comment
label:              ; label definition
  PUSH_INT 42       ; instruction with operand
  ADD               ; no-operand instruction
  PUSH_STR "hello"  ; string in double quotes
  JMP label         ; label reference
  LOAD 0            ; local variable slot
```

- Lines are case-insensitive for mnemonics
- `;` starts a comment (respects string literals)
- Labels end with `:`, referenced without `:`

## Key Semantics

- **Data stack** is shared across function calls (Forth convention). Arguments and return values stay on the stack.
- **CALL** pushes a new Frame (return address + fresh locals) onto the call stack and jumps to the target.
- **RET** pops the frame and jumps back. The data stack is untouched.
- **LOAD/STORE** access the current frame's local variable slots (0-255).
- **Truthiness:** 0 and empty string are falsy; everything else is truthy.
- **Type-aware arithmetic:** int+int→int, anything with float→float, str+str→concatenation.
- Running past the end of the program is an implicit HALT.

## Project Structure

```
funk/
├── CLAUDE.md
├── funklang/
│   ├── __init__.py
│   ├── opcodes.py      # instruction definitions
│   ├── vm.py            # virtual machine
│   └── assembler.py     # text → instructions
├── scripts/
│   └── run.py           # CLI entry point
├── examples/
│   ├── hello.funk       # Hello, World!
│   ├── fib.funk         # first 10 fibonacci numbers
│   └── factorial.funk   # 10! via recursion
└── tests/
    └── test_vm.py       # 47 tests (standalone, no pytest)
```
