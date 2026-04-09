# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Funk** is a stack-based programming language. Inspired by Jasmin (JVM assembler) and Forth (stack-based). Implementation in Python.

## Environment & Setup

> [!IMPORTANT]
> Always use the python interpreter located in `../.venv/bin/python` for running all scripts and commands in this project.
> Example: `../.venv/bin/python scripts/run.py examples/hello.funk`

No additional dependencies beyond Python 3.10+ standard library. TUI debugger/IDE require `textual`.

## Commands

```bash
# Run a program
../.venv/bin/python scripts/run.py examples/hello.funk

# Run with arguments
../.venv/bin/python scripts/run.py examples/args.funk hello world

# Run tests
../.venv/bin/python tests/test_vm.py

# TUI debugger
../.venv/bin/python scripts/debug.py examples/fib.funk

# TUI IDE
../.venv/bin/python scripts/ide.py examples/fib.funk
```

## Architecture

```
.funk source → Preprocessor (INCLUDE) → Assembler (two-pass) → list[Instruction] → VM
```

- **`funklang/opcodes.py`** — `Opcode` IntEnum (36 opcodes), `Instruction` NamedTuple, `ExternDecl` NamedTuple for FFI declarations. Classifies opcodes by operand type (LABEL_OPCODES, NAME_OPCODES, LITERAL_OPCODES).
- **`funklang/assembler.py`** — Preprocessor (INCLUDE expansion, alias expansion, EXTERN parsing) + two-pass assembler. Pass 1 collects labels and EXTERN declarations. Pass 2 resolves references.
- **`funklang/vm.py`** — Stack-based VM. Data stack (shared, Forth-style), call stack of Frames (return address + named locals dict), exception stack for TRY/CATCH. FFI via ctypes for CALL_NATIVE.
- **`stdlib/`** — Standard library modules (included via `INCLUDE "math.funk"`). Labels auto-namespaced by filename. Some modules use FFI (EXTERN + CALL_NATIVE) to wrap libc functions.

## Instruction Set (36 VM opcodes)

| Opcode | Hex | Operand | Description |
|--------|-----|---------|-------------|
| PUSH_INT | 0x01 | int | Push integer |
| PUSH_FLOAT | 0x02 | float | Push float |
| PUSH_STR | 0x03 | "string" | Push string |
| POP | 0x04 | — | Discard TOS |
| DUP | 0x05 | — | Duplicate TOS |
| SWAP | 0x06 | — | Swap top two |
| OVER | 0x07 | — | Copy second-from-top to TOS |
| ROT | 0x08 | — | Rotate top three (a b c → b c a) |
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
| LOAD | 0x50 | name | Push named variable |
| STORE | 0x51 | name | Pop into named variable |
| PRINT | 0x60 | — | Pop and print TOS |
| TO_INT | 0x61 | — | Convert TOS to int |
| TO_FLOAT | 0x62 | — | Convert TOS to float |
| TO_STR | 0x63 | — | Convert TOS to string |
| CHR | 0x64 | — | Int → single-char string |
| ORD | 0x65 | — | Single-char string → int |
| TRY | 0x70 | label | Push catch address; on error jump there |
| CATCH | 0x71 | — | Pop catch address (no error); both push error flag |
| CALL_NATIVE | 0x80 | extern | Call a declared native C function via FFI |
| HALT | 0xFF | — | Stop execution |

## Assembler Aliases (sugar, no new opcodes)

| Alias | Expands to | Description |
|-------|-----------|-------------|
| IPUSH n | PUSH_INT n | Short push integer |
| FPUSH n | PUSH_FLOAT n | Short push float |
| SPUSH "s" | PUSH_STR "s" | Short push string |
| SAY "s" | PUSH_STR "s" + PRINT | Push and print string |
| JEQ label | EQ + JMP_IF | Jump if equal |
| JNE label | EQ + JMP_IF_NOT | Jump if not equal |
| JLT label | LT + JMP_IF | Jump if less than |
| JGT label | GT + JMP_IF | Jump if greater than |
| JLE label | GT + JMP_IF_NOT | Jump if less or equal |
| JGE label | LT + JMP_IF_NOT | Jump if greater or equal |
| JZ label | PUSH_INT 0 + EQ + JMP_IF | Jump if TOS is zero |
| JNZ label | PUSH_INT 0 + EQ + JMP_IF_NOT | Jump if TOS is nonzero |

## FFI (Foreign Function Interface)

- `EXTERN "lib" funcname (arg_types) -> ret_type` declares a native C function
- `CALL_NATIVE funcname` pops args, calls via ctypes, pushes result
- Types: `int` (c_int), `long` (c_long), `float` (c_double), `str` (c_char_p), `handle` (c_void_p), `void` (return only)
- Library "libc" resolves automatically via `ctypes.util.find_library("c")`
- Libraries are cached per VM instance
- No structs, no pointers, no buffers — use C shims for complex cases

## INCLUDE & Namespacing

- `INCLUDE "path"` inlines a file before assembly
- Resolves relative to the including file, then falls back to `stdlib/`
- Labels in included files are auto-prefixed with filename: `square` in `math.funk` → `math.square`
- Labels starting with `.` are global (dot stripped, no prefix)
- Duplicate includes are silently skipped (idempotent)

## Key Semantics

- **Data stack** is shared across function calls (Forth convention)
- **CALL** pushes a new Frame (return address + fresh locals) and jumps
- **RET** pops the frame and returns. Data stack is untouched
- **LOAD/STORE** access named local variables in the current frame (dict-based, unlimited)
- **TRY/CATCH** — internal exception stack. Error → jump to catch address + push 1. No error → CATCH pushes 0. Nests correctly
- **Truthiness:** 0 and empty string are falsy; everything else is truthy
- **Arithmetic:** int+int→int, anything with float→float, str+str→concatenation
- **Operand order:** second-from-top OP top. `PUSH_INT 10; PUSH_INT 3; SUB` → 7
- **Args:** runner pushes argv strings then argc onto the stack before execution
- Running past end of program is implicit HALT

## Project Structure

```
funk/
├── CLAUDE.md
├── funklang/
│   ├── __init__.py
│   ├── opcodes.py      # instruction definitions
│   ├── vm.py            # virtual machine
│   └── assembler.py     # preprocessor + assembler
├── stdlib/
│   ├── math.funk        # square, abs, max, min
│   ├── string.funk      # length (via libc strlen)
│   └── io.funk          # getchar, putchar, readline (via libc FFI)
├── scripts/
│   ├── run.py           # CLI runner
│   ├── debug.py         # TUI debugger
│   └── ide.py           # TUI IDE
├── examples/
│   ├── hello.funk
│   ├── fib.funk
│   ├── factorial.funk
│   ├── trycatch.funk
│   ├── args.funk
│   ├── include_demo.funk
│   └── echo.funk
└── tests/
    └── test_vm.py       # 107 tests (standalone, no pytest)
```
