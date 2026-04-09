# Funk

A stack-based programming language inspired by [Jasmin](https://jasmin.sourceforge.net/) (JVM assembler) and [Forth](https://www.forth.com/forth/). Programs are written in a line-oriented assembly format, assembled into in-memory instructions, and executed by a stack-based virtual machine.

Implementation in Python. No dependencies beyond the standard library.

## Quick Start

```bash
# Run a program
../.venv/bin/python scripts/run.py examples/hello.funk

# Run with arguments
../.venv/bin/python scripts/run.py examples/args.funk hello world 42

# Run tests
../.venv/bin/python tests/test_vm.py
```

## Hello, World!

```asm
PUSH_STR "Hello, World!"
PRINT
HALT
```

## Assembly Syntax

```asm
; This is a comment
label:                  ; label definition (jump target / function entry)
  PUSH_INT 42           ; instruction with operand
  ADD                   ; no-operand instruction
  PUSH_STR "hello"      ; string in double quotes
  JMP label             ; label reference
  STORE x               ; named variable
  LOAD x                ; recall variable
```

- One instruction per line
- `;` starts a comment (respects string literals)
- Mnemonics are case-insensitive
- Labels end with `:`, referenced without `:`

## Instruction Set

36 VM opcodes. The assembler also provides sugar aliases that expand to these.

### Stack Operations

| Instruction | Description |
|------------|-------------|
| `PUSH_INT n` | Push integer |
| `PUSH_FLOAT n` | Push float |
| `PUSH_STR "s"` | Push string |
| `POP` | Discard top of stack (TOS) |
| `DUP` | Duplicate TOS |
| `SWAP` | Swap top two elements |
| `OVER` | Copy second-from-top to TOS |
| `ROT` | Rotate top three: a b c → b c a |

### Arithmetic

All binary ops pop two values: **a** (second) and **b** (top). Result: `a OP b`.

| Instruction | Description |
|------------|-------------|
| `ADD` | a + b (int+int=int, float=float, str+str=concat) |
| `SUB` | a - b |
| `MUL` | a * b |
| `DIV` | a / b (int//int for integers, else float division) |
| `MOD` | a % b |

### Comparison & Logic

| Instruction | Description |
|------------|-------------|
| `EQ` | Push 1 if a == b, else 0 |
| `LT` | Push 1 if a < b, else 0 |
| `GT` | Push 1 if a > b, else 0 |
| `NOT` | Push 1 if TOS is falsy, else 0 |
| `AND` | Logical AND of top two → 0 or 1 |
| `OR` | Logical OR of top two → 0 or 1 |

Truthiness: `0` and `""` are falsy. Everything else is truthy.

### Control Flow

| Instruction | Description |
|------------|-------------|
| `JMP label` | Unconditional jump |
| `JMP_IF label` | Pop TOS; jump if truthy |
| `JMP_IF_NOT label` | Pop TOS; jump if falsy |

### Functions

| Instruction | Description |
|------------|-------------|
| `CALL label` | Push new frame, jump to label |
| `RET` | Pop frame, return to caller |

The **data stack is shared** across calls (Forth convention). Arguments and return values stay on the stack. Each `CALL` creates a fresh frame with its own local variables.

### Variables

| Instruction | Description |
|------------|-------------|
| `STORE name` | Pop TOS into named variable |
| `LOAD name` | Push variable value onto stack |

Variables are **named** and **function-scoped** — each call frame has its own set. No limit on number of variables.

### I/O & Type Conversion

| Instruction | Description |
|------------|-------------|
| `PRINT` | Pop and print TOS |
| `TO_INT` | Convert TOS to integer (truncates floats, parses strings) |
| `TO_FLOAT` | Convert TOS to float (promotes ints, parses strings) |
| `TO_STR` | Convert TOS to string representation |
| `CHR` | Integer → single-character string (e.g., 65 → "A") |
| `ORD` | Single-character string → integer (e.g., "A" → 65) |

### Error Handling

| Instruction | Description |
|------------|-------------|
| `TRY label` | Register catch address; on error, jump there |
| `CATCH` | Reached normally: remove catch address |

Both `TRY` (on error) and `CATCH` (on success) push an **error flag**: 1 = error occurred, 0 = no error.

```asm
TRY on_error
  PUSH_INT 10
  PUSH_INT 0
  DIV               ; error!
CATCH
POP                 ; pop error flag (0)
JMP done

on_error:
  POP               ; pop error flag (1)
  SAY "caught it!"

done:
HALT
```

TRY/CATCH nests correctly — inner handlers catch first.

### System

| Instruction | Description |
|------------|-------------|
| `HALT` | Stop execution |

Running past the end of the program is an implicit HALT.

### FFI (Foreign Function Interface)

| Instruction | Description |
|------------|-------------|
| `CALL_NATIVE name` | Call a declared native C function |

Declare native functions with the `EXTERN` directive before calling them:

```asm
EXTERN "libc" strlen (str) -> long
EXTERN "libc" abs (int) -> int
EXTERN "libc" rand () -> int
EXTERN "libc" srand (int) -> void

PUSH_STR "hello"
CALL_NATIVE strlen    ; → 5
PRINT
```

**FFI types:** `int` (c_int), `long` (c_long), `float` (c_double), `str` (c_char_p), `handle` (c_void_p), `void` (return only).

No structs, pointers, or buffers — use C shims for complex cases. Memory management is manual (Forth tradition: expose `malloc`/`free` via EXTERN if needed).

## Assembler Aliases

These are syntactic sugar — they expand to real opcodes at assembly time, with zero runtime cost.

### Short Push & Print

| Alias | Expands to |
|-------|-----------|
| `IPUSH n` | `PUSH_INT n` |
| `FPUSH n` | `PUSH_FLOAT n` |
| `SPUSH "s"` | `PUSH_STR "s"` |
| `SAY "s"` | `PUSH_STR "s"` + `PRINT` |

### Comparison Jumps

Pop two values, compare, and jump. Same operand order as the comparison ops.

| Alias | Condition |
|-------|-----------|
| `JEQ label` | Jump if a == b |
| `JNE label` | Jump if a != b |
| `JLT label` | Jump if a < b |
| `JGT label` | Jump if a > b |
| `JLE label` | Jump if a <= b |
| `JGE label` | Jump if a >= b |

### Zero Checks

Pop TOS and compare to zero.

| Alias | Condition |
|-------|-----------|
| `JZ label` | Jump if TOS == 0 |
| `JNZ label` | Jump if TOS != 0 |

## INCLUDE & Namespacing

`INCLUDE` inlines another file before assembly:

```asm
JMP main
INCLUDE "math.funk"

main:
PUSH_INT 7
CALL math.square
PRINT
HALT
```

- Path is resolved relative to the including file, then the `stdlib/` directory
- **Auto-namespacing:** labels in included files are prefixed with the filename. `square` in `math.funk` becomes `math.square`. Internal labels like `abs.done` become `math.abs.done`.
- **Global labels:** prefix a label with `.` to skip namespacing: `.shared:` stays `shared`
- Duplicate includes are silently skipped
- Use `JMP main` before `INCLUDE` to skip over library code

## Command-Line Arguments

The runner pushes arguments onto the stack before execution:

```
../.venv/bin/python scripts/run.py program.funk arg1 arg2
```

Stack at program start: `["arg1", "arg2", 2]` — args as strings, argc on top. No args → just `0`.

## Standard Library

Located in `stdlib/`. Include with `INCLUDE "filename.funk"`.

### math.funk

| Function | Stack effect | Description |
|----------|-------------|-------------|
| `math.square` | ( n -- n*n ) | Square |
| `math.abs` | ( n -- \|n\| ) | Absolute value |
| `math.max` | ( a b -- max ) | Maximum of two |
| `math.min` | ( a b -- min ) | Minimum of two |

### string.funk (FFI)

| Function | Stack effect | Description |
|----------|-------------|-------------|
| `string.length` | ( str -- int ) | String length (via libc `strlen`) |

### io.funk (FFI)

| Function | Stack effect | Description |
|----------|-------------|-------------|
| `io.getchar` | ( -- int ) | Read one char (-1 on EOF) |
| `io.putchar` | ( int -- ) | Write one char by code |
| `io.readline` | ( -- str ) | Read line from stdin (no trailing newline) |

## Tools

### TUI Debugger

```bash
../.venv/bin/python scripts/debug.py program.funk
```

Step-by-step execution with source highlighting, data stack, call stack, locals, and output panels.

| Key | Action |
|-----|--------|
| `s` / `space` | Step one instruction |
| `o` | Step over (run until same call depth) |
| `r` | Run to completion |
| `t` | Reset |
| `q` | Quit |

### TUI IDE

```bash
../.venv/bin/python scripts/ide.py program.funk
```

Text editor with integrated debugger.

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save |
| `Ctrl+R` | Run to completion |
| `Ctrl+D` | Enter debug mode |
| `Ctrl+N` | Step (debug mode) |
| `Ctrl+O` | Step over (debug mode) |
| `Escape` | Back to editor |
| `Ctrl+Q` | Quit |

## Examples

| File | Description |
|------|-------------|
| `hello.funk` | Hello, World! |
| `fib.funk` | First 10 Fibonacci numbers |
| `factorial.funk` | 10! via recursion |
| `trycatch.funk` | Error handling demo |
| `args.funk` | Command-line arguments |
| `include_demo.funk` | Using the stdlib |
| `echo.funk` | Read input, print with length (FFI demo) |
