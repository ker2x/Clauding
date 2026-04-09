# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Senpai** is a statically-typed, object-oriented programming language that compiles to LLVM IR. Python-like syntax, everything-is-an-object philosophy (Ruby/Smalltalk), explicit fixed-width types, single inheritance. Compiles via text IR emission → clang → native binary.

## Environment & Setup

> [!IMPORTANT]
> Always use the python interpreter located in `../.venv/bin/python` for running all scripts and commands in this project.

No dependencies beyond Python 3.10+ standard library. Requires `clang` (Xcode Command Line Tools).

## Commands

```bash
# Run a program (compile + execute)
../.venv/bin/python scripts/run.py examples/hello.sen

# Emit LLVM IR (for inspection)
../.venv/bin/python scripts/run.py examples/hello.sen --ir

# Run tests
../.venv/bin/python tests/test_compiler.py
```

## Architecture

```
.sen source → Lexer → Tokens → Parser → AST → TypeChecker → Typed AST → CodeGen → LLVM IR (.ll) → clang → binary
```

- **`senpai/tokens.py`** — Token enum + Lexer. Handles significant indentation (INDENT/DEDENT tokens). String escapes, two-char operators.
- **`senpai/ast_nodes.py`** — Dataclass-based AST nodes. Expressions (IntLit, BinOp, Call, etc.), Statements (LetStmt, IfStmt, WhileStmt, etc.), Declarations (FnDecl, ClassDecl).
- **`senpai/parser.py`** — Recursive descent parser with precedence climbing for operators. Produces untyped AST from token stream.
- **`senpai/types.py`** — Type system and checker. Validates types, resolves aliases (Int→I64), checks function signatures, enforces type compatibility.
- **`senpai/codegen.py`** — LLVM IR text emitter. SSA register allocation, basic blocks, function codegen. Emits IR as strings (no llvmlite dependency).
- **`senpai/compiler.py`** — Orchestrator: chains lex → parse → type-check → codegen → clang. Handles temp files, error reporting.

## Type System

| Senpai type | LLVM type | Description |
|-------------|-----------|-------------|
| `I8`–`I64` | `i8`–`i64` | Signed integers (explicit width) |
| `U8`–`U64` | `i8`–`i64` | Unsigned integers (explicit width) |
| `Float` | `float` | 32-bit IEEE 754 |
| `Double` | `double` | 64-bit IEEE 754 |
| `Bool` | `i1` | Boolean |
| `Str` | `ptr` | Heap-allocated string (concat with `+`) |
| `Void` | `void` | No value (return type only) |
| `Int` | `i64` | Alias for `I64` |

No ambiguous types — every numeric type has an explicit width.

## Key Semantics

- **Classes** with single inheritance, constructors (`__init__`), fields, methods, vtable dispatch
- **super** calls for invoking parent methods directly (no vtable dispatch)
- **Operator methods** (`__add__`, `__sub__`, `__mul__`, `__div__`, `__eq__`, `__lt__`, `__neg__`, etc.)
- **print()** is a built-in that handles all numeric types, Bool, and Str
- **to_str()** built-in method on all numeric types and Bool, returns Str
- **Type casting** via `expr as Type` for numeric conversions (int↔float, width changes)
- **for loops** via `for x in range(n):` and `for x in range(start, end):`
- **Array[T]** generic growable array with `push(val)`, `get(idx)`, `set(idx, val)`, `len()`
- **Integer/float literals** adapt to declared types (e.g., `let x: I8 = 42` works); `f` suffix for Float32
- **Functions** prefixed with `senpai_` in IR; methods as `senpai_ClassName_methodName`
- **Objects** are heap-allocated via `malloc`; struct layout: `{ vtable_ptr, fields... }`
- **Integer division** truncates (sdiv for signed, udiv for unsigned)
- **No implicit type conversions** — mismatched types are compile errors
- **Subtype compatibility** — child class instances accepted where parent type expected

## Project Structure

```
senpai/
├── CLAUDE.md
├── senpai/
│   ├── __init__.py
│   ├── tokens.py        # lexer
│   ├── ast_nodes.py     # AST definitions
│   ├── parser.py        # parser
│   ├── types.py         # type checker
│   ├── codegen.py       # LLVM IR emitter
│   └── compiler.py      # orchestrator
├── scripts/
│   └── run.py           # CLI runner
├── examples/
│   ├── hello.sen
│   ├── factorial.sen
│   └── fib.sen
└── tests/
    └── test_compiler.py # 112 tests (standalone, no pytest)
```
