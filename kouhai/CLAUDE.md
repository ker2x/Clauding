# CLAUDE.md

## Project Overview

**Kouhai** is a statically-typed, systems-oriented programming language with Python-like syntax that compiles to LLVM IR. It is NOT a Python transpiler — it uses Python syntax but has its own distinct semantics: explicit fixed-width types, manual memory management, single inheritance OOP, and no Python stdlib compatibility.

Lessons from senpai:
- Cleaner codegen via an IR builder abstraction layer (`ir_builder.py`)
- Runtime helpers live in `runtime/runtime.ll` (not embedded in the compiler)
- The language has a clear identity: Python syntax, systems semantics

## Environment

> [!IMPORTANT]
> Always use `../.venv/bin/python` for running all scripts and commands.

No dependencies beyond Python 3.10+ stdlib. Requires `clang` (Xcode Command Line Tools).

## Commands

```bash
# Run a program (compile + execute)
../.venv/bin/python scripts/run.py examples/hello.kou

# Emit LLVM IR (for inspection)
../.venv/bin/python scripts/run.py examples/hello.kou --ir

# Run unit tests
../.venv/bin/python tests/test_compiler.py

# Run e2e tests
make -j16 -C tests test
```

## Architecture

```
.kou source → Lexer → Tokens → Parser → AST → TypeChecker → Typed AST → CodeGen → IRBuilder → LLVM IR → clang → binary
                                                                                                    ↑
                                                                                          runtime/runtime.ll (linked)
```

- **`kouhai/tokens.py`** — Lexer. Significant indentation (INDENT/DEDENT). String escapes, two-char operators.
- **`kouhai/ast_nodes.py`** — Dataclass AST nodes.
- **`kouhai/parser.py`** — Recursive descent parser with precedence climbing.
- **`kouhai/types.py`** — Type system and checker.
- **`kouhai/ir_builder.py`** — LLVM IR builder abstraction. Clean API over raw IR text emission.
- **`kouhai/codegen.py`** — Lowers typed AST to IR using `ir_builder`. No raw string concatenation.
- **`kouhai/compiler.py`** — Orchestrator: chains all passes, invokes clang with `runtime.ll`.
- **`runtime/runtime.ll`** — LLVM IR runtime: print, array, string ops, math wrappers.

## Type System

| Kouhai type | LLVM type | Notes |
|-------------|-----------|-------|
| `I8`–`I64`  | `i8`–`i64` | Signed integers |
| `U8`–`U64`  | `i8`–`i64` | Unsigned integers |
| `Float`     | `float`    | 32-bit, use `1.0f` literal |
| `Double`    | `double`   | 64-bit |
| `Bool`      | `i1`       | |
| `Str`       | `ptr`      | C string, heap-allocated |
| `Ptr`       | `ptr`      | Raw pointer for FFI |
| `Void`      | `void`     | Return type only |
| `Int`       | `i64`      | Alias for `I64` |

## Key Semantics

Same as senpai with one clarification: **this is a systems language with Python syntax, not a Python transpiler**. No GC, no Python stdlib, no implicit conversions. Explicit width types, `malloc`-based allocation, single inheritance classes with vtable dispatch.

- `print()` polymorphic built-in for all primitive types
- `to_str()` on all numeric types and Bool
- UFCS: `x.sqrt()` == `sqrt(x)`
- `for x in range(n):` and `for x in range(start, end):`
- `Array[T]` generic growable array
- `extern fn` + `link "lib"` for FFI
- `import "file.kou"` with namespace access
- Operator overloading via `__add__`, `__eq__`, etc.
- `super.__init__(...)` for parent constructor calls
- `sizeof(Type)` → `I64`
- `nil` for null Ptr
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
- Cast: `expr as Type`
- Ternary: `value if condition else other`

## Project Structure

```
kouhai/
├── CLAUDE.md
├── kouhai/
│   ├── __init__.py
│   ├── tokens.py
│   ├── ast_nodes.py
│   ├── parser.py
│   ├── types.py
│   ├── ir_builder.py      ← key improvement over senpai
│   ├── codegen.py
│   └── compiler.py
├── runtime/
│   └── runtime.ll         ← runtime helpers (not embedded in compiler)
├── scripts/
│   ├── run.py
│   └── compile.py
├── examples/
└── tests/
    ├── test_compiler.py
    ├── Makefile
    └── cases/
```
