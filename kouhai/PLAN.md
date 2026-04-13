# Kouhai — Build Plan

## What is Kouhai

A **Python-syntax systems language** that compiles to LLVM IR. It is NOT a Python transpiler — it uses Python syntax but has its own distinct semantics. Fixes the "personality disorder" of its predecessor `senpai` by having a clear identity: Python syntax, systems semantics, no GC, no Python stdlib pretense.

- File extension: `.kou`
- Target: arm64-apple-macosx (Mac Mini M4)
- Dependencies: Python 3.10+ stdlib only + `clang`

## Key Improvements Over Senpai

1. **IR builder abstraction layer** (`ir_builder.py`) — no more raw string concatenation in codegen
2. **Separate `runtime/runtime.ll`** — runtime helpers are a real linkable IR file, not embedded strings in the compiler
3. **Clear language identity** — systems lang with Python syntax, explicitly not Python
4. **`module_programs`, `module_info` on `Program` dataclass** — no more dynamic attribute assignment

## Already Written

- `kouhai/CLAUDE.md` — project docs
- `kouhai/kouhai/tokens.py` — lexer (identical logic to senpai, just renamed)
- `kouhai/kouhai/ast_nodes.py` — AST nodes (same as senpai but with `module_programs` and `module_info` fields on `Program` dataclass, avoiding the dynamic attribute hack)

## Files Still To Write

### `kouhai/kouhai/__init__.py`
Empty file.

### `kouhai/kouhai/parser.py`
Copy of `senpai/senpai/parser.py` with:
- All imports changed from `.tokens` / `.ast_nodes` to local equivalents (already same names)
- No logic changes needed — the parser is clean

### `kouhai/kouhai/types.py`
Copy of `senpai/senpai/types.py` with:
- `prog.module_programs` and `prog.module_info` accessed directly (no `hasattr` needed since they're now real fields on `Program`)
- No other logic changes

### `kouhai/kouhai/ir_builder.py` ← KEY NEW FILE

This is the main architectural improvement. Provides a clean API over LLVM IR text emission.

```python
class IRModule:
    """Top-level IR module. Collects globals, type decls, function definitions."""
    def declare(self, name, ret_type, param_types, vararg=False)
    def define(self, name, ret_type, params: list[tuple[str,str]], linkage="") -> IRFunction
    def global_string(self, s: str) -> str        # returns "@.str.N", emits global
    def global_array(self, name, elem_type, values: list[str]) -> str
    def type_def(self, name, fields: list[str])   # %struct.Foo = type { ... }
    def raw_global(self, line: str)               # escape hatch for manual globals
    def emit(self) -> str                         # produce final IR text

class IRFunction:
    """Builds a single LLVM IR function. Tracks current block, SSA names."""
    # Called from IRModule.define()
    def new_block(self, name="") -> str           # creates block, returns label name
    def set_block(self, label: str)               # switch current insertion block
    def current_block(self) -> str

    # SSA helpers
    def _tmp(self) -> str                         # returns "%t1", "%t2", ...
    def _label(self, prefix="L") -> str           # returns "L1", "L2", ...

    # Arithmetic (int)
    def add(self, ty, a, b) -> str
    def sub(self, ty, a, b) -> str
    def mul(self, ty, a, b) -> str
    def sdiv(self, ty, a, b) -> str
    def udiv(self, ty, a, b) -> str
    def srem(self, ty, a, b) -> str
    def urem(self, ty, a, b) -> str
    def and_(self, ty, a, b) -> str
    def or_(self, ty, a, b) -> str
    def xor(self, ty, a, b) -> str
    def shl(self, ty, a, b) -> str
    def lshr(self, ty, a, b) -> str
    def ashr(self, ty, a, b) -> str

    # Arithmetic (float)
    def fadd(self, ty, a, b) -> str
    def fsub(self, ty, a, b) -> str
    def fmul(self, ty, a, b) -> str
    def fdiv(self, ty, a, b) -> str

    # Comparison
    def icmp(self, pred, ty, a, b) -> str   # pred: eq/ne/slt/sle/sgt/sge/ult/ule/ugt/uge
    def fcmp(self, pred, ty, a, b) -> str   # pred: oeq/one/olt/ole/ogt/oge

    # Memory
    def alloca(self, ty, name="") -> str
    def load(self, ty, ptr) -> str
    def store(self, ty, val, ptr)
    def gep(self, ty, ptr, *indices) -> str  # getelementptr inbounds

    # Conversions
    def trunc(self, from_ty, val, to_ty) -> str
    def zext(self, from_ty, val, to_ty) -> str
    def sext(self, from_ty, val, to_ty) -> str
    def fptrunc(self, from_ty, val, to_ty) -> str
    def fpext(self, from_ty, val, to_ty) -> str
    def fptosi(self, from_ty, val, to_ty) -> str
    def fptoui(self, from_ty, val, to_ty) -> str
    def sitofp(self, from_ty, val, to_ty) -> str
    def uitofp(self, from_ty, val, to_ty) -> str
    def ptrtoint(self, val, to_ty) -> str
    def inttoptr(self, val) -> str

    # Control flow
    def ret(self, ty="void", val=None)
    def br(self, label)
    def cond_br(self, cond, true_label, false_label)
    def phi(self, ty, *incoming: tuple[str,str]) -> str  # incoming = (val, label) pairs
    def select(self, cond, ty, true_val, false_val) -> str

    # Calls
    def call(self, ret_ty, fn_ref, args: list[str], arg_types: list[str]=None) -> str
    def call_indirect(self, ret_ty, fn_ptr, fn_type, args: list[str]) -> str

    # Misc
    def unreachable(self)
    def emit(self) -> str
```

### `kouhai/kouhai/codegen.py`

Same logic as `senpai/senpai/codegen.py` but:
- Uses `IRModule` / `IRFunction` from `ir_builder.py` instead of raw `self._emit(f"...")`
- All `self._tmp()`, `self._label()` calls become `fn._tmp()`, `fn._label()`
- All `self._emit(f"  %t = add i64 {a}, {b}")` become `fn.add("i64", a, b)`
- `_emit_runtime_functions()` is DELETED — runtime is now in `runtime/runtime.ll`
- The compiler links `runtime.ll` separately
- Function prefix stays `kouhai_` (rename from `senpai_`)
- `_method_fn_name` returns `@kouhai_{class}_{method}`
- `_struct_name` stays `%struct.{name}`

### `kouhai/runtime/runtime.ll`

Extracted directly from `senpai/senpai/codegen.py`'s `_emit_runtime_functions()`.

Contains (all as static LLVM IR, no Python generation):
- C stdlib declares: `malloc`, `realloc`, `free`, `memcpy`, `strlen`, `printf`, `puts`, `snprintf`, `memcmp`, `strstr`
- LLVM intrinsic declares: `llvm.sqrt.f64`, `llvm.sin.f64`, etc.
- `_rt_str_concat(ptr, ptr) -> ptr`
- `_rt_print_i64(i64)`, `_rt_print_u64(i64)`, `_rt_print_double(double)`, `_rt_print_bool(i1)`, `_rt_print_str(ptr)`, `_rt_print_ptr(ptr)`
- `_rt_i64_to_str(i64) -> ptr`, `_rt_u64_to_str(i64) -> ptr`, `_rt_double_to_str(double) -> ptr`, `_rt_bool_to_str(i1) -> ptr`
- `_rt_array_new(i64) -> ptr`, `_rt_array_ensure_cap(ptr, i64)`
- `kouhai_str_len`, `kouhai_str_char_at`, `kouhai_str_substring`, `kouhai_str_starts_with`, `kouhai_str_index_of`, `kouhai_str_from_char`
- `kouhai_sqrt`, `kouhai_sin`, `kouhai_cos`, `kouhai_tan`, `kouhai_exp`, `kouhai_log`, `kouhai_log2`, `kouhai_log10`, `kouhai_floor`, `kouhai_ceil`, `kouhai_round`, `kouhai_trunc`, `kouhai_abs`
- `kouhai_pow`, `kouhai_fma`
- `kouhai_abs_i64`, `kouhai_clz`, `kouhai_ctz`, `kouhai_popcount`, `kouhai_bswap`
- `kouhai_panic`
- `%struct.Array = type { i64, i64, ptr }`

Note: runtime.ll must NOT have a `target triple` line so clang can link it with user IR.

### `kouhai/kouhai/compiler.py`

Copy of `senpai/senpai/compiler.py` with:
- Import from `.tokens`, `.parser`, `.types`, `.codegen` (same names)
- `compile_source` passes `runtime.ll` path to clang:
  ```python
  runtime_ll = Path(__file__).parent.parent / "runtime" / "runtime.ll"
  # clang args: ["clang", "-x", "ir", user.ll, str(runtime_ll), "-o", output, "-O2", "-Wno-override-module"] + link_flags
  ```
- File extension references change from `.sen` to `.kou` in comments/strings
- `run_file` unchanged

### `kouhai/scripts/run.py`

```python
#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from kouhai.compiler import compile_file, run_file, CompileError

def main():
    ap = argparse.ArgumentParser(description="Kouhai compiler")
    ap.add_argument("file", help=".kou source file")
    ap.add_argument("--ir", action="store_true", help="emit LLVM IR instead of running")
    ap.add_argument("-o", "--output", help="output binary path")
    args = ap.parse_args()
    try:
        if args.ir:
            ir = compile_file(args.file, emit_ir=True)
            print(ir)
        elif args.output:
            compile_file(args.file, output_path=args.output)
        else:
            sys.exit(run_file(args.file))
    except CompileError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### `kouhai/scripts/compile.py`

Same as run.py but always compiles to a binary (no run). Alias for `run.py -o`.

### `kouhai/examples/hello.kou`
```
fn main():
    print("Hello, World!")
```

### `kouhai/examples/factorial.kou`
```
fn factorial(n: I64) -> I64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main():
    print(factorial(10))
```

### `kouhai/examples/classes.kou`
Port of `senpai/examples/classes.sen` — Animal/Dog example with inheritance.

### `kouhai/examples/arrays.kou`
Port of `senpai/examples/arrays.sen`.

### `kouhai/examples/strings.kou`
Port of `senpai/examples/strings.sen`.

### `kouhai/examples/fib.kou`
Port of `senpai/examples/fib.sen`.

### `kouhai/tests/test_compiler.py`

Port of `senpai/tests/test_compiler.py` — standalone assert-based tests (no pytest).
Tests: lex, parse, typecheck, and end-to-end compile+run for:
- Hello world
- Arithmetic
- Functions
- Classes / inheritance
- Arrays
- Strings
- for/while loops
- Type errors (should raise CompileError)

### `kouhai/tests/Makefile`

Port of `senpai/tests/Makefile` for parallel e2e test runs.

### `kouhai/tests/cases/`

Port key test cases from `senpai/tests/cases/` with `.kou` extension.

## Type System Reference

| Kouhai | LLVM  | Notes |
|--------|-------|-------|
| I8–I64 | i8–i64 | signed |
| U8–U64 | i8–i64 | unsigned |
| Float  | float  | 32-bit, `1.0f` literal |
| Double | double | 64-bit |
| Bool   | i1     | |
| Str    | ptr    | C string |
| Ptr    | ptr    | raw pointer |
| Void   | void   | return only |
| Int    | i64    | alias for I64 |

## Codegen Conventions

- User functions: `@kouhai_<name>`
- Methods: `@kouhai_<ClassName>_<method>`
- Module functions: `@kouhai_<modname>_<name>`
- Struct types: `%struct.<ClassName>`
- Vtable globals: `@vtable.<ClassName>`
- String constants: `@.str.N`

## Class Layout

```
%struct.Dog = type { ptr, ptr, ... }   ; ptr = vtable ptr, then fields
@vtable.Dog = global [N x ptr] [ptr @kouhai_Dog_speak, ...]
```

Structs (no vtable):
```
%struct.Point = type { double, double }
```

## Array Layout

```
%struct.Array = type { i64, i64, ptr }  ; len, cap, data
```

## Phased Execution Plan (Continue Here)

### Phase 0 - Scaffold and parity baseline

Goal: make Kouhai runnable end-to-end with minimal divergence from Senpai.

- [ ] Create missing directory skeleton: `runtime/`, `scripts/`, `examples/`, `tests/cases/`
- [ ] Add ports of parser/typechecker/compiler entrypoints from Senpai
- [ ] Keep behavior identical where possible; rename only public language identifiers (`senpai_` -> `kouhai_`)
- [ ] Wire `scripts/run.py` and `scripts/compile.py`
- [ ] Confirm `examples/hello.kou` compiles and runs

Exit criteria:
- `../.venv/bin/python scripts/run.py examples/hello.kou` prints expected output
- `--ir` mode emits valid LLVM IR

### Phase 1 - IR builder introduction (architectural delta)

Goal: replace ad-hoc IR string emission with structured builder API.

- [ ] Implement `IRModule` and `IRFunction` in `kouhai/kouhai/ir_builder.py`
- [ ] Add escape hatches (`raw_global`) for transitional compatibility
- [ ] Keep generated IR semantically equivalent to current Senpai backend
- [ ] Migrate codegen in small slices (literals/exprs -> control flow -> classes -> arrays)

Exit criteria:
- Codegen no longer directly concatenates most instruction strings
- Existing simple samples (`hello`, `factorial`, arithmetic) still compile+run

### Phase 2 - Runtime extraction and linking

Goal: move runtime helpers out of Python codegen and into a standalone IR runtime.

- [ ] Create `runtime/runtime.ll` from Senpai runtime emission
- [ ] Remove `_emit_runtime_functions()` from codegen
- [ ] Update compiler to link user IR + `runtime/runtime.ll` in one clang invocation
- [ ] Ensure runtime file avoids hardcoded `target triple`

Exit criteria:
- Runtime helpers resolve through linking instead of inlined emission
- `strings`, `arrays`, and math helper examples run correctly

### Phase 3 - Test migration and confidence

Goal: establish fast feedback and regression coverage.

- [ ] Port `tests/test_compiler.py` and key e2e cases to `.kou`
- [ ] Port `tests/Makefile` and parallel case execution
- [ ] Add focused assertions for parser/typechecker behavior around module metadata fields
- [ ] Include at least one negative test per major feature area

Exit criteria:
- `../.venv/bin/python tests/test_compiler.py` passes
- `make -j16 -C tests test` passes

### Phase 4 - Language identity polish

Goal: remove "Python transpiler" ambiguity and lock naming/docs consistency.

- [ ] Audit diagnostics/messages for `.kou` and "Kouhai" wording
- [ ] Verify all generated symbol prefixes are `kouhai_`
- [ ] Ensure docs/examples avoid Python stdlib expectations
- [ ] Add one "systems semantics" example: explicit cast or bitwise ops

Exit criteria:
- No lingering `senpai` naming in user-visible outputs (except migration comments)
- Documentation and examples consistently reflect language identity

## Immediate Next 10 Tasks

1. Copy `senpai/senpai/parser.py` -> `kouhai/kouhai/parser.py`
2. Copy `senpai/senpai/types.py` -> `kouhai/kouhai/types.py` and replace dynamic module attrs with direct field access
3. Add initial `kouhai/kouhai/compiler.py` (direct port, temporary)
4. Add `kouhai/scripts/run.py`
5. Add `kouhai/scripts/compile.py`
6. Add `kouhai/examples/hello.kou`
7. Run hello path to validate pipeline before IR builder migration
8. Introduce `kouhai/kouhai/ir_builder.py` with minimal methods used by simple programs first
9. Start migrating `codegen.py` expression emission to builder API
10. Create `runtime/runtime.ll` and switch compiler link step once parity checks pass

## Risk Register

- **IR drift during migration**: builder output could subtly change semantics.  
  Mitigation: migrate incrementally and diff emitted IR on representative fixtures.
- **Runtime linkage issues**: symbol mismatch between codegen and `runtime.ll`.  
  Mitigation: lock helper names in one reference list and add smoke tests.
- **Typechecker/module metadata regressions**: replacing dynamic attrs may miss corner paths.  
  Mitigation: add parser+typechecker tests with imported module scenarios.
- **Scope creep from "improvements"**: refactors beyond parity delay baseline.  
  Mitigation: enforce parity-first rule until all phase exit criteria pass.

## Definition of Done (v1)

Kouhai v1 is done when all conditions are true:

- Parser, typechecker, compiler, and scripts exist and run from clean checkout
- Codegen uses `ir_builder.py` for core emission paths
- Runtime helpers live in `runtime/runtime.ll` and are linked by compiler
- Core examples (`hello`, `factorial`, `classes`, `arrays`, `strings`) run successfully
- Unit and e2e tests pass with `.kou` cases
- Project docs and CLI wording consistently describe Kouhai as a systems language with Python syntax
