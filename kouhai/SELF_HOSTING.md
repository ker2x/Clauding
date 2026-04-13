# Kouhai Self-Hosting Plan

## Current Status: PROOF-OF-CONCEPT

The `src/*.kou` files are skeletal demonstrations, not working compiler components. Only the lexer has substantial logic; the type checker and codegen are stubs.

## What Self-Hosting Actually Means

Self-hosting means a Kouhai compiler written in Kouhai can compile itself. The process:

1. **Stage 0 (bootstrap)**: The Python compiler (`kouhai/*.py`) compiles the Kouhai-written compiler (`src/*.kou`) into a native binary.
2. **Stage 1**: That binary compiles the same `src/*.kou` source, producing a second binary.
3. **Stage 2**: The stage-1 binary compiles `src/*.kou` again. If stage-1 and stage-2 binaries produce identical output, the compiler is self-hosting.

We are nowhere near this. The `src/` files cannot currently be compiled by the Python compiler because they use runtime primitives that don't exist yet and the code has bugs.

## The Python Compiler (Reference Implementation)

| Component | File | Lines | What It Does |
|-----------|------|------:|--------------|
| Lexer | `kouhai/tokens.py` | 296 | Tokenization with INDENT/DEDENT via line-by-line processing |
| AST | `kouhai/ast_nodes.py` | 226 | Dataclass nodes for all language constructs |
| Parser | `kouhai/parser.py` | 548 | Recursive descent with precedence climbing |
| Type checker | `kouhai/types.py` | 1012 | Full type inference, class/struct/module resolution |
| IR builder | `kouhai/ir_builder.py` | 295 | Clean LLVM IR emission API |
| Code generator | `kouhai/codegen.py` | 2094 | AST -> LLVM IR via ir_builder |
| Compiler | `kouhai/compiler.py` | 190 | Orchestrator: lex -> parse -> check -> codegen -> clang |
| Runtime | `runtime/runtime.ll` | 332 | Print, string ops, array ops, math wrappers |
| **Total** | | **~4,993** | |

## What Exists in `src/` Today

| Component | File | Lines | Honest Assessment |
|-----------|------|------:|-------------------|
| Token defs | `src/token.kou` | 156 | Redundant with `src/lexer.kou` (defines same Token struct + same name lookup) |
| Lexer | `src/lexer.kou` | 429 | ~70% — tokenizes chars but **missing INDENT/DEDENT** (the hard part) |
| AST defs | `src/ast.kou` | 229 | ~90% — node classes look complete, matches Python AST |
| Parser | `src/parser.kou` | 213 | ~20% — only basic exprs/stmts, no struct/class/import/for/while/assignment/type annotations. Has bugs (wrong operator in comparison, 3-arg call to 2-param function, IntLit always 0) |
| Type checker | `src/type_checker.kou` | 69 | ~5% — pure stub. Every function returns `true`. Has copy-paste `return true` duplicates |
| Code gen | `src/codegen.kou` | 33 | ~5% — stub. Emits hardcoded `define i64 @main() { ret i64 0 }` regardless of input |
| Main CLI | `src/main.kou` | 13 | ~5% — stub. Lexes a hardcoded string, prints token count |

## Language Gaps Blocking Self-Hosting

The Kouhai-written compiler needs to manipulate strings, arrays, and symbol tables extensively. Several required features are missing or unproven:

### Must Work (used by the self-hosted source)
- `str_len()`, `str_char_at()`, `str_substring()` — UFCS string methods (exist in runtime.ll)
- `Array[T].push()`, `.get()`, `.len()` — array methods
- String comparison (`==` on Str)
- String concatenation (`+` on Str)
- Struct field access and mutation

### Must Be Added (needed but don't exist)
- **File I/O** — the compiler needs to read `.kou` source files. No `read_file()` in runtime
- **Process exit with error** — `panic()` exists but no structured error reporting
- **String-to-integer conversion** — parsing integer literals requires `atoi` equivalent
- **Integer-to-string conversion** — IR generation requires building strings from numbers
- **HashMap or equivalent** — the Python compiler uses dicts for symbol tables, class info, module info. Kouhai has no map type. Options: (a) add a HashMap to the runtime, (b) use sorted arrays with linear scan, (c) use a simple hash table struct in Kouhai
- **Dynamic dispatch or tagged unions** — the Python compiler uses `isinstance()` to distinguish AST node types. Kouhai classes have vtables but no `isinstance`. Options: (a) add a type tag field to every AST node, (b) use integer type IDs with manual dispatch

## Phased Plan

### Phase 1: Fix and Complete the Lexer
**Effort: Small (~1 session)**

The lexer is close but missing the critical INDENT/DEDENT logic. The Python lexer works line-by-line (split on newlines, count leading spaces, maintain an indent stack). The Kouhai lexer currently works character-by-character and doesn't track indentation at all.

Tasks:
- [ ] Rewrite `src/lexer.kou` to use line-by-line processing (or add indent stack to char-by-char approach)
- [ ] Add comment skipping (`#` to end of line)
- [ ] Add `0.0f` float32 literal support
- [ ] Remove redundant `src/token.kou` (merge into lexer or choose one)
- [ ] Add string escape sequence handling (currently skips escape char but doesn't translate `\n` -> char 10)
- [ ] Verify it compiles and runs with the Python compiler on a test string

Exit criteria: `src/lexer.kou` lexes `examples/hello.kou` and `examples/factorial.kou` correctly, producing the same token sequence as the Python lexer.

### Phase 2: Complete the Parser
**Effort: Medium (~2-3 sessions)**

Rewrite `src/parser.kou` to handle all Kouhai syntax. The Python parser is 548 lines; expect ~600-800 lines in Kouhai due to verbose type dispatch.

Tasks:
- [ ] Fix existing bugs (operator value extraction, IntLit value parsing, parser_at_any arity)
- [ ] Add type annotation parsing (`name: Type`)
- [ ] Add function parameter and return type parsing
- [ ] Add struct declaration parsing
- [ ] Add class declaration parsing (methods, inheritance)
- [ ] Add import/link/extern parsing
- [ ] Add for/while loop parsing
- [ ] Add assignment statement parsing (including field assignment)
- [ ] Add postfix expression parsing (method calls, field access, array indexing, `as` cast)
- [ ] Add ternary expression parsing
- [ ] Add bitwise operator precedence levels (bit_or > bit_xor > bit_and > shift > add)
- [ ] Add sizeof expression parsing

Exit criteria: `src/parser.kou` parses all files in `examples/` without error.

### Phase 3: Implement the Type Checker
**Effort: Large (~3-5 sessions)**

The Python type checker is 1012 lines and is the most complex component. It resolves types, checks assignments, handles class hierarchies, module imports, and generic arrays.

This is where **HashMap** becomes critical. The type checker maintains:
- Variable scopes (name -> type mapping)
- Function signatures (name -> params + return type)
- Class info (name -> fields, methods, parent)
- Struct info (name -> fields)
- Module info (name -> exported symbols)

**Decision needed**: How to implement maps in Kouhai?
- Option A: Add `HashMap[K,V]` to runtime.ll (new runtime primitive)
- Option B: Use `Array[Pair]` with linear scan (simple but O(n) lookup)
- Option C: Write a hash table in Kouhai using arrays + hashing

Recommendation: **Option A** for the first pass (runtime HashMap), with Option C as a later self-hosting goal.

Tasks:
- [ ] Decide on map data structure strategy
- [ ] Implement scope/environment as nested maps
- [ ] Implement expression type inference (literals, vars, binops, calls, field access, method calls)
- [ ] Implement statement checking (let, assign, return, if, while, for)
- [ ] Implement function declaration checking (param types, return type validation)
- [ ] Implement struct/class checking (field types, method signatures, inheritance)
- [ ] Implement module import resolution
- [ ] Implement type coercion rules (integer widening, int-to-float)
- [ ] Implement error reporting with line numbers

Exit criteria: `src/type_checker.kou` catches type errors in intentionally broken test files and passes valid files.

### Phase 4: Implement the Code Generator
**Effort: Very Large (~5-8 sessions)**

The Python codegen is 2094 lines. This is the hardest component to port because it generates LLVM IR text through heavy string manipulation. Every AST node type needs a code generation method.

Key challenges:
- **String building**: LLVM IR is text. Need extensive `str_concat`, integer-to-string, and string formatting
- **SSA register tracking**: Need counter-based temporary names (`%t1`, `%t2`, ...)
- **Symbol tables**: Track variable allocas, function signatures, struct layouts
- **IR structure**: Functions, basic blocks, terminators, phi nodes

Tasks:
- [ ] Implement IR builder equivalent in Kouhai (string-based module/function builders)
- [ ] Implement LLVM type mapping (Kouhai type -> LLVM type string)
- [ ] Implement literal codegen (int, float, string, bool, nil)
- [ ] Implement variable load/store codegen
- [ ] Implement binary/unary operator codegen (arithmetic, comparison, logical, bitwise)
- [ ] Implement function call codegen
- [ ] Implement control flow codegen (if/elif/else, while, for)
- [ ] Implement struct codegen (type def, field access, construction)
- [ ] Implement class codegen (vtable, method dispatch, inheritance, super)
- [ ] Implement array codegen (construction, push, get, set, len)
- [ ] Implement string codegen (concat, comparison, escape sequences in constants)
- [ ] Implement module codegen (imports, namespaced symbol access)
- [ ] Implement cast/sizeof/ternary codegen
- [ ] Implement print() built-in dispatch (type-dependent)
- [ ] Wire up main CLI to run full pipeline

Exit criteria: `src/codegen.kou` generates LLVM IR for `examples/hello.kou` that clang can compile and execute.

### Phase 5: Runtime Extensions
**Effort: Medium (~1-2 sessions)**

Add missing runtime primitives needed by the self-hosted compiler itself.

Tasks:
- [ ] Add `read_file(path: Str) -> Str` to runtime.ll
- [ ] Add `str_to_i64(s: Str) -> I64` to runtime.ll
- [ ] Add `i64_to_str(n: I64) -> Str` to runtime.ll (may already exist as `_rt_i64_to_str`)
- [ ] Add `exit(code: I64)` wrapper to runtime.ll
- [ ] Add `HashMap` if chosen in Phase 3 (or verify Kouhai-native implementation works)
- [ ] Add command-line argument access (`argc`/`argv` passthrough)

Exit criteria: The self-hosted compiler can read a file, compile it, and write output.

### Phase 6: Bootstrap
**Effort: Small-Medium (~1-2 sessions), but depends on all prior phases**

Tasks:
- [ ] Compile `src/*.kou` with the Python compiler to produce `kouhaic-stage0`
- [ ] Use `kouhaic-stage0` to compile `src/*.kou` to produce `kouhaic-stage1`
- [ ] Use `kouhaic-stage1` to compile `src/*.kou` to produce `kouhaic-stage2`
- [ ] Verify `kouhaic-stage1` and `kouhaic-stage2` produce identical IR output
- [ ] Create bootstrap script that automates this process
- [ ] Document the bootstrap procedure

Exit criteria: Stage-1 and stage-2 outputs are bit-identical. The compiler is self-hosting.

## Effort Summary

| Phase | Component | Est. Lines (Kouhai) | Sessions | Depends On |
|-------|-----------|--------------------:|---------:|------------|
| 1 | Lexer | ~500 | 1 | — |
| 2 | Parser | ~700 | 2-3 | Phase 1 |
| 3 | Type checker | ~1200 | 3-5 | Phase 2, map decision |
| 4 | Code generator | ~2500+ | 5-8 | Phase 3 |
| 5 | Runtime extensions | ~200 (LL) | 1-2 | Phase 4 |
| 6 | Bootstrap | ~100 (scripts) | 1-2 | All above |
| **Total** | | **~5,200+** | **~13-21** | |

Kouhai code is more verbose than Python (no dicts, no isinstance, no f-strings, no list comprehensions), so the self-hosted compiler will likely be 10-30% larger than the Python version.

## Key Design Decisions (To Be Made)

1. **Map data structure**: Runtime HashMap vs. Kouhai-native hash table vs. linear-scan arrays?
2. **AST node dispatch**: Type tag integers with switch-like chains vs. vtable methods?
3. **Error handling**: Panic-and-exit vs. error return codes?
4. **String building**: Repeated concat vs. buffer-based StringBuilder?

## Honest Assessment

This is a substantial project — roughly 15-20 focused sessions of work. The Python compiler is ~5,000 lines; the Kouhai version will be larger. The codegen alone (Phase 4) is nearly half the effort. None of the existing `src/` stubs save meaningful work except the AST definitions and the lexer skeleton.

The hardest part is not any single component — it's that the self-hosted compiler needs *every* Kouhai feature to work correctly, including features that are themselves being compiled. A bug in class codegen means you can't use classes in the compiler. A bug in string handling means you can't generate IR text. This circular dependency is what makes self-hosting famously difficult.

## File Inventory

### Currently Useful
- `src/ast.kou` — AST node definitions. Mostly complete, good starting point.
- `src/lexer.kou` — Character-level tokenization. Needs INDENT/DEDENT but has real logic.

### Need Complete Rewrite
- `src/parser.kou` — Only handles trivial cases, multiple bugs.
- `src/type_checker.kou` — Pure stub.
- `src/codegen.kou` — Pure stub.
- `src/main.kou` — Pure stub.

### Should Be Deleted
- `src/token.kou` — Duplicates Token struct and token_type_name() already in lexer.kou.
