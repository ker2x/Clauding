# Senpai Self-Hosting Roadmap

What's missing for Senpai to compile its own compiler.

## Critical Blockers

### 1. File I/O

No `open()`, `read()`, `write()`, `close()`. The compiler's first job is reading `.sen` files. Needs at minimum: read file to string, write string to file. Could be done by extending `extern` to support `Str` params and calling libc (`fopen`, `fread`, `fwrite`, `fclose`).

### 2. String Manipulation

Only `+` and `to_str()` exist today. A lexer/parser needs: `charAt(i)`, `substring(start, end)`, `length()`, `startsWith()`, `indexOf()`, char-to-int / int-to-char, and string ordering (`<`, `>`). Currently impossible to iterate over characters or inspect string contents at all.

### 3. Hash Maps / Dictionaries

No `Map[K,V]`. Symbol tables, type environments, vtable registries, module caches — the compiler is full of dicts. `Array[T]` alone can't substitute without O(n) lookups everywhere.

### 4. Enums / Tagged Unions

No way to represent AST nodes. The AST has ~30+ node types (BinaryOp, IfStmt, ClassDef...). Without enums or sum types, you'd need a class hierarchy, but even then there's no `isinstance()` or pattern matching to dispatch on node type.

### 5. Nullable / Optional Types

No `None` or `Option[T]`. Parser results, optional parent classes, optional return types — the compiler is full of `Optional[X]`. No way to represent "absence of value."

## Major Gaps

### 6. Process Execution

Can't invoke `clang`. The final compilation step shells out to clang. Need `system()` or `exec()` via extern/FFI.

### 7. User-Defined Generics

Only `Array[T]` is generic. User-defined generic classes/functions would unlock `Map[K,V]`, `Option[T]`, `Result[T,E]` as library code.

### 8. First-Class Functions / Closures

No callbacks or visitors. Tree traversal patterns (codegen, type checking) rely heavily on passing functions around or using closures.

### 9. Error Handling

No exceptions, no `Result` type. Compiler errors need structured reporting with no way to signal or recover from errors today.

### 10. Str in Extern Signatures

FFI blocks on strings. Even calling libc for file I/O is impossible since `Str` isn't supported in `extern fn` parameter/return types.

## Minor but Needed

- **String-to-number parsing** (`parseInt`, `parseFloat`)
- **Command-line argument access** (`argc`/`argv`)
- **Exit codes** (`exit(1)`)
- **Multi-line string literals** or raw strings (for emitting LLVM IR)
- **Bitwise operators** (`&`, `|`, `<<`, `>>`) — useful for flags/hashing
- **Break/continue** in loops

## Suggested Priority Path

| Step | Feature | Unlocks |
|------|---------|---------|
| 1 | Str in extern FFI | libc string functions + file I/O |
| 2 | String methods (len, charAt, substring) | Lexer |
| 3 | HashMap built-in or user generics + bitwise ops | Symbol tables, environments |
| 4 | Enums or tagged unions | AST representation |
| 5 | Nullable types | Optional values everywhere |
| 6 | argv access + exit() | CLI scripts |
| 7 | Process execution (libc `system()`) | Calling clang |

Steps 1-3 let you write the lexer in Senpai. Adding 4-5 gets the parser and type checker. Steps 6-7 complete the orchestration layer.
