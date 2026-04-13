# Kouhai Self-Hosting Plan

## Status: Phase 1 Complete - Lexer Working! ✅

Self-hosting means the Kouhai compiler can compile its own source code.

## What's Needed

### 1. ✅ Map Data Structure (Done!)
`examples/map_demo.kou` - Working hash map implementation:
- `struct Entry` - key/value storage with used flag
- `class Map` - put/get/contains operations
- `fn hash_i64()` - FNV-1a hash for I64 keys

### 2. String Hash for Str Keys
Need to add `kouhai_hash_str(ptr) -> i64` to runtime.ll for string-keyed maps.

### 3. ✅ File I/O (Done via extern)
`extern fn fopen(path: Str, mode: Str) -> Ptr`
`extern fn fread(ptr, size, count, file) -> I64`
`extern fn fclose(file) -> I64`

## Completed Work

### Phase 1: Lexer in Kouhai ✅ DONE
`src/lexer.kou` and `lexer.kou`:
- Token struct with type, value, line, col fields
- Lexer scans source string character by character
- Returns `Array[Token]` containing all tokens
- Correct token type IDs matching Python compiler
- Keyword detection with elif chains
- Delimiter handling (parentheses, brackets, etc.)
- Operator recognition including two-char operators

**Test result**: Lexer correctly tokenizes `"fn main():\n    print(42)\n"` into 12 tokens:
- FN, IDENT, LPAREN, RPAREN, COLON, NEWLINE, INDENT, IDENT, LPAREN, INT_LIT, RPAREN, NEWLINE

### Phase 2: Parser in Kouhai - In Progress
`src/parser.kou`:
- Parser struct with tokens: Array[Token] and pos: I64
- parser_peek_type() extracts token type from Token
- parser_at() checks if current token matches type
- parser_eat() consumes token and advances
- Parser skeleton ready, full AST generation pending

## Remaining Phases

### Phase 3: Type Checker in Kouhai
Write `src/type_checker.kou`:
- Validates types and resolves names

### Phase 4: Code Generator in Kouhai
Write `src/codegen.kou`:
- Generates LLVM IR from AST
- Can re-use runtime.ll helpers

### Phase 5: Bootstrap
- Compile Kouhai lexer/parser/typechecker/codegen with Python Kouhai
- Verify self-compilation produces identical output
- Delete Python implementation (optional)

## Implementation Order

1. [x] Map data structure (done in examples/map_demo.kou)
2. [ ] Add string hash to runtime.ll
3. [x] Write token.kou - token type constants
4. [x] Write lexer.kou - source scanning
5. [ ] Write ast.kou - AST node definitions
6. [ ] Write parser.kou - recursive descent parser
7. [ ] Write type_checker.kou - type validation
8. [ ] Write codegen.kou - IR generation
9. [ ] Write main.kou - CLI entry point
10. [ ] Test: Compile Kouhai source with Kouhai compiler

## Bootstrap Strategy

Two-pass approach:
1. Write compiler components in Kouhai, compile with Python Kouhai
2. Self-compiled Kouhai compiles itself

This verifies the language is powerful enough to express its own compiler.

## Key Files

- `lexer.kou` - Working self-hosted lexer (in root dir for import)
- `src/lexer.kou` - Copy of working lexer (in src for organized structure)
- `src/parser.kou` - Parser with lexer integration
- `src/token.kou` - Token type definitions (reference)