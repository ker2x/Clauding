# Kouhai Self-Hosting Status

## ⚠️ Current Status: PROOF-OF-CONCEPT

The self-hosted components are **simplified/skeletal versions** demonstrating the concept, not full implementations.

## Self-Hosted Components (Simplified)

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Lexer | `src/lexer.kou` | ~200 | Basic tokenization |
| Token types | `src/token.kou` | ~150 | Constants |
| AST nodes | `src/ast.kou` | ~225 | Definitions |
| Parser | `src/parser.kou` | 213 | Basic expressions |
| Type checker | `src/type_checker.kou` | 69 | Stub (all `return true`) |
| Code generator | `src/codegen.kou` | 33 | Stub (hardcoded output) |
| Main CLI | `src/main.kou` | 13 | Stub (tests lexer) |

## Interpreting Self-Hosted Code

A **Kouhai interpreter** (`kouhai_interpreter.py`) was added that:
- Parses Kouhai source using Python tools
- Interprets the resulting AST
- Enables `--interpret` mode to run self-hosted components

## What Needs Work

1. **Parser**: Only handles basic expressions, missing struct/class/import
2. **Type checker**: All functions return `true` - does nothing
3. **Codegen**: Hardcodes `define i64 @main() { ret i64 0 }`
4. **Full bootstrapping**: Need a path to compile Kouhai with Kouhai

## Key Files

- `src/lexer.kou` - Basic lexer
- `src/parser.kou` - Basic recursive descent  
- `src/ast.kou` - AST definitions
- `kouhai_interpreter.py` - Interpreter for AST execution
- `scripts/run.py --interpret` - Run with interpreter
