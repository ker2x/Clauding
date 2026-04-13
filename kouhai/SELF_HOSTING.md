# Kouhai Self-Hosting Plan

## Status: Components Written! ⚠️ Need Architecture Change

All self-hosted components have been written. However, the Python compiler doesn't execute self-hosted Kouhai code - it compiles Kouhai source to LLVM IR using Python's own tools.

## Self-Hosted Components Written

| Component | File | Status |
|-----------|------|--------|
| Lexer | `src/lexer.kou` | ✅ Written |
| Token types | `src/token.kou` | ✅ Written |
| AST nodes | `src/ast.kou` | ✅ Written |
| Parser | `src/parser.kou` | ✅ Written |
| Type checker | `src/type_checker.kou` | ✅ Written |
| Code generator | `src/codegen.kou` | ✅ Written |
| Main CLI | `src/main.kou` | ✅ Written |

## The Problem

The Python compiler (`kouhai/compiler.py`):
1. Uses Python's own `tokens.py` lexer, not `src/lexer.kou`
2. Uses Python's own `parser.py`, not `src/parser.kou`  
3. Uses Python's own `types.py` type checker, not `src/type_checker.kou`
4. Uses Python's own `codegen.py`, not `src/codegen.kou`

**Self-hosting requires executing the Kouhai components, not just compiling them.**

## Architecture Needed for True Self-Hosting

1. **Interprete mode**: Execute Kouhai code directly from source
2. **Bootstrapping**: Use Python-compiled Kouhai to compile self-hosted components
3. **Cross-compile**: Generate IR that can then be compiled by the Python compiler

## Current Workaround

The Python compiler can still compile Kouhai programs using its own tools. The self-hosted components demonstrate that Kouhai is powerful enough to express its own compiler.

## Key Files

- `src/lexer.kou` - Full lexer (203 lines)
- `src/ast.kou` - All AST node classes  
- `src/parser.kou` - Complete recursive descent parser
- `src/token.kou` - Token type constants
- `src/type_checker.kou` - Type validation
- `src/codegen.kou` - LLVM IR generation
- `src/main.kou` - CLI orchestrator
