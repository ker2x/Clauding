# Senpai/Kouhai Post-Mortem

## What Was Built

**Senpai** — A statically-typed, object-oriented programming language compiling to LLVM IR via pure Python (no dependencies beyond Python 3.10+ and clang).

- ~5,000 lines of Python across lexer, parser, type checker, code generator
- Full language: classes with inheritance, arrays, strings, for/while loops, operator overloading, UFCS, C FFI
- 119 end-to-end tests passing
- Working examples: hello world, factorial, fibonacci, classes, arrays, strings
- A lexer written in Kouhai itself (partial self-hosting attempt)

**Kouhai** — A cleaner reboot of Senpai with better architecture:
- IR builder abstraction layer (cleaner codegen)
- External runtime/runtime.ll (no embedded runtime strings)
- Clear systems-language identity

## What Went Wrong

### 1. Self-Hosting as the Goal, Not the Product

The fatal mistake: self-hosting was set as the *completion criteria*, not a feature. This meant:

- Every feature had to be perfect before the project was "done"
- A circular dependency: the compiler needs features that require the compiler to be done
- No shippable product until the entire mountain was climbed

**Lesson**: Self-hosting is an interesting property, not a success metric. Ship something useful first.

### 2. Feature Count Creep

Senpai accumulated features like a kitchen-sink language:
- Operator overloading
- UFCS (Uniform Function Call Syntax)
- Method chaining
- Generic arrays
- Bitwise operators on all widths
- Ternary expressions
- First-class classes with vtables
- Module system with namespacing
- Extern/FFI with string support

Each feature multiplied the complexity of the type checker and code generator. A bug in one area (e.g., string handling) blocked entire other areas.

**Lesson**: Define a minimal viable language. Ship with 80% of features cut, then add only what users actually need.

### 3. The Self-Hosting Dependency Chain

The SELF_HOSTING.md documents this well, but the core problem:

```
Need File I/O → Need Str in FFI → Need string manipulation → Need HashMap
Need HashMap → Need generics → Need runtime HashMap → Need process execution
Need process execution → Need FFI with strings → ...
```

Every "need" is also a blocker for other "needs". The network effect is brutal.

**Lesson**: Self-hosting requires bootstrapping, and bootstrapping requires careful sequencing of minimal features. It's not a hill you climb linearly.

### 4. Python as the Implementation Language

Python made prototyping fast but created problems:
- No static analysis (type errors only at runtime)
- The self-hosted version can't use Python's data structures
- Dict operations that are 1 line in Python become 50 lines in Kouhai

**Lesson**: If self-hosting is the goal, implement in a language that can eventually compile itself. Or accept that self-hosting is an asymptotic goal, not a milestone.

### 5. No Incremental Shipped Value

The project had no shipped artifacts until everything was working. Months of work with zero feedback loop.

**Lesson**: Ship early, ship often. Even a partial lexer or parser is progress. A working compiler that compiles "print(1+1)" is still valuable.

## What Was Learned

### Compiler Architecture

- **Lexer → Parser → AST → Type Checker → Codegen → IR → clang** is a proven pipeline
- Separating concerns makes incremental improvements possible
- The type checker is typically the hardest component (1012 of ~5000 lines)

### LLVM IR Generation

- Text-based IR emission (no llvmlite dependency) works fine
- SSA register allocation via temporary counters is simple and effective
- Runtime helpers extracted to a separate .ll file is cleaner than embedded strings

### Language Design

- Python-like syntax is compelling but creates "is this Python?" confusion
- Explicit fixed-width types (I64, not "int") prevent a whole class of bugs
- Everything-is-an-object (like Ruby) is elegant but adds vtable overhead everywhere
- UFCS is powerful but makes type checking harder

### Self-Hosting Reality

- ~5,000 lines of Python → ~5,200+ lines of Kouhai (10-30% more verbose)
- 13-21 sessions of focused work to bootstrap from zero
- Codegen alone (Phase 4) is nearly half the total effort
- The circular dependency problem: you need features to build the compiler, and you need the compiler to build features

### What Would Have Worked Better

1. **Start with a tiny subset**: Just integers, functions, if/else. No classes, no strings, no arrays.
2. **Ship the tiny subset**: Compile "fn main() { if true { print(42) } }" and call it done.
3. **Add one feature at a time**: Each feature ships. Each feature gets user feedback.
4. **Forget self-hosting**: Build a compiler that compiles something useful. Self-hosting is a curiosity.
5. **Use Rust instead of Python**: Better for systems programming, can eventually self-host, static types catch bugs early.

## Key Statistics

| Metric | Value |
|--------|-------|
| Python compiler lines | ~5,000 |
| Kouhai self-hosting estimate | ~5,200+ |
| Time to first working compiler | ~2-3 sessions |
| Time to self-hosting (estimated) | 13-21 sessions |
| Tests at completion | 119 e2e tests |
| Features implemented | ~15 major language features |
| Self-hosting progress | ~30% (lexer only) |

## The Hard Truth

Senpai was a success as a learning project and a technical artifact. It was a failure as a product because it never shipped a product — just an aspiration.

The lexer works. The type checker works. The code generator works. The tests pass. The examples run. But without self-hosting, the project was never "complete" in its own eyes, so it never stopped.

Sometimes the most ambitious thing you can do is ship.

## Next Time

1. Define "shipped" on day one
2. Cut features until shipped
3. Add features one at a time, each time asking "is this shipped now?"
4. Treat self-hosting as a milestone, not the goal
5. Build in Rust if self-hosting is a real requirement

---

*"The best compiler is the one that compiles your program."* — Not the one that compiles itself.
