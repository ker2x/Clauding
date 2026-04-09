# Senpai

A statically-typed, object-oriented programming language that compiles to native binaries via LLVM IR.

- **Python-like syntax** with significant indentation
- **Static typing** with explicit fixed-width numeric types
- **Single inheritance** with vtable dispatch
- **Operator overloading** via dunder methods (`__add__`, `__eq__`, etc.)
- **Zero dependencies** beyond Python 3.10+ stdlib and `clang`

## Quick Start

```bash
# Requires: Python 3.10+, clang (Xcode Command Line Tools on macOS)

# Run a program
python scripts/run.py examples/hello.sen

# Inspect generated LLVM IR
python scripts/run.py examples/hello.sen --ir

# Run tests (112 tests)
python tests/test_compiler.py
```

## A Taste of Senpai

```python
class Animal(Object):
    fn __init__(self, name: Str, sound: Str):
        self.name = name
        self.sound = sound

    fn speak(self) -> Str:
        return self.name + " says " + self.sound

class Dog(Animal):
    fn __init__(self, name: Str):
        super.__init__(name, "woof")

    fn fetch(self) -> Str:
        return self.name + " fetches the ball!"

fn main():
    let d = Dog("Rex")
    print(d.speak())       # Rex says woof
    print(d.fetch())       # Rex fetches the ball!
```

### Operator Overloading

```python
class Vec2(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y

    fn __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    fn __eq__(self, other: Vec2) -> Bool:
        if self.x == other.x:
            if self.y == other.y:
                return true
        return false

fn main():
    let v = Vec2(1, 2) + Vec2(3, 4)
    print(v.x)    # 4
    print(v.y)    # 6
```

### Arrays and For Loops

```python
fn sum(arr: Array[I64]) -> I64:
    let total: I64 = 0
    for i in range(arr.len()):
        total = total + arr.get(i)
    return total

fn main():
    let squares = Array[I64]()
    for i in range(1, 11):
        squares.push(i * i)
    print("Sum: " + sum(squares).to_str())
    let avg: Double = sum(squares) as Double / 10.0
    print("Average: " + avg.to_str())
```

## Type System

| Type | Description |
|------|-------------|
| `I8`, `I16`, `I32`, `I64` | Signed integers (explicit width) |
| `U8`, `U16`, `U32`, `U64` | Unsigned integers |
| `Float` | 32-bit IEEE 754 (`3.14f` suffix) |
| `Double` | 64-bit IEEE 754 (`3.14`) |
| `Bool` | `true` / `false` |
| `Str` | Heap-allocated string, concatenation with `+` |
| `Array[T]` | Generic growable array |
| `Void` | No value (return type only) |
| `Int` | Alias for `I64` |

No implicit conversions. Use `expr as Type` for explicit casts between numeric types.

## Language Features

**Variables** &mdash; `let` bindings with optional type annotation:
```python
let x: I64 = 42
let name = "hello"       # type inferred from literal
```

**Functions** &mdash; with typed parameters and return:
```python
fn add(a: I64, b: I64) -> I64:
    return a + b
```

**Control flow** &mdash; if/elif/else, while, for-in-range:
```python
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

while x > 0:
    x = x - 1

for i in range(10):
    print(i)
```

**Classes** &mdash; single inheritance, constructors, fields, methods:
```python
class Dog(Animal):
    fn __init__(self, name: Str):
        super.__init__(name, "woof")
        self.tricks = 0

    fn learn_trick(self) -> Dog:
        self.tricks = self.tricks + 1
        return self
```

**Operator methods** &mdash; `__add__`, `__sub__`, `__mul__`, `__div__`, `__eq__`, `__lt__`, `__gt__`, `__neg__`, `__mod__`

**Built-in methods** &mdash; `.to_str()` on all numeric types and Bool

**Type casting** &mdash; `expr as Type` for int/float conversions and width changes

**Arrays** &mdash; `Array[T]()`, `.push(val)`, `.get(idx)`, `.set(idx, val)`, `.len()`

## How It Works

```
.sen source --> Lexer --> Tokens --> Parser --> AST --> TypeChecker --> Typed AST --> CodeGen --> LLVM IR --> clang --> native binary
```

The compiler emits LLVM IR as plain text strings (no llvmlite or LLVM bindings), pipes to `clang` for assembly and linking. Objects are heap-allocated with vtable pointers for dynamic dispatch. Primitives are unboxed for performance.

## Examples

| File | What it demonstrates |
|------|---------------------|
| `examples/hello.sen` | Hello World |
| `examples/factorial.sen` | Recursion |
| `examples/fib.sen` | Fibonacci with while loop |
| `examples/float_demo.sen` | Float/Double arithmetic |
| `examples/strings.sen` | String concatenation |
| `examples/classes.sen` | Inheritance, super, operator overloading, method chaining |
| `examples/arrays.sen` | Arrays, for loops, to_str, type casting |
