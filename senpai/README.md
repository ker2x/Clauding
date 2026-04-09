# Senpai

A statically-typed, object-oriented programming language that compiles to **native binaries** via LLVM IR. Python-like syntax, zero dependencies, and it runs *fast*.

```python
fn main():
    print("Hello, World!")
```

```bash
python scripts/run.py examples/hello.sen   # compiles and runs instantly
```

## Why Senpai?

- **Native speed** — compiles to optimized machine code through LLVM, not interpreted
- **Python-like syntax** — significant indentation, clean and readable, no semicolons or braces
- **Truly static types** — every type error caught at compile time, no runtime surprises
- **Zero dependencies** — just Python 3.10+ and `clang` (Xcode CLI tools on macOS). No pip install, no npm, no cargo
- **Dead simple toolchain** — one command to compile and run: `python scripts/run.py your_program.sen`

## Features

### Static Type System with Explicit Widths

No ambiguous `int` or `long`. Every numeric type has a clear, explicit width.

| Type | Description |
|------|-------------|
| `I8`, `I16`, `I32`, `I64` | Signed integers |
| `U8`, `U16`, `U32`, `U64` | Unsigned integers |
| `Float` | 32-bit IEEE 754 (`3.14f` suffix) |
| `Double` | 64-bit IEEE 754 (`3.14`) |
| `Bool` | `true` / `false` |
| `Str` | Heap-allocated string |
| `Array[T]` | Generic growable array |
| `Int` | Alias for `I64` |
| `Void` | No value (return type only) |

No implicit conversions — mismatched types are compile errors. Use `expr as Type` for explicit casts.

### Classes with Single Inheritance

Full object system: constructors, fields, methods, and vtable-based dynamic dispatch.

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

Define how your types behave with standard operators via dunder methods.

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

Supported operators: `__add__`, `__sub__`, `__mul__`, `__div__`, `__mod__`, `__eq__`, `__lt__`, `__gt__`, `__neg__`

### Generic Growable Arrays

Type-safe dynamic arrays with bounds checking.

```python
fn main():
    let names = Array[Str]()
    names.push("Alice")
    names.push("Bob")
    names.push("Charlie")

    for i in range(names.len()):
        print(names.get(i))
```

Methods: `.push(val)`, `.get(idx)`, `.set(idx, val)`, `.len()`

Works with any type: `Array[I64]`, `Array[Double]`, `Array[Str]`, even `Array[MyClass]`.

### For Loops and While Loops

```python
# Count from 0 to 9
for i in range(10):
    print(i)

# Count from 5 to 9
for i in range(5, 10):
    print(i)

# While loop
let x: I64 = 100
while x > 1:
    x = x / 2
```

### If / Elif / Else

```python
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")
```

### Boolean Logic

Short-form `and`, `or`, `not` — no `&&` or `||` to remember.

```python
if x > 0 and x < 100:
    print("in range")

if not found or retries == 0:
    print("giving up")
```

### Type Casting

Explicit casts between numeric types with `as`. No silent truncation.

```python
let total: I64 = 42
let avg: Double = total as Double / 10.0

let big: I64 = 1000
let small: I8 = big as I8    # explicit truncation
```

### Built-in `to_str()` on Everything

All numeric types and Bool have a `.to_str()` method for easy string building.

```python
fn main():
    let x: I64 = 42
    let pi: Double = 3.14159
    let flag: Bool = true
    print("x = " + x.to_str())
    print("pi = " + pi.to_str())
    print("flag = " + flag.to_str())
```

### Module System

Split your code across files. Imported modules are namespaced — no name collisions.

```python
# mathlib.sen
fn abs(x: I64) -> I64:
    if x < 0:
        return 0 - x
    return x

class Vec2(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y
```

```python
# main.sen
import "mathlib.sen"

fn main():
    print(mathlib.abs(-42))
    let v = mathlib.Vec2(3, 4)
    print(v.x)
```

### C FFI with `extern` and `link`

Call any C function directly. Declare it with `extern`, link the library with `link`.

```python
link "m"

extern fn sqrt(x: Double) -> Double
extern fn pow(base: Double, exp: Double) -> Double

fn main():
    print("sqrt(144) = " + sqrt(144.0).to_str())
    print("2^10 = " + pow(2.0, 10.0).to_str())
```

Link directives bubble up through imports — if a module declares `link "m"`, any program importing it gets the library automatically. No build scripts needed.

Currently supports numeric and Bool types in extern signatures. Pointer types coming soon.

### Method Chaining

Methods that return `self` or another object can be chained naturally.

```python
class Builder(Object):
    fn __init__(self):
        self.count = 0

    fn add(self) -> Builder:
        self.count = self.count + 1
        return self

fn main():
    let b = Builder()
    b.add().add().add()
    print(b.count)    # 3
```

### Subtype Polymorphism

Pass a child class anywhere a parent type is expected.

```python
fn greet(a: Animal):
    print(a.speak())

fn main():
    let d = Dog("Rex")
    greet(d)    # works — Dog is an Animal
```

## How It Works

```
.sen source --> Lexer --> Parser --> AST --> TypeChecker --> CodeGen --> LLVM IR --> clang --> native binary
```

The compiler emits LLVM IR as plain text strings — no llvmlite, no LLVM bindings, no C++ needed. The IR is piped to `clang` for assembly and linking. Objects are heap-allocated with vtable pointers for dynamic dispatch. Primitives are unboxed for performance.

## Quick Start

```bash
# Requires: Python 3.10+, clang (Xcode Command Line Tools on macOS)

# Run a program
python scripts/run.py examples/hello.sen

# Inspect generated LLVM IR
python scripts/run.py examples/hello.sen --ir

# Link an external library via CLI
python scripts/run.py myprogram.sen --link z

# Run tests (126 tests)
python tests/test_compiler.py
```

## Examples

| File | What it demonstrates |
|------|---------------------|
| `examples/hello.sen` | Hello World |
| `examples/factorial.sen` | Recursion |
| `examples/fib.sen` | Fibonacci with while loop |
| `examples/float_demo.sen` | Float/Double arithmetic |
| `examples/strings.sen` | String concatenation |
| `examples/classes.sen` | Inheritance, super, operators, method chaining |
| `examples/arrays.sen` | Arrays, for loops, to_str, type casting |
| `examples/mathlib.sen` | Reusable module (functions + classes) |
| `examples/import_demo.sen` | Importing and using a module |
| `examples/extern_math.sen` | C FFI with extern and link |
