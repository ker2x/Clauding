#!/usr/bin/env python3
"""Tests for the Senpai compiler. Standalone, no pytest."""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from senpai.tokens import lex, LexError, TT
from senpai.parser import Parser, ParseError
from senpai.types import check_program, TypeError_
from senpai.codegen import CodeGen
from senpai.compiler import compile_source, CompileError


# --- Helpers ---

def compile_and_run(source: str) -> str:
    """Compile source, run binary, return stdout."""
    binary = compile_source(source)
    try:
        result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
        return result.stdout
    finally:
        Path(binary).unlink(missing_ok=True)


def compile_and_run_with_modules(source: str, modules: dict[str, str]) -> str:
    """Compile source with module files, run binary, return stdout.
    modules: dict of filename -> source code (written to temp dir).
    """
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, msrc in modules.items():
            Path(tmpdir, fname).write_text(msrc)
        Path(tmpdir, "main.sen").write_text(source)
        binary = compile_source(source, source_dir=tmpdir)
        try:
            result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
            return result.stdout
        finally:
            Path(binary).unlink(missing_ok=True)


def expect_compile_error(source: str, expected_fragment: str, source_dir: str | None = None):
    """Assert that compilation fails with an error containing the fragment."""
    try:
        compile_source(source, source_dir=source_dir)
        assert False, f"expected CompileError containing '{expected_fragment}'"
    except CompileError as e:
        assert expected_fragment in str(e), f"expected '{expected_fragment}' in '{e}'"


passed = 0
failed = 0

def test(name):
    def decorator(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    return decorator


# === Lexer tests ===

@test("lex: basic tokens")
def _():
    tokens = lex('let x: I64 = 42')
    types = [t.type for t in tokens if t.type not in (TT.NEWLINE, TT.EOF)]
    assert types == [TT.LET, TT.IDENT, TT.COLON, TT.IDENT, TT.EQ, TT.INT_LIT]

@test("lex: string literal")
def _():
    tokens = lex('let s = "hello"')
    str_tok = [t for t in tokens if t.type == TT.STR_LIT][0]
    assert str_tok.value == "hello"

@test("lex: string escape sequences")
def _():
    tokens = lex(r'let s = "a\nb"')
    str_tok = [t for t in tokens if t.type == TT.STR_LIT][0]
    assert str_tok.value == "a\nb"

@test("lex: float literal")
def _():
    tokens = lex('let f = 3.14')
    float_tok = [t for t in tokens if t.type == TT.FLOAT_LIT][0]
    assert float_tok.value == (3.14, False)

@test("lex: float32 literal with f suffix")
def _():
    tokens = lex('let f = 3.14f')
    float_tok = [t for t in tokens if t.type == TT.FLOAT_LIT][0]
    assert float_tok.value == (3.14, True)

@test("lex: operators")
def _():
    tokens = lex('a == b != c <= d >= e -> f')
    ops = [t.type for t in tokens if t.type not in (TT.IDENT, TT.NEWLINE, TT.EOF)]
    assert ops == [TT.EQEQ, TT.NEQ, TT.LE, TT.GE, TT.ARROW]

@test("lex: indent/dedent")
def _():
    tokens = lex('if true:\n    x = 1\ny = 2')
    types = [t.type for t in tokens]
    assert TT.INDENT in types
    assert TT.DEDENT in types

@test("lex: keywords")
def _():
    tokens = lex('fn class let if elif else while return self super and or not')
    kw_types = [t.type for t in tokens if t.type not in (TT.NEWLINE, TT.EOF)]
    expected = [TT.FN, TT.CLASS, TT.LET, TT.IF, TT.ELIF, TT.ELSE, TT.WHILE,
                TT.RETURN, TT.SELF, TT.SUPER, TT.AND, TT.OR, TT.NOT]
    assert kw_types == expected

@test("lex: bool literals")
def _():
    tokens = lex('true false')
    bools = [t for t in tokens if t.type == TT.BOOL_LIT]
    assert len(bools) == 2
    assert bools[0].value is True
    assert bools[1].value is False

@test("lex: comments ignored")
def _():
    tokens = lex('x = 1  # this is a comment')
    types = [t.type for t in tokens if t.type not in (TT.NEWLINE, TT.EOF)]
    assert types == [TT.IDENT, TT.EQ, TT.INT_LIT]

@test("lex: unterminated string")
def _():
    try:
        lex('let s = "hello')
        assert False
    except LexError:
        pass


# === Parser tests ===

@test("parse: simple function")
def _():
    tokens = lex('fn main():\n    return\n')
    prog = Parser(tokens).parse()
    assert len(prog.functions) == 1
    assert prog.functions[0].name == "main"

@test("parse: function with params and return type")
def _():
    tokens = lex('fn add(a: I64, b: I64) -> I64:\n    return a + b\n')
    prog = Parser(tokens).parse()
    fn = prog.functions[0]
    assert fn.name == "add"
    assert len(fn.params) == 2
    assert fn.ret_type == "I64"

@test("parse: let statement")
def _():
    from senpai.ast_nodes import LetStmt, IntLit
    tokens = lex('fn main():\n    let x: I64 = 42\n')
    prog = Parser(tokens).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert stmt.name == "x"
    assert isinstance(stmt.value, IntLit)

@test("parse: if/elif/else")
def _():
    from senpai.ast_nodes import IfStmt
    src = 'fn main():\n    if true:\n        return\n    elif false:\n        return\n    else:\n        return\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, IfStmt)
    assert len(stmt.elif_clauses) == 1
    assert len(stmt.else_body) == 1

@test("parse: while loop")
def _():
    from senpai.ast_nodes import WhileStmt
    src = 'fn main():\n    while true:\n        return\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, WhileStmt)

@test("parse: operator precedence")
def _():
    from senpai.ast_nodes import BinOp
    src = 'fn main():\n    let x: I64 = 1 + 2 * 3\n'
    prog = Parser(lex(src)).parse()
    from senpai.ast_nodes import LetStmt
    stmt = prog.functions[0].body[0]
    # Should be Add(1, Mul(2, 3))
    assert isinstance(stmt.value, BinOp)
    assert stmt.value.op == "+"
    assert isinstance(stmt.value.right, BinOp)
    assert stmt.value.right.op == "*"

@test("parse: unary minus")
def _():
    from senpai.ast_nodes import UnaryOp
    src = 'fn main():\n    let x: I64 = -42\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt.value, UnaryOp)
    assert stmt.value.op == "-"

@test("parse: function call")
def _():
    from senpai.ast_nodes import ExprStmt, Call
    src = 'fn foo(x: I64) -> I64:\n    return x\nfn main():\n    foo(42)\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[1].body[0]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, Call)
    assert stmt.expr.func == "foo"

@test("parse: nested expressions with parens")
def _():
    from senpai.ast_nodes import BinOp
    src = 'fn main():\n    let x: I64 = (1 + 2) * 3\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt.value, BinOp)
    assert stmt.value.op == "*"
    assert isinstance(stmt.value.left, BinOp)
    assert stmt.value.left.op == "+"


# === Type checker tests ===

@test("type: valid program")
def _():
    src = 'fn main():\n    let x: I64 = 42\n    print(x)\n'
    prog = Parser(lex(src)).parse()
    check_program(prog)  # should not raise

@test("type: type mismatch in let")
def _():
    expect_compile_error(
        'fn main():\n    let x: I64 = true\n',
        "type mismatch"
    )

@test("type: undefined variable")
def _():
    expect_compile_error(
        'fn main():\n    print(x)\n',
        "undefined variable"
    )

@test("type: undefined function")
def _():
    expect_compile_error(
        'fn main():\n    foo()\n',
        "undefined function"
    )

@test("type: wrong number of args")
def _():
    expect_compile_error(
        'fn foo(a: I64) -> I64:\n    return a\nfn main():\n    foo(1, 2)\n',
        "expects 1 args"
    )

@test("type: wrong arg type")
def _():
    expect_compile_error(
        'fn foo(a: I64) -> I64:\n    return a\nfn main():\n    foo(true)\n',
        "expected I64, got Bool"
    )

@test("type: return type mismatch")
def _():
    expect_compile_error(
        'fn foo() -> I64:\n    return true\n',
        "return type mismatch"
    )

@test("type: if condition must be Bool")
def _():
    expect_compile_error(
        'fn main():\n    if 42:\n        return\n',
        "must be Bool"
    )

@test("type: cannot add I64 and Bool")
def _():
    expect_compile_error(
        'fn main():\n    let x: I64 = 1 + true\n',
        "cannot apply '+'"
    )

@test("type: comparison returns Bool")
def _():
    src = 'fn main():\n    let x: Bool = 1 < 2\n'
    prog = Parser(lex(src)).parse()
    check_program(prog)

@test("type: assign wrong type")
def _():
    expect_compile_error(
        'fn main():\n    let x: I64 = 1\n    x = true\n',
        "cannot assign Bool to I64"
    )

@test("type: Int alias resolves to I64")
def _():
    src = 'fn foo(x: Int) -> Int:\n    return x\nfn main():\n    print(foo(42))\n'
    prog = Parser(lex(src)).parse()
    check_program(prog)


# === End-to-end compile+run tests ===

@test("e2e: hello world")
def _():
    out = compile_and_run('fn main():\n    print("Hello, World!")\n')
    assert out.strip() == "Hello, World!"

@test("e2e: integer arithmetic")
def _():
    out = compile_and_run('fn main():\n    print(2 + 3)\n')
    assert out.strip() == "5"

@test("e2e: subtraction")
def _():
    out = compile_and_run('fn main():\n    print(10 - 3)\n')
    assert out.strip() == "7"

@test("e2e: multiplication")
def _():
    out = compile_and_run('fn main():\n    print(6 * 7)\n')
    assert out.strip() == "42"

@test("e2e: division")
def _():
    out = compile_and_run('fn main():\n    print(10 / 3)\n')
    assert out.strip() == "3"

@test("e2e: modulo")
def _():
    out = compile_and_run('fn main():\n    print(10 % 3)\n')
    assert out.strip() == "1"

@test("e2e: negative number")
def _():
    out = compile_and_run('fn main():\n    print(-42)\n')
    assert out.strip() == "-42"

@test("e2e: factorial")
def _():
    src = '''fn factorial(n: I64) -> I64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main():
    print(factorial(10))
'''
    out = compile_and_run(src)
    assert out.strip() == "3628800"

@test("e2e: fibonacci")
def _():
    src = '''fn fib(n: I64) -> I64:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fn main():
    print(fib(10))
'''
    out = compile_and_run(src)
    assert out.strip() == "55"

@test("e2e: while loop")
def _():
    src = '''fn main():
    let i: I64 = 0
    let sum: I64 = 0
    while i < 10:
        sum = sum + i
        i = i + 1
    print(sum)
'''
    out = compile_and_run(src)
    assert out.strip() == "45"

@test("e2e: if/else")
def _():
    src = '''fn main():
    let x: I64 = 5
    if x > 3:
        print(1)
    else:
        print(0)
'''
    out = compile_and_run(src)
    assert out.strip() == "1"

@test("e2e: elif")
def _():
    src = '''fn classify(n: I64) -> I64:
    if n > 0:
        return 1
    elif n == 0:
        return 0
    else:
        return -1

fn main():
    print(classify(5))
    print(classify(0))
    print(classify(-3))
'''
    out = compile_and_run(src)
    assert out.strip() == "1\n0\n-1"

@test("e2e: multiple functions")
def _():
    src = '''fn double(x: I64) -> I64:
    return x * 2

fn triple(x: I64) -> I64:
    return x * 3

fn main():
    print(double(5))
    print(triple(5))
'''
    out = compile_and_run(src)
    assert out.strip() == "10\n15"

@test("e2e: boolean print")
def _():
    src = '''fn main():
    print(true)
    print(false)
'''
    out = compile_and_run(src)
    assert out.strip() == "true\nfalse"

@test("e2e: boolean logic")
def _():
    src = '''fn main():
    let a: Bool = true
    let b: Bool = false
    if a and not b:
        print(1)
    else:
        print(0)
'''
    out = compile_and_run(src)
    assert out.strip() == "1"

@test("e2e: string print")
def _():
    out = compile_and_run('fn main():\n    print("hello senpai")\n')
    assert out.strip() == "hello senpai"

@test("e2e: nested calls")
def _():
    src = '''fn add(a: I64, b: I64) -> I64:
    return a + b

fn main():
    print(add(add(1, 2), add(3, 4)))
'''
    out = compile_and_run(src)
    assert out.strip() == "10"

@test("e2e: complex expression")
def _():
    src = '''fn main():
    let x: I64 = (2 + 3) * (4 - 1)
    print(x)
'''
    out = compile_and_run(src)
    assert out.strip() == "15"

@test("e2e: let without type annotation")
def _():
    src = '''fn main():
    let x = 42
    print(x)
'''
    out = compile_and_run(src)
    assert out.strip() == "42"

@test("e2e: comparison operators")
def _():
    src = '''fn check(a: I64, b: I64):
    if a == b:
        print(1)
    if a != b:
        print(2)
    if a < b:
        print(3)
    if a > b:
        print(4)
    if a <= b:
        print(5)
    if a >= b:
        print(6)

fn main():
    check(3, 5)
'''
    out = compile_and_run(src)
    # 3 == 5? no. 3 != 5? yes(2). 3 < 5? yes(3). 3 > 5? no. 3 <= 5? yes(5). 3 >= 5? no.
    assert out.strip() == "2\n3\n5"


# === Phase 2: Float/Double tests ===

@test("e2e: double arithmetic")
def _():
    src = '''fn main():
    let x: Double = 3.14
    let y: Double = 2.0
    print(x + y)
'''
    out = compile_and_run(src)
    assert out.strip() == "5.14"

@test("e2e: double multiplication")
def _():
    src = '''fn main():
    print(2.5 * 4.0)
'''
    out = compile_and_run(src)
    assert out.strip() == "10"

@test("e2e: double division")
def _():
    src = '''fn main():
    print(10.0 / 4.0)
'''
    out = compile_and_run(src)
    assert out.strip() == "2.5"

@test("e2e: double comparison")
def _():
    src = '''fn main():
    let x: Double = 3.14
    if x > 3.0:
        print(1)
    else:
        print(0)
'''
    out = compile_and_run(src)
    assert out.strip() == "1"

@test("e2e: double negation")
def _():
    src = '''fn main():
    let x: Double = 3.14
    print(-x)
'''
    out = compile_and_run(src)
    assert out.strip() == "-3.14"

@test("e2e: double function")
def _():
    src = '''fn average(a: Double, b: Double) -> Double:
    return (a + b) / 2.0

fn main():
    print(average(3.0, 7.0))
'''
    out = compile_and_run(src)
    assert out.strip() == "5"

@test("e2e: double let without annotation")
def _():
    src = '''fn main():
    let pi = 3.14159
    print(pi)
'''
    out = compile_and_run(src)
    assert out.strip() == "3.14159"

@test("type: double mismatch with int")
def _():
    expect_compile_error(
        'fn main():\n    let x: Double = 1.0\n    let y: I64 = x\n',
        "type mismatch"
    )

@test("type: cannot add double and int")
def _():
    expect_compile_error(
        'fn main():\n    let x: I64 = 1 + 2.0\n',
        "mismatched types"
    )

@test("e2e: float32 literal")
def _():
    src = '''fn main():
    let x = 3.14f
    print(x)
'''
    out = compile_and_run(src)
    assert out.strip() == "3.14"

@test("e2e: float32 arithmetic")
def _():
    src = '''fn main():
    let a = 2.5f
    let b = 1.5f
    print(a + b)
    print(a * b)
'''
    out = compile_and_run(src)
    assert out.strip() == "4\n3.75"

@test("type: cannot mix Float and Double")
def _():
    expect_compile_error(
        'fn main():\n    let a = 1.0f\n    let b = 2.0\n    let c = a + b\n',
        "mismatched types"
    )


# === Phase 2: String concatenation tests ===

@test("e2e: string concatenation")
def _():
    src = '''fn main():
    let a: Str = "hello"
    let b: Str = " world"
    print(a + b)
'''
    out = compile_and_run(src)
    assert out.strip() == "hello world"

@test("e2e: string concat chain")
def _():
    src = '''fn main():
    print("a" + "b" + "c")
'''
    out = compile_and_run(src)
    assert out.strip() == "abc"

@test("e2e: string concat in function")
def _():
    src = '''fn greet(name: Str) -> Str:
    return "Hello, " + name + "!"

fn main():
    print(greet("Senpai"))
'''
    out = compile_and_run(src)
    assert out.strip() == "Hello, Senpai!"


# === Phase 2: Integer width tests ===

@test("e2e: I32 arithmetic")
def _():
    src = '''fn add32(a: I32, b: I32) -> I32:
    return a + b

fn main():
    print(add32(100, 200))
'''
    out = compile_and_run(src)
    assert out.strip() == "300"

@test("e2e: I8 variable")
def _():
    src = '''fn main():
    let x: I8 = 42
    print(x)
'''
    out = compile_and_run(src)
    assert out.strip() == "42"

@test("e2e: U64 arithmetic")
def _():
    src = '''fn main():
    let x: U64 = 100
    let y: U64 = 30
    print(x / y)
'''
    out = compile_and_run(src)
    assert out.strip() == "3"

@test("type: cannot mix I32 and I64")
def _():
    expect_compile_error(
        'fn foo(a: I32, b: I64) -> I32:\n    return a + b\n',
        "mismatched types"
    )


# === Phase 3: Class tests ===

@test("e2e: basic class with fields and methods")
def _():
    src = '''class Counter(Object):
    fn __init__(self, start: I64):
        self.value = start

    fn get(self) -> I64:
        return self.value

    fn increment(self):
        self.value = self.value + 1

fn main():
    let c = Counter(0)
    c.increment()
    c.increment()
    c.increment()
    print(c.get())
'''
    out = compile_and_run(src)
    assert out.strip() == "3"

@test("e2e: class with multiple fields")
def _():
    src = '''class Point(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y

    fn sum(self) -> I64:
        return self.x + self.y

fn main():
    let p = Point(3, 4)
    print(p.sum())
    print(p.x)
    print(p.y)
'''
    out = compile_and_run(src)
    assert out.strip() == "7\n3\n4"

@test("e2e: method returning self for chaining")
def _():
    src = '''class Builder(Object):
    fn __init__(self, val: I64):
        self.val = val

    fn add(self, n: I64) -> Builder:
        self.val = self.val + n
        return self

    fn result(self) -> I64:
        return self.val

fn main():
    let b = Builder(0)
    print(b.add(10).add(20).add(12).result())
'''
    out = compile_and_run(src)
    assert out.strip() == "42"

@test("e2e: class as function parameter")
def _():
    src = '''class Box(Object):
    fn __init__(self, value: I64):
        self.value = value

    fn get(self) -> I64:
        return self.value

fn unbox(b: Box) -> I64:
    return b.get()

fn main():
    let b = Box(99)
    print(unbox(b))
'''
    out = compile_and_run(src)
    assert out.strip() == "99"

@test("e2e: class with string field")
def _():
    src = '''class Dog(Object):
    fn __init__(self, name: Str):
        self.name = name

    fn greet(self) -> Str:
        return "Woof! I am " + self.name

fn main():
    let d = Dog("Rex")
    print(d.greet())
'''
    out = compile_and_run(src)
    assert out.strip() == "Woof! I am Rex"

@test("e2e: multiple classes")
def _():
    src = '''class Cat(Object):
    fn __init__(self, name: Str):
        self.name = name

    fn speak(self) -> Str:
        return self.name + " says meow"

class Dog(Object):
    fn __init__(self, name: Str):
        self.name = name

    fn speak(self) -> Str:
        return self.name + " says woof"

fn main():
    let c = Cat("Whiskers")
    let d = Dog("Rex")
    print(c.speak())
    print(d.speak())
'''
    out = compile_and_run(src)
    assert out.strip() == "Whiskers says meow\nRex says woof"

@test("e2e: class with double field")
def _():
    src = '''class Circle(Object):
    fn __init__(self, radius: Double):
        self.radius = radius

    fn area(self) -> Double:
        return 3.14159265 * self.radius * self.radius

fn main():
    let c = Circle(5.0)
    print(c.area())
'''
    out = compile_and_run(src)
    assert out.strip() == "78.5398"

@test("type: undefined field access")
def _():
    expect_compile_error(
        '''class Foo(Object):
    fn __init__(self):
        self.x = 1

fn main():
    let f = Foo()
    print(f.y)
''',
        "has no field 'y'"
    )

@test("type: undefined method call")
def _():
    expect_compile_error(
        '''class Foo(Object):
    fn __init__(self):
        self.x = 1

fn main():
    let f = Foo()
    f.bar()
''',
        "has no method 'bar'"
    )

@test("type: wrong constructor args")
def _():
    expect_compile_error(
        '''class Foo(Object):
    fn __init__(self, x: I64):
        self.x = x

fn main():
    let f = Foo()
''',
        "expects 1 args"
    )

@test("e2e: inheritance basic")
def _():
    src = '''class Animal(Object):
    fn __init__(self, name: Str):
        self.name = name

    fn who(self) -> Str:
        return self.name

class Dog(Animal):
    fn __init__(self, name: Str, breed: Str):
        self.name = name
        self.breed = breed

    fn info(self) -> Str:
        return self.name + " the " + self.breed

fn main():
    let d = Dog("Rex", "Labrador")
    print(d.who())
    print(d.info())
'''
    out = compile_and_run(src)
    assert out.strip() == "Rex\nRex the Labrador"


# === Phase 3: super calls ===

@test("e2e: super.__init__")
def _():
    src = '''class Animal(Object):
    fn __init__(self, name: Str):
        self.name = name

    fn speak(self) -> Str:
        return self.name + " speaks"

class Dog(Animal):
    fn __init__(self, name: Str, breed: Str):
        super.__init__(name)
        self.breed = breed

    fn speak(self) -> Str:
        return super.speak() + " (woof!)"

fn main():
    let d = Dog("Rex", "Labrador")
    print(d.speak())
    print(d.name)
    print(d.breed)
'''
    out = compile_and_run(src)
    assert out.strip() == "Rex speaks (woof!)\nRex\nLabrador"

@test("e2e: super method call")
def _():
    src = '''class Base(Object):
    fn __init__(self, x: I64):
        self.x = x

    fn value(self) -> I64:
        return self.x

class Derived(Base):
    fn __init__(self, x: I64, y: I64):
        super.__init__(x)
        self.y = y

    fn value(self) -> I64:
        return super.value() + self.y

fn main():
    let d = Derived(10, 20)
    print(d.value())
'''
    out = compile_and_run(src)
    assert out.strip() == "30"


# === Phase 3: operator methods ===

@test("e2e: __add__ operator")
def _():
    src = '''class Vec2(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y

    fn __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

fn main():
    let a = Vec2(1, 2)
    let b = Vec2(3, 4)
    let c = a + b
    print(c.x)
    print(c.y)
'''
    out = compile_and_run(src)
    assert out.strip() == "4\n6"

@test("e2e: __eq__ and __ne__ operators")
def _():
    src = '''class Box(Object):
    fn __init__(self, val: I64):
        self.val = val

    fn __eq__(self, other: Box) -> Bool:
        return self.val == other.val

fn main():
    let a = Box(42)
    let b = Box(42)
    let c = Box(99)
    print(a == b)
    print(a != c)
    print(a == c)
'''
    out = compile_and_run(src)
    assert out.strip() == "true\ntrue\nfalse"

@test("e2e: __lt__ operator")
def _():
    src = '''class Score(Object):
    fn __init__(self, pts: I64):
        self.pts = pts

    fn __lt__(self, other: Score) -> Bool:
        return self.pts < other.pts

fn main():
    let a = Score(10)
    let b = Score(20)
    if a < b:
        print(1)
    else:
        print(0)
'''
    out = compile_and_run(src)
    assert out.strip() == "1"

@test("e2e: __neg__ operator")
def _():
    src = '''class Num(Object):
    fn __init__(self, val: I64):
        self.val = val

    fn __neg__(self) -> Num:
        return Num(0 - self.val)

fn main():
    let a = Num(42)
    let b = -a
    print(b.val)
'''
    out = compile_and_run(src)
    assert out.strip() == "-42"

@test("e2e: __sub__ and __mul__ operators")
def _():
    src = '''class Vec2(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y

    fn __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    fn __mul__(self, other: Vec2) -> Vec2:
        return Vec2(self.x * other.x, self.y * other.y)

fn main():
    let a = Vec2(10, 20)
    let b = Vec2(3, 5)
    let c = a - b
    let d = a * b
    print(c.x)
    print(c.y)
    print(d.x)
    print(d.y)
'''
    out = compile_and_run(src)
    assert out.strip() == "7\n15\n30\n100"


# === Phase 4: for loops, casting, to_str, arrays ===

@test("e2e: for-in-range basic")
def _():
    src = '''fn main():
    for i in range(5):
        print(i)
'''
    out = compile_and_run(src)
    assert out.strip() == "0\n1\n2\n3\n4"

@test("e2e: for-in-range with start and end")
def _():
    src = '''fn main():
    for i in range(3, 7):
        print(i)
'''
    out = compile_and_run(src)
    assert out.strip() == "3\n4\n5\n6"

@test("e2e: for loop sum")
def _():
    src = '''fn main():
    let sum: I64 = 0
    for i in range(1, 11):
        sum = sum + i
    print(sum)
'''
    out = compile_and_run(src)
    assert out.strip() == "55"

@test("e2e: nested for loops")
def _():
    src = '''fn main():
    let count: I64 = 0
    for i in range(3):
        for j in range(4):
            count = count + 1
    print(count)
'''
    out = compile_and_run(src)
    assert out.strip() == "12"

@test("e2e: as cast int narrowing")
def _():
    src = '''fn main():
    let x: I64 = 42
    let y: I32 = x as I32
    print(y)
'''
    out = compile_and_run(src)
    assert out.strip() == "42"

@test("e2e: as cast int to double")
def _():
    src = '''fn main():
    let x: I64 = 100
    let d: Double = x as Double
    print(d)
'''
    out = compile_and_run(src)
    assert out.strip() == "100"

@test("e2e: as cast double to int")
def _():
    src = '''fn main():
    let d: Double = 3.99
    let x: I64 = d as I64
    print(x)
'''
    out = compile_and_run(src)
    assert out.strip() == "3"

@test("e2e: as cast float widening")
def _():
    src = '''fn main():
    let f: Float = 2.5f
    let d: Double = f as Double
    print(d)
'''
    out = compile_and_run(src)
    assert out.strip() == "2.5"

@test("type error: invalid cast")
def _():
    expect_compile_error('''fn main():
    let s: Str = "hello"
    let x: I64 = s as I64
''', "cannot cast")

@test("e2e: to_str on I64")
def _():
    src = '''fn main():
    let x: I64 = 42
    let s: Str = x.to_str()
    print(s)
'''
    out = compile_and_run(src)
    assert out.strip() == "42"

@test("e2e: to_str on Double")
def _():
    src = '''fn main():
    let d: Double = 3.14
    print(d.to_str())
'''
    out = compile_and_run(src)
    assert out.strip() == "3.14"

@test("e2e: to_str on Bool")
def _():
    src = '''fn main():
    let b: Bool = true
    print(b.to_str())
'''
    out = compile_and_run(src)
    assert out.strip() == "true"

@test("e2e: to_str string concatenation")
def _():
    src = '''fn main():
    let x: I64 = 42
    print(x.to_str() + " is the answer")
'''
    out = compile_and_run(src)
    assert out.strip() == "42 is the answer"

@test("e2e: Array[I64] basic")
def _():
    src = '''fn main():
    let a = Array[I64]()
    a.push(10)
    a.push(20)
    a.push(30)
    print(a.len())
    print(a.get(0))
    print(a.get(1))
    print(a.get(2))
'''
    out = compile_and_run(src)
    assert out.strip() == "3\n10\n20\n30"

@test("e2e: Array set")
def _():
    src = '''fn main():
    let a = Array[I64]()
    a.push(1)
    a.push(2)
    a.set(0, 99)
    print(a.get(0))
    print(a.get(1))
'''
    out = compile_and_run(src)
    assert out.strip() == "99\n2"

@test("e2e: Array growth")
def _():
    src = '''fn main():
    let a = Array[I64]()
    for i in range(20):
        a.push(i * i)
    print(a.len())
    print(a.get(0))
    print(a.get(9))
    print(a.get(19))
'''
    out = compile_and_run(src)
    assert out.strip() == "20\n0\n81\n361"

@test("e2e: Array[Str]")
def _():
    src = '''fn main():
    let a = Array[Str]()
    a.push("hello")
    a.push("world")
    print(a.get(0))
    print(a.get(1))
    print(a.len())
'''
    out = compile_and_run(src)
    assert out.strip() == "hello\nworld\n2"

@test("e2e: Array[Double]")
def _():
    src = '''fn main():
    let a = Array[Double]()
    a.push(1.5)
    a.push(2.5)
    print(a.get(0))
    print(a.get(1))
'''
    out = compile_and_run(src)
    assert out.strip() == "1.5\n2.5"

@test("e2e: Array with for loop")
def _():
    src = '''fn main():
    let a = Array[I64]()
    for i in range(5):
        a.push(i + 1)
    let sum: I64 = 0
    for i in range(a.len()):
        sum = sum + a.get(i)
    print(sum)
'''
    out = compile_and_run(src)
    assert out.strip() == "15"

@test("e2e: Array as function parameter")
def _():
    src = '''fn sum_array(arr: Array[I64]) -> I64:
    let total: I64 = 0
    for i in range(arr.len()):
        total = total + arr.get(i)
    return total

fn main():
    let a = Array[I64]()
    a.push(10)
    a.push(20)
    a.push(30)
    print(sum_array(a))
'''
    out = compile_and_run(src)
    assert out.strip() == "60"

@test("type error: Array push wrong type")
def _():
    expect_compile_error('''fn main():
    let a = Array[I64]()
    a.push("hello")
''', "push() expects I64")

@test("type error: Array unknown method")
def _():
    expect_compile_error('''fn main():
    let a = Array[I64]()
    a.pop()
''', "Array has no method 'pop'")

@test("e2e: for loop + array + classes")
def _():
    src = '''class Point(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y

fn main():
    let points = Array[Point]()
    for i in range(3):
        points.push(Point(i, i * 2))
    for i in range(points.len()):
        let p = points.get(i)
        print(p.x)
        print(p.y)
'''
    out = compile_and_run(src)
    assert out.strip() == "0\n0\n1\n2\n2\n4"


# === Phase 4b: Imports/Modules ===

@test("e2e: import function")
def _():
    modules = {"mymath.sen": '''fn double(x: I64) -> I64:
    return x * 2

fn triple(x: I64) -> I64:
    return x * 3
'''}
    src = '''import "mymath.sen"

fn main():
    print(mymath.double(5))
    print(mymath.triple(5))
'''
    out = compile_and_run_with_modules(src, modules)
    assert out.strip() == "10\n15"

@test("e2e: import class")
def _():
    modules = {"shapes.sen": '''class Rect(Object):
    fn __init__(self, w: I64, h: I64):
        self.w = w
        self.h = h

    fn area(self) -> I64:
        return self.w * self.h
'''}
    src = '''import "shapes.sen"

fn main():
    let r = shapes.Rect(3, 4)
    print(r.area())
    print(r.w)
'''
    out = compile_and_run_with_modules(src, modules)
    assert out.strip() == "12\n3"

@test("e2e: import class with operator")
def _():
    modules = {"vec.sen": '''class Vec2(Object):
    fn __init__(self, x: I64, y: I64):
        self.x = x
        self.y = y

    fn __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)
'''}
    src = '''import "vec.sen"

fn main():
    let a = vec.Vec2(1, 2)
    let b = vec.Vec2(3, 4)
    let c = a + b
    print(c.x)
    print(c.y)
'''
    out = compile_and_run_with_modules(src, modules)
    assert out.strip() == "4\n6"

@test("e2e: import multiple modules")
def _():
    modules = {
        "mod_a.sen": '''fn greet() -> Str:
    return "hello"
''',
        "mod_b.sen": '''fn farewell() -> Str:
    return "goodbye"
''',
    }
    src = '''import "mod_a.sen"
import "mod_b.sen"

fn main():
    print(mod_a.greet())
    print(mod_b.farewell())
'''
    out = compile_and_run_with_modules(src, modules)
    assert out.strip() == "hello\ngoodbye"

@test("e2e: import ignores module main")
def _():
    modules = {"lib.sen": '''fn helper() -> I64:
    return 42

fn main():
    print(0)
'''}
    src = '''import "lib.sen"

fn main():
    print(lib.helper())
'''
    out = compile_and_run_with_modules(src, modules)
    assert out.strip() == "42"

@test("type error: import nonexistent module")
def _():
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        expect_compile_error('''import "nonexistent.sen"

fn main():
    print(1)
''', "cannot find module", source_dir=tmpdir)

@test("type error: import unknown function")
def _():
    modules = {"lib.sen": '''fn foo() -> I64:
    return 1
'''}
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, msrc in modules.items():
            Path(tmpdir, fname).write_text(msrc)
        expect_compile_error('''import "lib.sen"

fn main():
    lib.bar()
''', "has no 'bar'", source_dir=tmpdir)


# === Report ===

print(f"\n{'='*40}")
print(f"  {passed} passed, {failed} failed")
print(f"{'='*40}")
sys.exit(1 if failed else 0)
