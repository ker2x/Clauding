#!/usr/bin/env python3
"""Unit tests for the Kouhai compiler (lex/parse/typecheck). Standalone, no pytest.

E2E tests live in tests/cases/ and are run via: make -j16 -C tests test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kouhai.tokens import lex, LexError, TT
from kouhai.parser import Parser, ParseError
from kouhai.types import check_program, TypeError_
from kouhai.compiler import compile_source, CompileError


# --- Helpers ---

def expect_compile_error(source: str, expected_fragment: str, source_dir: str | None = None):
    """Assert that compilation fails with an error containing the fragment."""
    try:
        compile_source(source, source_dir=source_dir)
        assert False, f"expected CompileError containing '{expected_fragment}'"
    except CompileError as e:
        assert expected_fragment in str(e), f"expected '{expected_fragment}' in '{e}'"


tests = []

def test(name):
    def decorator(fn):
        tests.append((name, fn))
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
    from kouhai.ast_nodes import LetStmt, IntLit
    tokens = lex('fn main():\n    let x: I64 = 42\n')
    prog = Parser(tokens).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert stmt.name == "x"
    assert isinstance(stmt.value, IntLit)

@test("parse: if/elif/else")
def _():
    from kouhai.ast_nodes import IfStmt
    src = 'fn main():\n    if true:\n        return\n    elif false:\n        return\n    else:\n        return\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, IfStmt)
    assert len(stmt.elif_clauses) == 1
    assert len(stmt.else_body) == 1

@test("parse: while loop")
def _():
    from kouhai.ast_nodes import WhileStmt
    src = 'fn main():\n    while true:\n        return\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, WhileStmt)

@test("parse: operator precedence")
def _():
    from kouhai.ast_nodes import BinOp
    src = 'fn main():\n    let x: I64 = 1 + 2 * 3\n'
    prog = Parser(lex(src)).parse()
    from kouhai.ast_nodes import LetStmt
    stmt = prog.functions[0].body[0]
    # Should be Add(1, Mul(2, 3))
    assert isinstance(stmt.value, BinOp)
    assert stmt.value.op == "+"
    assert isinstance(stmt.value.right, BinOp)
    assert stmt.value.right.op == "*"

@test("parse: unary minus")
def _():
    from kouhai.ast_nodes import UnaryOp
    src = 'fn main():\n    let x: I64 = -42\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt.value, UnaryOp)
    assert stmt.value.op == "-"

@test("parse: function call")
def _():
    from kouhai.ast_nodes import ExprStmt, Call
    src = 'fn foo(x: I64) -> I64:\n    return x\nfn main():\n    foo(42)\n'
    prog = Parser(lex(src)).parse()
    stmt = prog.functions[1].body[0]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, Call)
    assert stmt.expr.func == "foo"

@test("parse: nested expressions with parens")
def _():
    from kouhai.ast_nodes import BinOp
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


# === Run tests ===

if __name__ == "__main__":
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")
    sys.exit(1 if failed else 0)
