#!/usr/bin/env python3
"""Test suite for the Funk VM and assembler."""

import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from funklang.opcodes import Opcode, Instruction
from funklang.vm import VM, FunkError
from funklang.assembler import assemble, assemble_file

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")


def run_vm(instructions):
    """Helper: run instructions and return the VM."""
    vm = VM(instructions)
    vm.run()
    return vm


def capture_output(fn):
    """Capture stdout from fn() and return it as a string."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        fn()
        return sys.stdout.getvalue()
    finally:
        sys.stdout = old


# ── Stack operations ──

def test_push_int():
    vm = run_vm([Instruction(Opcode.PUSH_INT, 42), Instruction(Opcode.HALT)])
    assert vm.data_stack == [42], f"expected [42], got {vm.data_stack}"

def test_push_float():
    vm = run_vm([Instruction(Opcode.PUSH_FLOAT, 3.14), Instruction(Opcode.HALT)])
    assert vm.data_stack == [3.14]

def test_push_str():
    vm = run_vm([Instruction(Opcode.PUSH_STR, "hello"), Instruction(Opcode.HALT)])
    assert vm.data_stack == ["hello"]

def test_pop():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.PUSH_INT, 2),
        Instruction(Opcode.POP),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_dup():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 7),
        Instruction(Opcode.DUP),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [7, 7]

def test_swap():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.PUSH_INT, 2),
        Instruction(Opcode.SWAP),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [2, 1]

def test_over():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.PUSH_INT, 2),
        Instruction(Opcode.OVER),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1, 2, 1]

# ── Arithmetic ──

def test_add_int():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.PUSH_INT, 4),
        Instruction(Opcode.ADD),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [7]

def test_add_float():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.PUSH_FLOAT, 0.5),
        Instruction(Opcode.ADD),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [3.5]

def test_add_str():
    vm = run_vm([
        Instruction(Opcode.PUSH_STR, "he"),
        Instruction(Opcode.PUSH_STR, "llo"),
        Instruction(Opcode.ADD),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == ["hello"]

def test_sub():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 10),
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.SUB),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [7], f"expected [7], got {vm.data_stack}"

def test_mul():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 6),
        Instruction(Opcode.PUSH_INT, 7),
        Instruction(Opcode.MUL),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [42]

def test_div_int():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 10),
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.DIV),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [3]  # integer division

def test_div_float():
    vm = run_vm([
        Instruction(Opcode.PUSH_FLOAT, 10.0),
        Instruction(Opcode.PUSH_INT, 4),
        Instruction(Opcode.DIV),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [2.5]

def test_mod():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 10),
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.MOD),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_div_by_zero():
    try:
        run_vm([
            Instruction(Opcode.PUSH_INT, 1),
            Instruction(Opcode.PUSH_INT, 0),
            Instruction(Opcode.DIV),
        ])
        assert False, "expected FunkError"
    except FunkError as e:
        assert "division by zero" in str(e)

# ── Comparison / logic ──

def test_eq_true():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 5),
        Instruction(Opcode.PUSH_INT, 5),
        Instruction(Opcode.EQ),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_eq_false():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 5),
        Instruction(Opcode.PUSH_INT, 6),
        Instruction(Opcode.EQ),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [0]

def test_lt():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.PUSH_INT, 5),
        Instruction(Opcode.LT),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_gt():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 5),
        Instruction(Opcode.PUSH_INT, 3),
        Instruction(Opcode.GT),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_not():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 0),
        Instruction(Opcode.NOT),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_not_truthy():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 42),
        Instruction(Opcode.NOT),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [0]

def test_and():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.AND),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

def test_and_false():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.PUSH_INT, 0),
        Instruction(Opcode.AND),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [0]

def test_or():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 0),
        Instruction(Opcode.PUSH_INT, 1),
        Instruction(Opcode.OR),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [1]

# ── Control flow ──

def test_jmp():
    vm = run_vm([
        Instruction(Opcode.JMP, 2),          # 0: jump to HALT
        Instruction(Opcode.PUSH_INT, 999),    # 1: skipped
        Instruction(Opcode.HALT),             # 2: halt
    ])
    assert vm.data_stack == []

def test_jmp_if():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 1),      # 0: truthy
        Instruction(Opcode.JMP_IF, 3),         # 1: jump to index 3
        Instruction(Opcode.PUSH_INT, 999),     # 2: skipped
        Instruction(Opcode.HALT),              # 3: halt
    ])
    assert vm.data_stack == []

def test_jmp_if_not():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 0),       # 0: falsy
        Instruction(Opcode.JMP_IF_NOT, 3),     # 1: jump to index 3
        Instruction(Opcode.PUSH_INT, 999),     # 2: skipped
        Instruction(Opcode.HALT),              # 3: halt
    ])
    assert vm.data_stack == []

def test_loop():
    """Sum 1+2+3+4+5 using a loop."""
    program = assemble("""\
PUSH_INT 0
STORE 0        ; sum = 0
PUSH_INT 5
STORE 1        ; i = 5
loop:
  LOAD 0
  LOAD 1
  ADD
  STORE 0      ; sum += i
  LOAD 1
  PUSH_INT 1
  SUB
  DUP
  STORE 1      ; i--
  PUSH_INT 0
  GT           ; i > 0?
  JMP_IF loop
LOAD 0
HALT
""")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [15], f"expected [15], got {vm.data_stack}"

# ── Local variables ──

def test_store_load():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 42),
        Instruction(Opcode.STORE, 0),
        Instruction(Opcode.LOAD, 0),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [42]

def test_uninitialized_local():
    try:
        run_vm([
            Instruction(Opcode.LOAD, 0),
        ])
        assert False, "expected FunkError"
    except FunkError as e:
        assert "uninitialized" in str(e)

# ── Functions ──

def test_call_ret():
    """Simple function: double the top of stack."""
    program = assemble("""\
PUSH_INT 21
CALL double
HALT

double:
  PUSH_INT 2
  MUL
  RET
""")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [42], f"expected [42], got {vm.data_stack}"

def test_factorial():
    """Recursive factorial(5) = 120."""
    program = assemble("""\
PUSH_INT 5
CALL factorial
HALT

factorial:
  DUP
  PUSH_INT 2
  LT
  JMP_IF base
  DUP
  PUSH_INT 1
  SUB
  CALL factorial
  MUL
  RET
base:
  POP
  PUSH_INT 1
  RET
""")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [120], f"expected [120], got {vm.data_stack}"

# ── Error cases ──

def test_stack_underflow():
    try:
        run_vm([Instruction(Opcode.POP)])
        assert False, "expected FunkError"
    except FunkError as e:
        assert "stack underflow" in str(e)

def test_mod_by_zero():
    try:
        run_vm([
            Instruction(Opcode.PUSH_INT, 5),
            Instruction(Opcode.PUSH_INT, 0),
            Instruction(Opcode.MOD),
        ])
        assert False, "expected FunkError"
    except FunkError as e:
        assert "division by zero" in str(e)

# ── Assembler ──

def test_assembler_comments_and_blanks():
    program = assemble("""\
; this is a comment

PUSH_INT 1
; another comment
PUSH_INT 2
ADD
HALT
""")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [3]

def test_assembler_string_with_semicolon():
    """Semicolons inside string literals should not be treated as comments."""
    program = assemble("""\
PUSH_STR "hello; world"
HALT
""")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == ["hello; world"]

def test_assembler_unknown_opcode():
    try:
        assemble("FOOBAR 42\n")
        assert False, "expected FunkError"
    except FunkError as e:
        assert "unknown" in str(e).lower()

def test_assembler_undefined_label():
    try:
        assemble("JMP nowhere\n")
        assert False, "expected FunkError"
    except FunkError as e:
        assert "undefined" in str(e).lower()

def test_assembler_duplicate_label():
    try:
        assemble("foo:\nfoo:\nHALT\n")
        assert False, "expected FunkError"
    except FunkError as e:
        assert "duplicate" in str(e).lower()

def test_assembler_case_insensitive():
    program = assemble("push_int 42\nhalt\n")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [42]

# ── End-to-end: example programs ──

def test_hello_example():
    program = assemble_file(os.path.join(EXAMPLES_DIR, "hello.funk"))
    output = capture_output(lambda: VM(program).run())
    assert output.strip() == "Hello, World!", f"got: {output.strip()!r}"

def test_fib_example():
    program = assemble_file(os.path.join(EXAMPLES_DIR, "fib.funk"))
    output = capture_output(lambda: VM(program).run())
    lines = output.strip().split("\n")
    nums = [int(line) for line in lines]
    assert nums == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], f"got: {nums}"

def test_factorial_example():
    program = assemble_file(os.path.join(EXAMPLES_DIR, "factorial.funk"))
    output = capture_output(lambda: VM(program).run())
    assert output.strip() == "3628800", f"got: {output.strip()!r}"

# ── PRINT ──

def test_print_int():
    program = [Instruction(Opcode.PUSH_INT, 42), Instruction(Opcode.PRINT), Instruction(Opcode.HALT)]
    output = capture_output(lambda: VM(program).run())
    assert output.strip() == "42"

def test_print_str():
    program = [Instruction(Opcode.PUSH_STR, "hello"), Instruction(Opcode.PRINT), Instruction(Opcode.HALT)]
    output = capture_output(lambda: VM(program).run())
    assert output.strip() == "hello"  # no quotes

# ── Implicit halt ──

def test_implicit_halt():
    vm = run_vm([Instruction(Opcode.PUSH_INT, 1)])
    assert vm.data_stack == [1]
    assert not vm.halted  # didn't hit explicit HALT


# ── Run all tests ──

if __name__ == "__main__":
    print("Funk VM tests\n")

    # Collect all test functions
    test_fns = [(name, obj) for name, obj in sorted(globals().items())
                if name.startswith("test_") and callable(obj)]

    for name, fn in test_fns:
        test(name, fn)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
