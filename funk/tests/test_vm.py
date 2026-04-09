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


def run_asm(source):
    """Assemble source and run to completion, return the VM."""
    program = assemble(source)
    vm = VM(program)
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
STORE sum      ; sum = 0
PUSH_INT 5
STORE i        ; i = 5
loop:
  LOAD sum
  LOAD i
  ADD
  STORE sum    ; sum += i
  LOAD i
  PUSH_INT 1
  SUB
  DUP
  STORE i      ; i--
  PUSH_INT 0
  GT           ; i > 0?
  JMP_IF loop
LOAD sum
HALT
""")
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [15], f"expected [15], got {vm.data_stack}"

# ── Local variables ──

def test_store_load():
    vm = run_vm([
        Instruction(Opcode.PUSH_INT, 42),
        Instruction(Opcode.STORE, "x"),
        Instruction(Opcode.LOAD, "x"),
        Instruction(Opcode.HALT),
    ])
    assert vm.data_stack == [42]

def test_undefined_variable():
    try:
        run_vm([
            Instruction(Opcode.LOAD, "nope"),
        ])
        assert False, "expected FunkError"
    except FunkError as e:
        assert "undefined" in str(e)

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


# ── TRY/CATCH ──

def test_try_catch_error():
    """TRY/CATCH catches a division by zero."""
    source = """
    TRY handler
      PUSH_INT 10
      PUSH_INT 0
      DIV
    CATCH
    POP
    JMP done
    handler:
      POP
      PUSH_STR "caught"
    done:
      HALT
    """
    vm = run_asm(source)
    assert vm.data_stack == ["caught"]

def test_try_catch_no_error():
    """CATCH reached normally pushes 0 (no error)."""
    source = """
    TRY handler
      PUSH_INT 10
      PUSH_INT 2
      DIV
    CATCH
    HALT
    handler:
      HALT
    """
    vm = run_asm(source)
    # stack: 5 (result of 10/2), then 0 (no-error flag from CATCH)
    assert vm.data_stack == [5, 0]

def test_try_catch_nested():
    """Inner TRY/CATCH handles error, outer continues."""
    source = """
    TRY outer
      TRY inner
        PUSH_INT 1
        PUSH_INT 0
        DIV
      CATCH
      POP
      JMP after_inner
      inner:
        POP
        PUSH_STR "inner"
      after_inner:
    CATCH
    POP
    JMP done
    outer:
      POP
      PUSH_STR "outer"
    done:
      HALT
    """
    vm = run_asm(source)
    assert vm.data_stack == ["inner"]

def test_try_catch_unwind_to_outer():
    """Error after inner CATCH unwinds to outer handler."""
    source = """
    TRY outer
      TRY inner
        PUSH_INT 10
        PUSH_INT 5
        DIV
      CATCH
      POP
      ; now cause an error outside inner but inside outer
      PUSH_INT 1
      PUSH_INT 0
      DIV
    CATCH
    POP
    JMP done
    inner:
      POP
      PUSH_STR "inner"
      JMP done
    outer:
      POP
      PUSH_STR "outer"
    done:
      HALT
    """
    vm = run_asm(source)
    assert vm.data_stack == [2, "outer"]

def test_catch_without_try():
    """CATCH without TRY raises an error."""
    try:
        run_asm("CATCH\nHALT\n")
        assert False, "should have raised"
    except FunkError:
        pass

def test_uncaught_error_still_raises():
    """Without TRY, errors still propagate normally."""
    try:
        run_asm("PUSH_INT 1\nPUSH_INT 0\nDIV\nHALT\n")
        assert False, "should have raised"
    except FunkError:
        pass

def test_trycatch_example():
    program = assemble_file(os.path.join(EXAMPLES_DIR, "trycatch.funk"))
    output = capture_output(lambda: VM(program).run())
    lines = output.strip().split("\n")
    assert lines == [
        "Caught division by zero!",
        "3",
        "Inner catch handled it",
        "Outer try continues fine",
    ], f"got: {lines}"

# ── Assembler aliases ──

def test_ipush():
    vm = run_asm("IPUSH 99\nHALT\n")
    assert vm.data_stack == [99]

def test_fpush():
    vm = run_asm("FPUSH 2.5\nHALT\n")
    assert vm.data_stack == [2.5]

def test_spush():
    vm = run_asm('SPUSH "hi"\nHALT\n')
    assert vm.data_stack == ["hi"]

def test_say():
    output = capture_output(lambda: run_asm('SAY "hello world"\nHALT\n'))
    assert output.strip() == "hello world"

def test_say_no_stack_residue():
    """SAY should not leave anything on the stack."""
    vm_output = []
    program = assemble('SAY "test"\nHALT\n')
    vm = VM(program, on_print=lambda v: vm_output.append(str(v)))
    vm.run()
    assert vm.data_stack == []
    assert vm_output == ["test"]

def test_aliases_case_insensitive():
    vm = run_asm("ipush 7\nHALT\n")
    assert vm.data_stack == [7]

# ── INCLUDE ──

def test_include_stdlib():
    """INCLUDE resolves stdlib path."""
    program = assemble_file(os.path.join(EXAMPLES_DIR, "include_demo.funk"))
    def run_with_argc():
        vm = VM(program)
        vm._push(0)  # argc = 0, like the runner does
        vm.run()
    output = capture_output(run_with_argc)
    lines = output.strip().split("\n")
    assert lines == ["49", "15", "9", "3"], f"got: {lines}"

def test_include_relative():
    """INCLUDE resolves relative to the including file, with auto-namespacing."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "lib.funk"), "w") as f:
            f.write("double:\n  PUSH_INT 2\n  MUL\n  RET\n")
        with open(os.path.join(d, "main.funk"), "w") as f:
            f.write('JMP start\nINCLUDE "lib.funk"\nstart:\nPUSH_INT 5\nCALL lib.double\nHALT\n')
        program = assemble_file(os.path.join(d, "main.funk"))
        vm = VM(program)
        vm.run()
        assert vm.data_stack == [10]

def test_include_not_found():
    """INCLUDE of nonexistent file raises error."""
    try:
        assemble('INCLUDE "nope.funk"\nHALT\n')
        assert False, "should have raised"
    except FunkError as e:
        assert "not found" in str(e)

def test_include_idempotent():
    """Including the same file twice doesn't duplicate definitions."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "lib.funk"), "w") as f:
            f.write("noop:\n  RET\n")
        with open(os.path.join(d, "main.funk"), "w") as f:
            f.write('JMP start\nINCLUDE "lib.funk"\nINCLUDE "lib.funk"\nstart:\nHALT\n')
        program = assemble_file(os.path.join(d, "main.funk"))
        vm = VM(program)
        vm.run()

def test_include_namespace_isolation():
    """Labels from different includes don't collide."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "a.funk"), "w") as f:
            f.write("helper:\n  PUSH_INT 1\n  RET\n")
        with open(os.path.join(d, "b.funk"), "w") as f:
            f.write("helper:\n  PUSH_INT 2\n  RET\n")
        with open(os.path.join(d, "main.funk"), "w") as f:
            f.write('JMP start\nINCLUDE "a.funk"\nINCLUDE "b.funk"\nstart:\nCALL a.helper\nCALL b.helper\nADD\nHALT\n')
        program = assemble_file(os.path.join(d, "main.funk"))
        vm = VM(program)
        vm.run()
        assert vm.data_stack == [3]  # 1 + 2

def test_include_global_label():
    """Labels starting with . are global (not namespaced)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "lib.funk"), "w") as f:
            f.write(".shared:\n  PUSH_INT 99\n  RET\n")
        with open(os.path.join(d, "main.funk"), "w") as f:
            f.write('JMP start\nINCLUDE "lib.funk"\nstart:\nCALL shared\nHALT\n')
        program = assemble_file(os.path.join(d, "main.funk"))
        vm = VM(program)
        vm.run()
        assert vm.data_stack == [99]

# ── Comparison jump aliases ──

def test_jeq():
    vm = run_asm("PUSH_INT 5\nPUSH_INT 5\nJEQ yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jne():
    vm = run_asm("PUSH_INT 5\nPUSH_INT 3\nJNE yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jlt():
    vm = run_asm("PUSH_INT 3\nPUSH_INT 5\nJLT yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jgt():
    vm = run_asm("PUSH_INT 5\nPUSH_INT 3\nJGT yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jle():
    vm = run_asm("PUSH_INT 3\nPUSH_INT 5\nJLE yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jle_equal():
    vm = run_asm("PUSH_INT 5\nPUSH_INT 5\nJLE yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jge():
    vm = run_asm("PUSH_INT 5\nPUSH_INT 3\nJGE yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jge_equal():
    vm = run_asm("PUSH_INT 5\nPUSH_INT 5\nJGE yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

# ── Zero-comparison jump aliases ──

def test_jz():
    vm = run_asm("PUSH_INT 0\nJZ yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jz_nonzero():
    vm = run_asm("PUSH_INT 5\nJZ yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [0]

def test_jnz():
    vm = run_asm("PUSH_INT 7\nJNZ yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [1]

def test_jnz_zero():
    vm = run_asm("PUSH_INT 0\nJNZ yes\nPUSH_INT 0\nHALT\nyes:\nPUSH_INT 1\nHALT\n")
    assert vm.data_stack == [0]

# ── ROT ──

def test_rot():
    vm = run_asm("PUSH_INT 1\nPUSH_INT 2\nPUSH_INT 3\nROT\n")
    assert vm.data_stack == [2, 3, 1]

def test_rot_underflow():
    try:
        run_asm("PUSH_INT 1\nPUSH_INT 2\nROT\n")
        assert False, "should have raised"
    except FunkError:
        pass

# ── Type conversion ──

def test_to_int_from_float():
    vm = run_asm("PUSH_FLOAT 3.7\nTO_INT\n")
    assert vm.data_stack == [3]
    assert isinstance(vm.data_stack[0], int)

def test_to_int_from_str():
    vm = run_asm('PUSH_STR "42"\nTO_INT\n')
    assert vm.data_stack == [42]

def test_to_int_from_int():
    vm = run_asm("PUSH_INT 5\nTO_INT\n")
    assert vm.data_stack == [5]

def test_to_int_bad_str():
    try:
        run_asm('PUSH_STR "abc"\nTO_INT\n')
        assert False, "should have raised"
    except FunkError:
        pass

def test_to_float_from_int():
    vm = run_asm("PUSH_INT 5\nTO_FLOAT\n")
    assert vm.data_stack == [5.0]
    assert isinstance(vm.data_stack[0], float)

def test_to_float_from_str():
    vm = run_asm('PUSH_STR "3.14"\nTO_FLOAT\n')
    assert vm.data_stack[0] == 3.14

def test_to_float_bad_str():
    try:
        run_asm('PUSH_STR "nope"\nTO_FLOAT\n')
        assert False, "should have raised"
    except FunkError:
        pass

def test_to_str_from_int():
    vm = run_asm("PUSH_INT 42\nTO_STR\n")
    assert vm.data_stack == ["42"]

def test_to_str_from_float():
    vm = run_asm("PUSH_FLOAT 3.14\nTO_STR\n")
    assert vm.data_stack == ["3.14"]

def test_to_int_negative_str():
    vm = run_asm('PUSH_STR "-7"\nTO_INT\n')
    assert vm.data_stack == [-7]

# ── CHR / ORD ──

def test_chr():
    vm = run_asm("PUSH_INT 65\nCHR\n")
    assert vm.data_stack == ["A"]

def test_chr_space():
    vm = run_asm("PUSH_INT 32\nCHR\n")
    assert vm.data_stack == [" "]

def test_chr_invalid():
    try:
        run_asm("PUSH_INT -1\nCHR\n")
        assert False, "should have raised"
    except FunkError:
        pass

def test_ord():
    vm = run_asm('PUSH_STR "A"\nORD\n')
    assert vm.data_stack == [65]

def test_ord_newline():
    vm = run_asm('PUSH_INT 10\nCHR\nORD\n')
    assert vm.data_stack == [10]

def test_ord_not_single_char():
    try:
        run_asm('PUSH_STR "AB"\nORD\n')
        assert False, "should have raised"
    except FunkError:
        pass

def test_chr_ord_roundtrip():
    vm = run_asm("PUSH_INT 122\nCHR\nORD\n")
    assert vm.data_stack == [122]

# ── FFI (CALL_NATIVE) ──

def test_ffi_abs():
    vm = run_asm('EXTERN "libc" abs (int) -> int\nPUSH_INT -42\nCALL_NATIVE abs\n')
    assert vm.data_stack == [42]

def test_ffi_strlen():
    vm = run_asm('EXTERN "libc" strlen (str) -> long\nPUSH_STR "hello"\nCALL_NATIVE strlen\n')
    assert vm.data_stack == [5]

def test_ffi_no_args():
    # rand() returns some int — just verify it doesn't crash and pushes something
    vm = run_asm('EXTERN "libc" rand () -> int\nCALL_NATIVE rand\n')
    assert len(vm.data_stack) == 1
    assert isinstance(vm.data_stack[0], int)

def test_ffi_void_return():
    # srand() returns void — stack should be empty after call
    vm = run_asm('EXTERN "libc" srand (int) -> void\nPUSH_INT 42\nCALL_NATIVE srand\n')
    assert vm.data_stack == []

def test_ffi_undeclared():
    try:
        run_asm('CALL_NATIVE nope\n')
        assert False, "should have raised"
    except FunkError:
        pass

def test_ffi_bad_extern_syntax():
    try:
        run_asm('EXTERN bad\n')
        assert False, "should have raised"
    except FunkError:
        pass

def test_ffi_multiple_args():
    # strtol("123", NULL, 10) — but we can't pass NULL easily
    # Use atoi instead: atoi("123") -> 123
    vm = run_asm('EXTERN "libc" atoi (str) -> int\nPUSH_STR "123"\nCALL_NATIVE atoi\n')
    assert vm.data_stack == [123]

def test_ffi_atoi_negative():
    vm = run_asm('EXTERN "libc" atoi (str) -> int\nPUSH_STR "-99"\nCALL_NATIVE atoi\n')
    assert vm.data_stack == [-99]

# ── FFI stdlib ──

def test_string_length():
    source = 'JMP main\nINCLUDE "string.funk"\nmain:\nPUSH_STR "hello"\nCALL string.length\n'
    program = assemble(source, base_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stdlib"))
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [5]

def test_string_length_empty():
    source = 'JMP main\nINCLUDE "string.funk"\nmain:\nPUSH_STR ""\nCALL string.length\n'
    program = assemble(source, base_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stdlib"))
    vm = VM(program)
    vm.run()
    assert vm.data_stack == [0]

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
