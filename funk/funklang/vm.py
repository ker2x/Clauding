"""Stack-based virtual machine for Funk."""

import ctypes
import ctypes.util
from typing import Any

from .opcodes import ExternDecl, Instruction, Opcode

# C type mapping for FFI
_CTYPE_MAP = {
    "int": ctypes.c_int,
    "long": ctypes.c_long,
    "float": ctypes.c_double,
    "str": ctypes.c_char_p,
    "handle": ctypes.c_void_p,
}


class FunkError(Exception):
    """Runtime error in the Funk VM."""


class Frame:
    """Call stack frame with return address and named local variables."""

    __slots__ = ("return_addr", "locals")

    def __init__(self, return_addr: int):
        self.return_addr = return_addr
        self.locals: dict[str, Any] = {}


class VM:
    """Funk virtual machine.

    Executes a program represented as a list of Instructions.
    Uses a data stack for operands and a call stack for function frames.
    """

    def __init__(self, program: list[Instruction], on_print=None):
        self.program = program
        self.data_stack: list[Any] = []
        self.call_stack: list[Frame] = [Frame(return_addr=-1)]  # top-level frame
        self.ip = 0
        self.halted = False
        self.on_print = on_print  # callback for PRINT output
        self.exception_stack: list[int] = []  # catch addresses
        self._lib_cache: dict[str, ctypes.CDLL] = {}  # FFI library cache

    # -- FFI helpers --

    def _get_lib(self, name: str) -> ctypes.CDLL:
        if name not in self._lib_cache:
            if name == "libc":
                self._lib_cache[name] = ctypes.CDLL(ctypes.util.find_library("c"))
            else:
                self._lib_cache[name] = ctypes.CDLL(name)
        return self._lib_cache[name]

    # -- Stack helpers --

    def _push(self, value: Any) -> None:
        self.data_stack.append(value)

    def _pop(self) -> Any:
        if not self.data_stack:
            raise FunkError("stack underflow")
        return self.data_stack.pop()

    def _peek(self) -> Any:
        if not self.data_stack:
            raise FunkError("stack underflow")
        return self.data_stack[-1]

    # -- Main loop --

    @property
    def done(self) -> bool:
        return self.halted or self.ip >= len(self.program)

    def step(self) -> bool:
        """Execute one instruction. Returns True if still running."""
        if self.done:
            return False
        self._step()
        return not self.done

    def run(self) -> None:
        """Execute the loaded program until HALT or end of program."""
        while not self.done:
            self._step()

    def _step(self) -> None:
        try:
            self._execute()
        except FunkError:
            if self.exception_stack:
                self.ip = self.exception_stack.pop()
                self._push(1)  # error flag: 1 = error occurred
            else:
                raise

    def _execute(self) -> None:
        instr = self.program[self.ip]
        op = instr.opcode
        self.ip += 1

        # -- Stack operations --
        if op == Opcode.PUSH_INT:
            self._push(instr.operand)
        elif op == Opcode.PUSH_FLOAT:
            self._push(instr.operand)
        elif op == Opcode.PUSH_STR:
            self._push(instr.operand)
        elif op == Opcode.POP:
            self._pop()
        elif op == Opcode.DUP:
            self._push(self._peek())
        elif op == Opcode.SWAP:
            b = self._pop()
            a = self._pop()
            self._push(b)
            self._push(a)
        elif op == Opcode.OVER:
            b = self._pop()
            a = self._peek()
            self._push(b)
            self._push(a)
        elif op == Opcode.ROT:
            c = self._pop()
            b = self._pop()
            a = self._pop()
            self._push(b)
            self._push(c)
            self._push(a)

        # -- Arithmetic --
        elif op == Opcode.ADD:
            b = self._pop()
            a = self._pop()
            self._push(a + b)
        elif op == Opcode.SUB:
            b = self._pop()
            a = self._pop()
            self._push(a - b)
        elif op == Opcode.MUL:
            b = self._pop()
            a = self._pop()
            self._push(a * b)
        elif op == Opcode.DIV:
            b = self._pop()
            a = self._pop()
            if b == 0:
                raise FunkError("division by zero")
            if isinstance(a, int) and isinstance(b, int):
                self._push(a // b)
            else:
                self._push(a / b)
        elif op == Opcode.MOD:
            b = self._pop()
            a = self._pop()
            if b == 0:
                raise FunkError("division by zero")
            self._push(a % b)

        # -- Comparison / logic --
        elif op == Opcode.EQ:
            b = self._pop()
            a = self._pop()
            self._push(1 if a == b else 0)
        elif op == Opcode.LT:
            b = self._pop()
            a = self._pop()
            self._push(1 if a < b else 0)
        elif op == Opcode.GT:
            b = self._pop()
            a = self._pop()
            self._push(1 if a > b else 0)
        elif op == Opcode.NOT:
            a = self._pop()
            self._push(1 if not a else 0)
        elif op == Opcode.AND:
            b = self._pop()
            a = self._pop()
            self._push(1 if a and b else 0)
        elif op == Opcode.OR:
            b = self._pop()
            a = self._pop()
            self._push(1 if a or b else 0)

        # -- Control flow --
        elif op == Opcode.JMP:
            self.ip = instr.operand
        elif op == Opcode.JMP_IF:
            cond = self._pop()
            if cond:
                self.ip = instr.operand
        elif op == Opcode.JMP_IF_NOT:
            cond = self._pop()
            if not cond:
                self.ip = instr.operand

        # -- Functions --
        elif op == Opcode.CALL:
            frame = Frame(return_addr=self.ip)
            self.call_stack.append(frame)
            self.ip = instr.operand
        elif op == Opcode.RET:
            if len(self.call_stack) <= 1:
                raise FunkError("RET with no call frame")
            frame = self.call_stack.pop()
            self.ip = frame.return_addr

        # -- Variables --
        elif op == Opcode.LOAD:
            name = instr.operand
            frame = self.call_stack[-1]
            if name not in frame.locals:
                raise FunkError(f"undefined variable '{name}'")
            self._push(frame.locals[name])
        elif op == Opcode.STORE:
            name = instr.operand
            frame = self.call_stack[-1]
            frame.locals[name] = self._pop()

        # -- I/O --
        elif op == Opcode.PRINT:
            val = self._pop()
            if self.on_print:
                self.on_print(val)
            else:
                print(val)

        # -- Type conversion --
        elif op == Opcode.TO_INT:
            a = self._pop()
            if isinstance(a, int):
                self._push(a)
            elif isinstance(a, float):
                self._push(int(a))
            elif isinstance(a, str):
                try:
                    self._push(int(a))
                except ValueError:
                    raise FunkError(f"cannot convert '{a}' to int")
            else:
                raise FunkError(f"cannot convert to int")
        elif op == Opcode.TO_FLOAT:
            a = self._pop()
            if isinstance(a, float):
                self._push(a)
            elif isinstance(a, int):
                self._push(float(a))
            elif isinstance(a, str):
                try:
                    self._push(float(a))
                except ValueError:
                    raise FunkError(f"cannot convert '{a}' to float")
            else:
                raise FunkError(f"cannot convert to float")
        elif op == Opcode.TO_STR:
            a = self._pop()
            self._push(str(a))
        elif op == Opcode.CHR:
            a = self._pop()
            if not isinstance(a, int):
                raise FunkError(f"CHR requires an integer, got {type(a).__name__}")
            try:
                self._push(chr(a))
            except (ValueError, OverflowError):
                raise FunkError(f"CHR: invalid code point {a}")
        elif op == Opcode.ORD:
            a = self._pop()
            if not isinstance(a, str) or len(a) != 1:
                raise FunkError(f"ORD requires a single character string")
            self._push(ord(a))

        # -- Error handling --
        elif op == Opcode.TRY:
            self.exception_stack.append(instr.operand)
        elif op == Opcode.CATCH:
            if not self.exception_stack:
                raise FunkError("CATCH without matching TRY")
            self.exception_stack.pop()
            self._push(0)  # error flag: 0 = no error

        # -- FFI --
        elif op == Opcode.CALL_NATIVE:
            decl: ExternDecl = instr.operand
            lib = self._get_lib(decl.library)
            func = getattr(lib, decl.func_name)
            func.argtypes = [_CTYPE_MAP[t] for t in decl.arg_types]
            func.restype = _CTYPE_MAP[decl.ret_type] if decl.ret_type != "void" else None
            # Pop args (last arg on top of stack, first arg deepest)
            args = [self._pop() for _ in decl.arg_types]
            args.reverse()
            # Marshal: str → bytes for c_char_p
            c_args = []
            for val, typ in zip(args, decl.arg_types):
                if typ == "str":
                    c_args.append(val.encode("utf-8") if isinstance(val, str) else val)
                else:
                    c_args.append(val)
            result = func(*c_args)
            # Push result (unless void)
            if decl.ret_type != "void":
                if decl.ret_type == "str":
                    self._push(result.decode("utf-8") if isinstance(result, bytes) else result)
                else:
                    self._push(result)

        # -- System --
        elif op == Opcode.HALT:
            self.halted = True

        else:
            raise FunkError(f"unknown opcode: {op:#x}")
