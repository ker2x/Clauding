"""Stack-based virtual machine for Funk."""

from typing import Any

from .opcodes import Instruction, Opcode


class FunkError(Exception):
    """Runtime error in the Funk VM."""


class Frame:
    """Call stack frame with return address and local variable slots."""

    __slots__ = ("return_addr", "locals")

    def __init__(self, return_addr: int, num_locals: int = 256):
        self.return_addr = return_addr
        self.locals: list[Any] = [None] * num_locals


class VM:
    """Funk virtual machine.

    Executes a program represented as a list of Instructions.
    Uses a data stack for operands and a call stack for function frames.
    """

    def __init__(self, program: list[Instruction]):
        self.program = program
        self.data_stack: list[Any] = []
        self.call_stack: list[Frame] = [Frame(return_addr=-1)]  # top-level frame
        self.ip = 0
        self.halted = False

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

    def run(self) -> None:
        """Execute the loaded program until HALT or end of program."""
        while not self.halted:
            if self.ip >= len(self.program):
                break  # implicit halt
            self._step()

    def _step(self) -> None:
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
            slot = instr.operand
            frame = self.call_stack[-1]
            val = frame.locals[slot]
            if val is None:
                raise FunkError(f"uninitialized local variable in slot {slot}")
            self._push(val)
        elif op == Opcode.STORE:
            slot = instr.operand
            frame = self.call_stack[-1]
            frame.locals[slot] = self._pop()

        # -- I/O --
        elif op == Opcode.PRINT:
            val = self._pop()
            print(val)

        # -- System --
        elif op == Opcode.HALT:
            self.halted = True

        else:
            raise FunkError(f"unknown opcode: {op:#x}")
