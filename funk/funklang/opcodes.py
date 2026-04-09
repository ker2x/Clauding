"""Instruction set definitions for the Funk VM."""

from enum import IntEnum
from typing import Any, NamedTuple


class Opcode(IntEnum):
    # Stack operations
    PUSH_INT = 0x01
    PUSH_FLOAT = 0x02
    PUSH_STR = 0x03
    POP = 0x04
    DUP = 0x05
    SWAP = 0x06
    OVER = 0x07
    ROT = 0x08

    # Arithmetic
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14

    # Comparison / logic
    EQ = 0x20
    LT = 0x21
    GT = 0x22
    NOT = 0x23
    AND = 0x24
    OR = 0x25

    # Control flow
    JMP = 0x30
    JMP_IF = 0x31
    JMP_IF_NOT = 0x32

    # Functions
    CALL = 0x40
    RET = 0x41

    # Variables
    LOAD = 0x50
    STORE = 0x51

    # I/O
    PRINT = 0x60

    # Type conversion
    TO_INT = 0x61
    TO_FLOAT = 0x62
    TO_STR = 0x63
    CHR = 0x64      # int → single-char string
    ORD = 0x65      # single-char string → int

    # Error handling
    TRY = 0x70
    CATCH = 0x71

    # FFI
    CALL_NATIVE = 0x80

    # System
    HALT = 0xFF


class ExternDecl(NamedTuple):
    """Declaration of a native C function for FFI."""
    library: str
    func_name: str
    arg_types: tuple[str, ...]
    ret_type: str


class Instruction(NamedTuple):
    opcode: Opcode
    operand: Any = None


# Opcodes that take a label reference as operand (resolved to address by assembler)
LABEL_OPCODES = {Opcode.JMP, Opcode.JMP_IF, Opcode.JMP_IF_NOT, Opcode.CALL, Opcode.TRY}

# Opcodes that take a variable name as operand
NAME_OPCODES = {Opcode.LOAD, Opcode.STORE}

# Opcodes that take a typed literal as operand
LITERAL_OPCODES = {Opcode.PUSH_INT, Opcode.PUSH_FLOAT, Opcode.PUSH_STR}
