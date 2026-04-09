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

    # System
    HALT = 0xFF


class Instruction(NamedTuple):
    opcode: Opcode
    operand: Any = None


# Opcodes that take a label reference as operand (resolved to address by assembler)
LABEL_OPCODES = {Opcode.JMP, Opcode.JMP_IF, Opcode.JMP_IF_NOT, Opcode.CALL}

# Opcodes that take an integer slot index as operand
SLOT_OPCODES = {Opcode.LOAD, Opcode.STORE}

# Opcodes that take a typed literal as operand
LITERAL_OPCODES = {Opcode.PUSH_INT, Opcode.PUSH_FLOAT, Opcode.PUSH_STR}
