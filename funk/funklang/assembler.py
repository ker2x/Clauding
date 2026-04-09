"""Two-pass assembler: text source → list of Instructions."""

from .opcodes import Instruction, Opcode, LABEL_OPCODES, SLOT_OPCODES
from .vm import FunkError


# Map mnemonic (upper-case) → Opcode
_MNEMONIC_MAP = {op.name: op for op in Opcode}


def assemble(source: str) -> list[Instruction]:
    """Assemble source text into a list of Instructions."""
    raw_lines = source.splitlines()

    # Pre-process: strip comments, blank lines, identify labels vs instructions
    entries: list[tuple[int, str]] = []  # (original line number, content)
    for lineno, raw in enumerate(raw_lines, start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        entries.append((lineno, line))

    # Pass 1: collect labels → instruction index
    labels: dict[str, int] = {}
    instruction_entries: list[tuple[int, str]] = []  # entries that are instructions
    for lineno, line in entries:
        if line.endswith(":"):
            label = line[:-1].strip()
            if not label:
                raise FunkError(f"line {lineno}: empty label")
            if label in labels:
                raise FunkError(f"line {lineno}: duplicate label '{label}'")
            labels[label] = len(instruction_entries)
        else:
            instruction_entries.append((lineno, line))

    # Pass 2: parse instructions, resolve labels
    program: list[Instruction] = []
    for lineno, line in instruction_entries:
        instr = _parse_instruction(lineno, line, labels)
        program.append(instr)

    return program


def assemble_file(path: str) -> list[Instruction]:
    """Read a .funk file and assemble it."""
    with open(path) as f:
        return assemble(f.read())


def _strip_comment(line: str) -> str:
    """Remove ; comment from a line, respecting string literals."""
    in_string = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_string = not in_string
        elif ch == ';' and not in_string:
            return line[:i]
    return line


def _parse_instruction(lineno: int, line: str, labels: dict[str, int]) -> Instruction:
    """Parse a single instruction line into an Instruction."""
    # Split mnemonic from operand
    # Special handling for PUSH_STR: operand is the quoted string
    parts = line.split(None, 1)
    mnemonic = parts[0].upper()
    operand_str = parts[1].strip() if len(parts) > 1 else None

    if mnemonic not in _MNEMONIC_MAP:
        raise FunkError(f"line {lineno}: unknown instruction '{mnemonic}'")

    opcode = _MNEMONIC_MAP[mnemonic]

    # Parse operand based on opcode type
    if opcode == Opcode.PUSH_INT:
        if operand_str is None:
            raise FunkError(f"line {lineno}: PUSH_INT requires an operand")
        return Instruction(opcode, int(operand_str))

    elif opcode == Opcode.PUSH_FLOAT:
        if operand_str is None:
            raise FunkError(f"line {lineno}: PUSH_FLOAT requires an operand")
        return Instruction(opcode, float(operand_str))

    elif opcode == Opcode.PUSH_STR:
        if operand_str is None:
            raise FunkError(f"line {lineno}: PUSH_STR requires an operand")
        # Extract content between first and last double quotes
        first = operand_str.find('"')
        last = operand_str.rfind('"')
        if first == -1 or first == last:
            raise FunkError(f"line {lineno}: PUSH_STR operand must be a quoted string")
        return Instruction(opcode, operand_str[first + 1:last])

    elif opcode in LABEL_OPCODES:
        if operand_str is None:
            raise FunkError(f"line {lineno}: {mnemonic} requires a label operand")
        label = operand_str.strip()
        if label not in labels:
            raise FunkError(f"line {lineno}: undefined label '{label}'")
        return Instruction(opcode, labels[label])

    elif opcode in SLOT_OPCODES:
        if operand_str is None:
            raise FunkError(f"line {lineno}: {mnemonic} requires a slot index operand")
        return Instruction(opcode, int(operand_str))

    else:
        # No operand expected
        if operand_str is not None:
            raise FunkError(f"line {lineno}: {mnemonic} takes no operand")
        return Instruction(opcode)
