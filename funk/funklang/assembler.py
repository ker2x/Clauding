"""Two-pass assembler: text source → list of Instructions."""

import os

from .opcodes import Instruction, Opcode, LABEL_OPCODES, NAME_OPCODES
from .vm import FunkError


# Map mnemonic (upper-case) → Opcode
_MNEMONIC_MAP = {op.name: op for op in Opcode}

# Standard library search path (relative to this package)
_STDLIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stdlib")


def _namespace_source(source: str, namespace: str) -> str:
    """Prefix labels in source with namespace. to avoid collisions.

    Labels starting with . are global (the dot is stripped, no prefix).
    All other labels get prefixed: label → namespace.label
    """
    import re
    lines = source.splitlines()
    # Pass 1: collect labels, split into global vs local
    local_labels = set()
    global_labels = set()
    for line in lines:
        stripped = _strip_comment(line).strip()
        if stripped.endswith(":"):
            label = stripped[:-1].strip()
            if label.startswith("."):
                global_labels.add(label)
            else:
                local_labels.add(label)
    if not local_labels and not global_labels:
        return source
    # Pass 2: rewrite labels
    result = []
    for line in lines:
        stripped = _strip_comment(line).strip()
        if stripped.endswith(":"):
            label = stripped[:-1].strip()
            if label.startswith("."):
                result.append(f"{label[1:]}:")  # strip dot, no prefix
            else:
                result.append(f"{namespace}.{label}:")
        else:
            new_line = line
            # Replace local label references (longest first to avoid partial matches)
            for label in sorted(local_labels, key=len, reverse=True):
                new_line = re.sub(r'(?<![.\w])' + re.escape(label) + r'(?![.\w])', f"{namespace}.{label}", new_line)
            # Replace global label references (strip the dot)
            for label in sorted(global_labels, key=len, reverse=True):
                new_line = re.sub(re.escape(label) + r'\b', label[1:], new_line)
            result.append(new_line)
    return "\n".join(result)


def _preprocess(source: str, base_dir: str | None = None, _seen: set[str] | None = None) -> str:
    """Expand INCLUDE directives, returning the fully-inlined source.

    Resolves paths relative to the including file's directory,
    then falls back to the stdlib directory.
    Labels in included files are auto-prefixed with the filename as namespace.
    """
    if _seen is None:
        _seen = set()
    lines = source.splitlines()
    result = []
    for line in lines:
        stripped = _strip_comment(line).strip()
        if stripped.upper().startswith("INCLUDE"):
            parts = stripped.split(None, 1)
            if len(parts) < 2:
                raise FunkError("INCLUDE requires a path")
            path_str = parts[1].strip().strip('"')
            # Resolve: relative to including file first, then stdlib
            resolved = None
            if base_dir:
                candidate = os.path.join(base_dir, path_str)
                if os.path.isfile(candidate):
                    resolved = os.path.abspath(candidate)
            if resolved is None:
                candidate = os.path.join(_STDLIB_DIR, path_str)
                if os.path.isfile(candidate):
                    resolved = os.path.abspath(candidate)
            if resolved is None:
                raise FunkError(f"INCLUDE: file not found '{path_str}'")
            if resolved in _seen:
                continue  # skip already-included files (no error, just idempotent)
            _seen.add(resolved)
            with open(resolved) as f:
                included = f.read()
            # Derive namespace from filename (e.g., "math.funk" → "math")
            namespace = os.path.splitext(os.path.basename(resolved))[0]
            included = _namespace_source(included, namespace)
            inc_dir = os.path.dirname(resolved)
            result.append(_preprocess(included, inc_dir, _seen))
        else:
            result.append(line)
    return "\n".join(result)


def _expand_aliases(lineno: int, line: str) -> list[tuple[int, str]]:
    """Expand assembler aliases into real instructions.

    Returns a list of (lineno, instruction_text) tuples.
    """
    parts = line.split(None, 1)
    mnemonic = parts[0].upper()
    operand = parts[1].strip() if len(parts) > 1 else None

    if mnemonic == "IPUSH":
        return [(lineno, f"PUSH_INT {operand}" if operand else "PUSH_INT")]
    elif mnemonic == "FPUSH":
        return [(lineno, f"PUSH_FLOAT {operand}" if operand else "PUSH_FLOAT")]
    elif mnemonic == "SPUSH":
        return [(lineno, f"PUSH_STR {operand}" if operand else "PUSH_STR")]
    elif mnemonic == "SAY":
        return [(lineno, f"PUSH_STR {operand}" if operand else "PUSH_STR"),
                (lineno, "PRINT")]
    elif mnemonic == "JZ":
        return [(lineno, "PUSH_INT 0"), (lineno, "EQ"), (lineno, f"JMP_IF {operand}")]
    elif mnemonic == "JNZ":
        return [(lineno, "PUSH_INT 0"), (lineno, "EQ"), (lineno, f"JMP_IF_NOT {operand}")]
    elif mnemonic == "JEQ":
        return [(lineno, "EQ"), (lineno, f"JMP_IF {operand}")]
    elif mnemonic == "JNE":
        return [(lineno, "EQ"), (lineno, f"JMP_IF_NOT {operand}")]
    elif mnemonic == "JLT":
        return [(lineno, "LT"), (lineno, f"JMP_IF {operand}")]
    elif mnemonic == "JGT":
        return [(lineno, "GT"), (lineno, f"JMP_IF {operand}")]
    elif mnemonic == "JLE":
        return [(lineno, "GT"), (lineno, f"JMP_IF_NOT {operand}")]
    elif mnemonic == "JGE":
        return [(lineno, "LT"), (lineno, f"JMP_IF_NOT {operand}")]
    else:
        return [(lineno, line)]


def assemble(source: str, base_dir: str | None = None) -> list[Instruction]:
    """Assemble source text into a list of Instructions."""
    source = _preprocess(source, base_dir)
    raw_lines = source.splitlines()

    # Pre-process: strip comments, blank lines, identify labels vs instructions
    entries: list[tuple[int, str]] = []  # (original line number, content)
    for lineno, raw in enumerate(raw_lines, start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        entries.append((lineno, line))

    # Pass 1: collect labels → instruction index; expand aliases
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
            for expanded in _expand_aliases(lineno, line):
                instruction_entries.append(expanded)

    # Pass 2: parse instructions, resolve labels
    program: list[Instruction] = []
    for lineno, line in instruction_entries:
        instr = _parse_instruction(lineno, line, labels)
        program.append(instr)

    return program


def assemble_file(path: str) -> list[Instruction]:
    """Read a .funk file and assemble it."""
    with open(path) as f:
        return assemble(f.read(), base_dir=os.path.dirname(os.path.abspath(path)))


def assemble_debug(source: str, base_dir: str | None = None) -> tuple[list[Instruction], dict[int, int]]:
    """Assemble source and return (program, line_map).

    line_map maps instruction_index → source line number (1-based).
    """
    source = _preprocess(source, base_dir)
    raw_lines = source.splitlines()
    entries = []
    for lineno, raw in enumerate(raw_lines, start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        entries.append((lineno, line))

    instruction_entries = []
    for lineno, line in entries:
        if not line.endswith(":"):
            for expanded in _expand_aliases(lineno, line):
                instruction_entries.append(expanded)

    program = assemble(source)
    line_map = {i: lineno for i, (lineno, _) in enumerate(instruction_entries)}
    return program, line_map


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

    elif opcode in NAME_OPCODES:
        if operand_str is None:
            raise FunkError(f"line {lineno}: {mnemonic} requires a variable name")
        return Instruction(opcode, operand_str.strip())

    else:
        # No operand expected
        if operand_str is not None:
            raise FunkError(f"line {lineno}: {mnemonic} takes no operand")
        return Instruction(opcode)
