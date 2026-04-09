"""Compiler orchestrator: source → tokens → AST → type-check → LLVM IR → binary."""

import subprocess
import tempfile
from pathlib import Path

from .tokens import lex, LexError
from .parser import Parser, ParseError
from .types import check_program, TypeError_
from .codegen import CodeGen


class CompileError(Exception):
    pass


def _parse_source(source: str) -> "Program":
    """Lex + parse source into an AST."""
    try:
        tokens = lex(source)
    except LexError as e:
        raise CompileError(f"Lex error: {e}") from e
    try:
        parser = Parser(tokens)
        return parser.parse()
    except ParseError as e:
        raise CompileError(f"Parse error: {e}") from e


def _resolve_imports(program, source_dir: Path, loaded: dict | None = None):
    """Recursively resolve imports, attaching parsed module Programs.

    Populates program.module_programs: dict[str, Program] mapping
    module_name -> parsed+import-resolved Program.
    """
    if loaded is None:
        loaded = {}

    program.module_programs = {}

    for imp in program.imports:
        if imp.module_name in loaded:
            program.module_programs[imp.module_name] = loaded[imp.module_name]
            continue

        # Resolve path relative to the importing file's directory
        mod_path = source_dir / imp.path
        if not mod_path.exists():
            raise CompileError(f"line {imp.line}: cannot find module '{imp.path}'")

        mod_source = mod_path.read_text()
        mod_prog = _parse_source(mod_source)

        # Mark as loaded before recursing (prevents circular imports)
        loaded[imp.module_name] = mod_prog

        # Recursively resolve the module's own imports
        _resolve_imports(mod_prog, mod_path.parent, loaded)

        program.module_programs[imp.module_name] = mod_prog


def compile_source(source: str, output_path: str | None = None,
                   emit_ir: bool = False, source_dir: str | None = None) -> str:
    """Compile Senpai source to a native binary.

    Args:
        source: Senpai source code string
        output_path: path for the output binary (default: temp file)
        emit_ir: if True, return LLVM IR string instead of compiling
        source_dir: directory for resolving imports (default: cwd)

    Returns:
        Path to compiled binary, or LLVM IR string if emit_ir=True
    """
    if source_dir is None:
        source_dir = str(Path.cwd())

    # Parse
    program = _parse_source(source)

    # Resolve imports
    _resolve_imports(program, Path(source_dir))

    # Type check
    try:
        check_program(program)
    except TypeError_ as e:
        raise CompileError(f"Type error: {e}") from e

    # Code gen
    codegen = CodeGen()
    ir = codegen.generate(program)

    if emit_ir:
        return ir

    # Write IR to temp file and compile with clang
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(ir)
        ll_path = f.name

    if output_path is None:
        output_path = tempfile.mktemp(suffix="")

    try:
        result = subprocess.run(
            ["clang", "-o", output_path, ll_path, "-O2", "-Wno-override-module"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise CompileError(f"clang error:\n{result.stderr}")
    finally:
        Path(ll_path).unlink(missing_ok=True)

    return output_path


def compile_file(path: str, output_path: str | None = None,
                 emit_ir: bool = False) -> str:
    """Compile a .sen file."""
    p = Path(path)
    source = p.read_text()
    return compile_source(source, output_path=output_path, emit_ir=emit_ir,
                         source_dir=str(p.parent))


def run_file(path: str) -> int:
    """Compile and run a .sen file. Returns the exit code."""
    binary = compile_file(path)
    try:
        result = subprocess.run([binary], capture_output=False)
        return result.returncode
    finally:
        Path(binary).unlink(missing_ok=True)
