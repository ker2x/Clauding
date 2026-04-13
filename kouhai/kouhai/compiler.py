from typing import Optional
"""Compiler orchestrator: source → tokens → AST → type-check → LLVM IR → binary."""

import subprocess
import tempfile
from pathlib import Path

from .tokens import lex, LexError
from .parser import Parser, ParseError
from .types import check_program, TypeError_
from .codegen import CodeGen
from .kouhai_interpreter import interpret_program


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


def _resolve_imports(program, source_dir: Path, loaded: Optional[dict] = None):
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


def _collect_links(program, collected: Optional[set] = None) -> set[str]:
    """Collect all link directives from a program and its imported modules."""
    if collected is None:
        collected = set()
    for link in program.links:
        collected.add(link.lib_name)
    for mod_prog in program.module_programs.values():
        _collect_links(mod_prog, collected)
    return collected


def compile_source(
    source: str,
    output_path: Optional[str] = None,
    emit_ir: bool = False,
    source_dir: Optional[str] = None,
    extra_links: Optional[list[str]] = None,
) -> str:
    """Compile Kouhai source to a native binary.

    Args:
        source: Kouhai source code string
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

    # Write IR to temp file and compile with clang + runtime.ll
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(ir)
        ll_path = f.name
    runtime_ll = Path(__file__).resolve().parent.parent / "runtime" / "runtime.ll"

    if output_path is None:
        output_path = tempfile.mktemp(suffix="")

    # Collect link directives from source and imported modules
    link_libs = _collect_links(program)
    if extra_links:
        link_libs.update(extra_links)
    link_flags = [f"-l{lib}" for lib in sorted(link_libs)]

    try:
        result = subprocess.run(
            [
                "clang",
                "-x",
                "ir",
                "-o",
                output_path,
                ll_path,
                str(runtime_ll),
                "-O2",
                "-Wno-override-module",
            ]
            + link_flags,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CompileError(f"clang error:\n{result.stderr}")
    finally:
        Path(ll_path).unlink(missing_ok=True)

    return output_path


def compile_file(
    path: str,
    output_path: Optional[str] = None,
    emit_ir: bool = False,
    extra_links: Optional[list[str]] = None,
) -> str:
    """Compile a .kou file."""
    p = Path(path)
    source = p.read_text()
    return compile_source(
        source,
        output_path=output_path,
        emit_ir=emit_ir,
        source_dir=str(p.parent),
        extra_links=extra_links,
    )


def run_file(path: str, extra_links: Optional[list[str]] = None) -> int:
    """Compile and run a .kou file. Returns the exit code."""
    binary = compile_file(path, extra_links=extra_links)
    try:
        result = subprocess.run([binary], capture_output=False)
        return result.returncode
    finally:
        Path(binary).unlink(missing_ok=True)


def run_interpret(path: str) -> None:
    """Parse and interpret a .kou file using the Kouhai interpreter."""
    p = Path(path)
    source = p.read_text()
    program = _parse_source(source)
    _resolve_imports(program, p.parent)
    interpret_program(program)
