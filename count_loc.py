"""
LOC and Token Counter for codebase exploration.

Usage:
    python count_loc.py --total
    python count_loc.py --projects
    python count_loc.py --languages
    python count_loc.py --files --min-loc 100
    python count_loc.py --project 001 --details
    python count_loc.py --ext .py --ext .md --total
"""

import argparse
import os
import re
import sys
import tokenize
from io import StringIO
from pathlib import Path
from typing import Optional

LANGUAGE_MAP = {
    ".py": "Python",
    ".c": "C",
    ".h": "C Header",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c++": "C++",
    ".mm": "Objective-C++",
    ".swift": "Swift",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".md": "Markdown",
    ".txt": "Text",
    ".kou": "Kouhai",
    ".sen": "Senpai",
    ".funk": "Funk",
    ".rs": "Rust",
    ".jl": "Julia",
    ".java": "Java",
    ".go": "Go",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".rb": "Ruby",
    ".php": "PHP",
    ".lua": "Lua",
    ".R": "R",
    ".r": "R",
    ".scala": "Scala",
    ".hs": "Haskell",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".clj": "Clojure",
    ".ml": "OCaml",
    ".fs": "F#",
    ".ex": "Elixir",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
    ".xml": "XML",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
    ".sql": "SQL",
    ".graphql": "GraphQL",
    ".proto": "Protobuf",
}

COMMENT_PATTERNS = {
    ".py": None,
    ".c": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".cpp": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".cc": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".cxx": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".c++": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".mm": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".h": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".swift": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".sh": re.compile(r"#.*"),
    ".bash": re.compile(r"#.*"),
    ".zsh": re.compile(r"#.*"),
    ".md": re.compile(r"(?<!\\)#.*"),
    ".txt": None,
    ".kou": re.compile(r"(?<!\\)#.*"),
    ".sen": re.compile(r"(?<!\\)#.*"),
    ".funk": re.compile(r"(?<!\\)#.*"),
    ".rs": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".jl": re.compile(r"(?<!\\)#.*"),
    ".java": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".go": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".js": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".ts": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".tsx": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".jsx": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".rb": re.compile(r"(?<!\\)#.*"),
    ".php": re.compile(r"(?<!\\)/\*.*?\*/|//.*|#.*"),
    ".lua": re.compile(r"(?<!\\)--.*"),
    ".R": re.compile(r"(?<!\\)#.*"),
    ".r": re.compile(r"(?<!\\)#.*"),
    ".scala": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".hs": re.compile(r"(?<!\\)--.*"),
    ".ex": re.compile(r"(?<!\\)#.*"),
    ".exs": re.compile(r"(?<!\\)#.*"),
    ".erl": re.compile(r"(?<!\\)%.*"),
    ".clj": re.compile(r"(?<!\\);.*"),
    ".ml": re.compile(r"\(\*.*?\*\)"),
    ".fs": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".yaml": re.compile(r"(?<!\\)#.*"),
    ".yml": re.compile(r"(?<!\\)#.*"),
    ".toml": re.compile(r"(?<!\\)#.*"),
    ".json": None,
    ".xml": re.compile(r"(?<!\\)--.*"),
    ".html": re.compile(r"(?<!\\)--.*"),
    ".css": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".scss": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".sass": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".less": re.compile(r"(?<!\\)/\*.*?\*/|//.*"),
    ".sql": re.compile(r"(?<!\\)--.*"),
    ".graphql": re.compile(r"(?<!\\)#.*"),
    ".proto": re.compile(r"(?<!\\)//.*"),
}

DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "checkpoints",
    "checkpoints2",
    "data",
    "data-lfm",
    "models",
    "logs",
    "logs2",
    ".pytest_cache",
    ".mypy_cache",
    ".eggs",
    "dist",
    "build",
    ".tox",
    ".nox",
    ".dvc",
    ".ruff_cache",
    ".vscode",
    ".idea",
    "site-packages",
    "Caches",
    ".DS_Store",
}

BINARY_EXTS = {
    ".zip",
    ".pt",
    ".safetensors",
    ".pth",
    ".pkl",
    ".pickle",
    ".pb",
    ".onnx",
    ".h5",
    ".bin",
    ".dat",
    ".npy",
    ".npz",
    ".msgpack",
    ".parquet",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".pdf",
    ".mp4",
    ".mp3",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".pt",
}

BLOCK_COMMENT_END = {
    ".py": '"""',
    ".py": "'''",
}


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    return LANGUAGE_MAP.get(ext, ext.upper() if ext else "Unknown")


def count_tokens_python(content: str) -> int:
    try:
        tokens = list(tokenize.generate_tokens(StringIO(content).readline))
        return len(tokens)
    except tokenize.TokenError:
        return 0


def count_tokens_tiktoken(content: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(content))
    except Exception:
        return -1


def strip_comments(content: str, path: Path) -> str:
    ext = path.suffix.lower()
    pattern = COMMENT_PATTERNS.get(ext)

    if pattern is None:
        return content

    try:
        if ext == ".py":
            result = []
            lines = content.split("\n")
            in_block = False
            for line in lines:
                if in_block:
                    if '"""' in line or "'''" in line:
                        in_block = False
                        idx = line.find('"""' if '"""' in line else "'''")
                        line = line[idx + 3:]
                        if line.strip():
                            result.append(line)
                    else:
                        continue
                else:
                    triple_single = line.find("'''")
                    triple_double = line.find('"""')
                    if triple_double != -1 and (triple_single == -1 or triple_double < triple_single):
                        before = line[:triple_double]
                        after = line[triple_double + 3:]
                        if before.strip():
                            result.append(before)
                        if '"""' in after:
                            rest = after[after.find('"""') + 3:]
                            if rest.strip():
                                result.append(rest)
                        else:
                            in_block = True
                            if after.strip():
                                result.append(after)
                    elif triple_single != -1:
                        before = line[:triple_single]
                        after = line[triple_single + 3:]
                        if before.strip():
                            result.append(before)
                        if "'''" in after:
                            rest = after[after.find("'''") + 3:]
                            if rest.strip():
                                result.append(rest)
                        else:
                            in_block = True
                            if after.strip():
                                result.append(after)
                    else:
                        hash_idx = line.find("#")
                        if hash_idx != -1:
                            before = line[:hash_idx]
                            if before.strip():
                                result.append(before)
                        else:
                            result.append(line)
            return "\n".join(result)
        else:
            return pattern.sub("", content)
    except Exception:
        return content


def count_loc(content: str, path: Path) -> int:
    stripped = strip_comments(content, path)
    return sum(1 for line in stripped.split("\n") if line.strip())


def count_raw_chars(content: str) -> int:
    return len(content)


def count_tokens(content: str, path: Path) -> int:
    ext = path.suffix.lower()

    if ext == ".py":
        tok_count = count_tokens_python(content)
        if tok_count > 0:
            return tok_count

    tiktoken_count = count_tokens_tiktoken(content)
    if tiktoken_count > 0:
        return tiktoken_count

    CHAR_PER_TOKEN = 4.0
    non_ws = len("".join(content.split()))
    return int(non_ws / CHAR_PER_TOKEN)


def get_projects(root: Path) -> dict[str, Path]:
    projects = {}
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in DEFAULT_EXCLUDES:
            projects[entry.name] = entry
    return projects


def walk_files(
    root: Path,
    extensions: Optional[set[str]] = None,
    exclude_dirs: Optional[set[str]] = None,
    min_loc: int = 0,
    project_name: Optional[str] = None,
) -> list[tuple[Path, str, int, int, int]]:
    if exclude_dirs is None:
        exclude_dirs = set()
    exclude = DEFAULT_EXCLUDES | exclude_dirs

    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]

        if project_name:
            if dirpath.name != project_name and dirpath.parent != root:
                continue

        for filename in filenames:
            filepath = dirpath / filename
            ext = filepath.suffix.lower()

            if ext in BINARY_EXTS:
                continue

            if extensions and ext not in extensions:
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            loc = count_loc(content, filepath)
            if loc < min_loc:
                continue

            raw_chars = count_raw_chars(content)
            tokens = count_tokens(content, filepath)
            lang = detect_language(filepath)

            rel_path = filepath.relative_to(root)
            results.append((rel_path, lang, loc, raw_chars, tokens))

    return results


def fmt_num(n: int) -> str:
    return f"{n:,}"


def print_total(files: list[tuple[Path, str, int, int, int]], show_tokens: bool, show_raw_chars: bool):
    total_loc = sum(f[2] for f in files)
    total_chars = sum(f[3] for f in files)
    total_tokens = sum(f[4] for f in files)

    print(f"\n{'='*60}")
    print(f" TOTAL")
    print(f"{'='*60}")
    print(f"  Files:      {fmt_num(len(files))}")
    print(f"  LOC:        {fmt_num(total_loc)}")
    if show_raw_chars:
        print(f"  Raw chars:  {fmt_num(total_chars)}")
    if show_tokens:
        print(f"  Tokens:     {fmt_num(total_tokens)}")


def print_projects(
    files: list[tuple[Path, str, int, int, int]],
    show_tokens: bool,
    show_raw_chars: bool,
):
    by_project: dict[str, dict] = {}

    for path, lang, loc, raw_chars, tokens in files:
        project = path.parts[0] if len(path.parts) > 1 else "root"
        if project not in by_project:
            by_project[project] = {"files": 0, "loc": 0, "chars": 0, "tokens": 0}
        by_project[project]["files"] += 1
        by_project[project]["loc"] += loc
        by_project[project]["chars"] += raw_chars
        by_project[project]["tokens"] += tokens

    print(f"\n{'='*60}")
    print(f" PER-PROJECT STATS")
    print(f"{'='*60}")
    if show_tokens:
        print(f" {'Project':<20} {'Files':>8} {'LOC':>10} {'Tokens':>12}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10} {'-'*12:>12}")
        for project in sorted(by_project.keys()):
            stats = by_project[project]
            print(f" {project:<20} {stats['files']:>8,} {stats['loc']:>10,} {stats['tokens']:>12,}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10} {'-'*12:>12}")
        total_loc = sum(s["loc"] for s in by_project.values())
        total_files = sum(s["files"] for s in by_project.values())
        total_tokens = sum(s["tokens"] for s in by_project.values())
        print(f" {'TOTAL':<20} {total_files:>8,} {total_loc:>10,} {total_tokens:>12,}")
    else:
        print(f" {'Project':<20} {'Files':>8} {'LOC':>10}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10}")
        for project in sorted(by_project.keys()):
            stats = by_project[project]
            print(f" {project:<20} {stats['files']:>8,} {stats['loc']:>10,}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10}")
        total_loc = sum(s["loc"] for s in by_project.values())
        total_files = sum(s["files"] for s in by_project.values())
        print(f" {'TOTAL':<20} {total_files:>8,} {total_loc:>10,}")


def print_languages(
    files: list[tuple[Path, str, int, int, int]],
    show_tokens: bool,
    show_raw_chars: bool,
):
    by_lang: dict[str, dict] = {}

    for path, lang, loc, raw_chars, tokens in files:
        if lang not in by_lang:
            by_lang[lang] = {"files": 0, "loc": 0, "chars": 0, "tokens": 0}
        by_lang[lang]["files"] += 1
        by_lang[lang]["loc"] += loc
        by_lang[lang]["chars"] += raw_chars
        by_lang[lang]["tokens"] += tokens

    print(f"\n{'='*60}")
    print(f" PER-LANGUAGE STATS")
    print(f"{'='*60}")
    if show_tokens:
        print(f" {'Language':<20} {'Files':>8} {'LOC':>10} {'Tokens':>12}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10} {'-'*12:>12}")
        for lang in sorted(by_lang.keys(), key=lambda l: by_lang[l]["loc"], reverse=True):
            stats = by_lang[lang]
            print(f" {lang:<20} {stats['files']:>8,} {stats['loc']:>10,} {stats['tokens']:>12,}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10} {'-'*12:>12}")
        total_loc = sum(s["loc"] for s in by_lang.values())
        total_files = sum(s["files"] for s in by_lang.values())
        total_tokens = sum(s["tokens"] for s in by_lang.values())
        print(f" {'TOTAL':<20} {total_files:>8,} {total_loc:>10,} {total_tokens:>12,}")
    else:
        print(f" {'Language':<20} {'Files':>8} {'LOC':>10}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10}")
        for lang in sorted(by_lang.keys(), key=lambda l: by_lang[l]["loc"], reverse=True):
            stats = by_lang[lang]
            print(f" {lang:<20} {stats['files']:>8,} {stats['loc']:>10,}")
        print(f" {'-'*20:<20} {'-'*8:>8} {'-'*10:>10}")
        total_loc = sum(s["loc"] for s in by_lang.values())
        total_files = sum(s["files"] for s in by_lang.values())
        print(f" {'TOTAL':<20} {total_files:>8,} {total_loc:>10,}")


def print_files(
    files: list[tuple[Path, str, int, int, int]],
    show_tokens: bool,
    show_raw_chars: bool,
    details: bool = False,
):
    sorted_files = sorted(files, key=lambda f: f[2], reverse=True)

    if details:
        print(f"\n{'='*60}")
        print(f" PER-FILE DETAILS")
        print(f"{'='*60}")
        if show_tokens:
            print(f" {'File':<45} {'Lang':<12} {'LOC':>8} {'Tokens':>10}")
            print(f" {'-'*45:<45} {'-'*12:>12} {'-'*8:>8} {'-'*10:>10}")
            for path, lang, loc, raw_chars, tokens in sorted_files:
                path_str = str(path)
                if len(path_str) > 45:
                    path_str = "..." + path_str[-(42):]
                print(f" {path_str:<45} {lang:<12} {loc:>8,} {tokens:>10,}")
        else:
            print(f" {'File':<45} {'Lang':<12} {'LOC':>8}")
            print(f" {'-'*45:<45} {'-'*12:>12} {'-'*8:>8}")
            for path, lang, loc, raw_chars, tokens in sorted_files:
                path_str = str(path)
                if len(path_str) > 45:
                    path_str = "..." + path_str[-(42):]
                print(f" {path_str:<45} {lang:<12} {loc:>8,}")
    else:
        top = sorted_files[:20]
        print(f"\n{'='*60}")
        print(f" TOP 20 FILES BY LOC")
        print(f"{'='*60}")
        if show_tokens:
            print(f" {'File':<45} {'Lang':<12} {'LOC':>8} {'Tokens':>10}")
            print(f" {'-'*45:<45} {'-'*12:>12} {'-'*8:>8} {'-'*10:>10}")
            for path, lang, loc, raw_chars, tokens in top:
                path_str = str(path)
                if len(path_str) > 45:
                    path_str = "..." + path_str[-(42):]
                print(f" {path_str:<45} {lang:<12} {loc:>8,} {tokens:>10,}")
        else:
            print(f" {'File':<45} {'Lang':<12} {'LOC':>8}")
            print(f" {'-'*45:<45} {'-'*12:>12} {'-'*8:>8}")
            for path, lang, loc, raw_chars, tokens in top:
                path_str = str(path)
                if len(path_str) > 45:
                    path_str = "..." + path_str[-(42):]
                print(f" {path_str:<45} {lang:<12} {loc:>8,}")


def main():
    parser = argparse.ArgumentParser(description="Count LOC and tokens in codebase")
    parser.add_argument("--path", type=str, default=".", help="Starting directory (default: .)")
    parser.add_argument("--total", action="store_true", help="Show total counts only")
    parser.add_argument("--projects", action="store_true", help="Break down by project")
    parser.add_argument("--languages", action="store_true", help="Break down by language")
    parser.add_argument("--files", action="store_true", help="Show top files by LOC")
    parser.add_argument("--details", action="store_true", help="Full per-file listing")
    parser.add_argument("--ext", type=str, action="append", default=None, help="Filter by extension (e.g. .py)")
    parser.add_argument("--exclude-dir", type=str, action="append", default=None, help="Additional directory to exclude")
    parser.add_argument("--min-loc", type=int, default=0, help="Minimum LOC threshold")
    parser.add_argument("--project", type=str, default=None, help="Target a specific project only")
    parser.add_argument("--tokens", action="store_true", help="Show token counts")
    parser.add_argument("--raw-chars", action="store_true", help="Show raw character counts")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")

    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Error: path '{root}' does not exist", file=sys.stderr)
        sys.exit(1)

    extensions = None
    if args.ext:
        exts = set(e if e.startswith(".") else f".{e}" for e in args.ext)
        extensions = exts

    exclude_dirs = set(args.exclude_dir) if args.exclude_dir else set()

    files = walk_files(
        root,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        min_loc=args.min_loc,
        project_name=args.project,
    )

    show_tokens = args.tokens
    show_raw_chars = args.raw_chars

    if not any([args.total, args.projects, args.languages, args.files, args.details]):
        args.total = True
        args.projects = True
        args.languages = True
        args.files = True

    if args.total:
        print_total(files, show_tokens, show_raw_chars)
    if args.projects:
        print_projects(files, show_tokens, show_raw_chars)
    if args.languages:
        print_languages(files, show_tokens, show_raw_chars)
    if args.files:
        print_files(files, show_tokens, show_raw_chars, args.details)
    if args.details:
        print_files(files, show_tokens, show_raw_chars, details=True)


if __name__ == "__main__":
    main()
