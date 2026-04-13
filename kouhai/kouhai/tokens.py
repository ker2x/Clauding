from typing import Optional
"""Lexer for Kouhai: tokenizes source into a stream including INDENT/DEDENT."""

from enum import Enum, auto
from typing import NamedTuple


class TT(Enum):
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STR_LIT = auto()
    BOOL_LIT = auto()

    IDENT = auto()

    FN = auto()
    CLASS = auto()
    LET = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    RETURN = auto()
    SELF = auto()
    SUPER = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    NONE = auto()
    FOR = auto()
    IN = auto()
    RANGE = auto()
    AS = auto()
    IMPORT = auto()
    EXTERN = auto()
    LINK = auto()
    STRUCT = auto()
    SIZEOF = auto()
    NIL = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    EQEQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    ARROW = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()
    SHL = auto()
    SHR = auto()

    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COLON = auto()
    COMMA = auto()
    DOT = auto()

    INDENT = auto()
    DEDENT = auto()
    NEWLINE = auto()

    EOF = auto()


class Token(NamedTuple):
    type: TT
    value: object
    line: int
    col: int


KEYWORDS = {
    "fn": TT.FN,
    "class": TT.CLASS,
    "let": TT.LET,
    "if": TT.IF,
    "elif": TT.ELIF,
    "else": TT.ELSE,
    "while": TT.WHILE,
    "return": TT.RETURN,
    "self": TT.SELF,
    "super": TT.SUPER,
    "and": TT.AND,
    "or": TT.OR,
    "not": TT.NOT,
    "none": TT.NONE,
    "for": TT.FOR,
    "in": TT.IN,
    "range": TT.RANGE,
    "as": TT.AS,
    "import": TT.IMPORT,
    "extern": TT.EXTERN,
    "link": TT.LINK,
    "struct": TT.STRUCT,
    "sizeof": TT.SIZEOF,
    "nil": TT.NIL,
    "true": TT.BOOL_LIT,
    "false": TT.BOOL_LIT,
}

SINGLE_CHARS = {
    "+": TT.PLUS,
    "-": TT.MINUS,
    "*": TT.STAR,
    "/": TT.SLASH,
    "%": TT.PERCENT,
    "=": TT.EQ,
    "<": TT.LT,
    ">": TT.GT,
    "(": TT.LPAREN,
    ")": TT.RPAREN,
    ":": TT.COLON,
    ",": TT.COMMA,
    ".": TT.DOT,
    "[": TT.LBRACKET,
    "]": TT.RBRACKET,
    "&": TT.AMP,
    "|": TT.PIPE,
    "^": TT.CARET,
    "~": TT.TILDE,
}


class LexError(Exception):
    def __init__(self, line: int, col: int, msg: str):
        super().__init__(f"line {line}, col {col}: {msg}")
        self.line = line
        self.col = col


def lex(source: str) -> list[Token]:
    tokens: list[Token] = []
    lines = source.split("\n")
    indent_stack = [0]
    lineno = 0

    for lineno, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(stripped)

        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token(TT.INDENT, None, lineno, 0))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TT.DEDENT, None, lineno, 0))
            if indent != indent_stack[-1]:
                raise LexError(lineno, indent, "inconsistent indentation")

        i = indent
        while i < len(line):
            ch = line[i]

            if ch in " \t":
                i += 1
                continue

            if ch == "#":
                break

            col = i + 1

            if ch == '"':
                j = i + 1
                s = []
                while j < len(line):
                    if line[j] == "\\":
                        j += 1
                        if j >= len(line):
                            raise LexError(lineno, col, "unterminated string")
                        esc = line[j]
                        if esc == "n":
                            s.append("\n")
                        elif esc == "t":
                            s.append("\t")
                        elif esc == "\\":
                            s.append("\\")
                        elif esc == '"':
                            s.append('"')
                        elif esc == "0":
                            s.append("\0")
                        else:
                            raise LexError(lineno, j + 1, f"unknown escape \\{esc}")
                        j += 1
                    elif line[j] == '"':
                        break
                    else:
                        s.append(line[j])
                        j += 1
                if j >= len(line) or line[j] != '"':
                    raise LexError(lineno, col, "unterminated string")
                tokens.append(Token(TT.STR_LIT, "".join(s), lineno, col))
                i = j + 1
                continue

            if ch.isdigit():
                j = i
                while j < len(line) and line[j].isdigit():
                    j += 1
                if (
                    j < len(line)
                    and line[j] == "."
                    and (j + 1 < len(line) and line[j + 1].isdigit())
                ):
                    j += 1
                    while j < len(line) and line[j].isdigit():
                        j += 1
                    if j < len(line) and line[j] == "f":
                        tokens.append(
                            Token(TT.FLOAT_LIT, (float(line[i:j]), True), lineno, col)
                        )
                        j += 1
                    else:
                        tokens.append(
                            Token(TT.FLOAT_LIT, (float(line[i:j]), False), lineno, col)
                        )
                else:
                    tokens.append(Token(TT.INT_LIT, int(line[i:j]), lineno, col))
                i = j
                continue

            if ch.isalpha() or ch == "_":
                j = i
                while j < len(line) and (line[j].isalnum() or line[j] == "_"):
                    j += 1
                word = line[i:j]
                tt = KEYWORDS.get(word)
                if tt == TT.BOOL_LIT:
                    tokens.append(Token(TT.BOOL_LIT, word == "true", lineno, col))
                elif tt is not None:
                    tokens.append(Token(tt, word, lineno, col))
                else:
                    tokens.append(Token(TT.IDENT, word, lineno, col))
                i = j
                continue

            two = line[i : i + 2] if i + 1 < len(line) else ""
            if two == "==":
                tokens.append(Token(TT.EQEQ, "==", lineno, col))
                i += 2
                continue
            if two == "!=":
                tokens.append(Token(TT.NEQ, "!=", lineno, col))
                i += 2
                continue
            if two == "<=":
                tokens.append(Token(TT.LE, "<=", lineno, col))
                i += 2
                continue
            if two == ">=":
                tokens.append(Token(TT.GE, ">=", lineno, col))
                i += 2
                continue
            if two == "->":
                tokens.append(Token(TT.ARROW, "->", lineno, col))
                i += 2
                continue
            if two == "<<":
                tokens.append(Token(TT.SHL, "<<", lineno, col))
                i += 2
                continue
            if two == ">>":
                tokens.append(Token(TT.SHR, ">>", lineno, col))
                i += 2
                continue

            if ch in SINGLE_CHARS:
                tokens.append(Token(SINGLE_CHARS[ch], ch, lineno, col))
                i += 1
                continue

            raise LexError(lineno, col, f"unexpected character: {ch!r}")

        tokens.append(Token(TT.NEWLINE, None, lineno, 0))

    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TT.DEDENT, None, lineno, 0))

    tokens.append(Token(TT.EOF, None, 0, 0))
    return tokens
