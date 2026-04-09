"""Lexer for Senpai: tokenizes source into a stream including INDENT/DEDENT."""

from enum import Enum, auto
from typing import NamedTuple


class TT(Enum):
    """Token types."""
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STR_LIT = auto()
    BOOL_LIT = auto()

    # Identifier
    IDENT = auto()

    # Keywords
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

    # Operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    PERCENT = auto()    # %
    EQ = auto()         # =
    EQEQ = auto()      # ==
    NEQ = auto()        # !=
    LT = auto()         # <
    GT = auto()         # >
    LE = auto()         # <=
    GE = auto()         # >=
    ARROW = auto()      # ->

    # Delimiters
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    COLON = auto()      # :
    COMMA = auto()      # ,
    DOT = auto()        # .

    # Indentation
    INDENT = auto()
    DEDENT = auto()
    NEWLINE = auto()

    # End of file
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
    "true": TT.BOOL_LIT,
    "false": TT.BOOL_LIT,
}


class LexError(Exception):
    def __init__(self, line: int, col: int, msg: str):
        super().__init__(f"line {line}, col {col}: {msg}")
        self.line = line
        self.col = col


def lex(source: str) -> list[Token]:
    """Tokenize source code into a list of tokens with INDENT/DEDENT."""
    tokens: list[Token] = []
    lines = source.split("\n")
    indent_stack = [0]  # stack of indentation levels

    for lineno, line in enumerate(lines, 1):
        # Skip empty lines and comment-only lines
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue

        # Calculate indentation
        indent = len(line) - len(stripped)

        # Emit INDENT/DEDENT tokens
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token(TT.INDENT, None, lineno, 0))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TT.DEDENT, None, lineno, 0))
            if indent != indent_stack[-1]:
                raise LexError(lineno, indent, "inconsistent indentation")

        # Tokenize the line content
        i = indent
        while i < len(line):
            ch = line[i]

            # Whitespace
            if ch in " \t":
                i += 1
                continue

            # Comment
            if ch == "#":
                break

            col = i + 1

            # String literal
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

            # Number (int or float)
            if ch.isdigit():
                j = i
                while j < len(line) and line[j].isdigit():
                    j += 1
                if j < len(line) and line[j] == "." and (j + 1 < len(line) and line[j + 1].isdigit()):
                    j += 1
                    while j < len(line) and line[j].isdigit():
                        j += 1
                    # Check for 'f' suffix (32-bit float)
                    if j < len(line) and line[j] == "f":
                        tokens.append(Token(TT.FLOAT_LIT, (float(line[i:j]), True), lineno, col))
                        j += 1
                    else:
                        tokens.append(Token(TT.FLOAT_LIT, (float(line[i:j]), False), lineno, col))
                else:
                    tokens.append(Token(TT.INT_LIT, int(line[i:j]), lineno, col))
                i = j
                continue

            # Identifier or keyword
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

            # Two-character operators
            two = line[i:i + 2] if i + 1 < len(line) else ""
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

            # Single-character operators/delimiters
            SINGLE = {
                "+": TT.PLUS, "-": TT.MINUS, "*": TT.STAR, "/": TT.SLASH,
                "%": TT.PERCENT, "=": TT.EQ, "<": TT.LT, ">": TT.GT,
                "(": TT.LPAREN, ")": TT.RPAREN, ":": TT.COLON,
                ",": TT.COMMA, ".": TT.DOT,
                "[": TT.LBRACKET, "]": TT.RBRACKET,
            }
            if ch in SINGLE:
                tokens.append(Token(SINGLE[ch], ch, lineno, col))
                i += 1
                continue

            raise LexError(lineno, col, f"unexpected character: {ch!r}")

        tokens.append(Token(TT.NEWLINE, None, lineno, 0))

    # Close remaining indentation levels
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TT.DEDENT, None, lineno if lines else 0, 0))

    tokens.append(Token(TT.EOF, None, 0, 0))
    return tokens
