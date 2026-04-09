"""AST node definitions for Senpai."""

from dataclasses import dataclass, field


# --- Expressions ---

@dataclass
class Expr:
    line: int = 0

@dataclass
class IntLit(Expr):
    value: int = 0

@dataclass
class FloatLit(Expr):
    value: float = 0.0
    is_float32: bool = False  # True when literal has 'f' suffix

@dataclass
class StrLit(Expr):
    value: str = ""

@dataclass
class BoolLit(Expr):
    value: bool = False

@dataclass
class NoneLit(Expr):
    pass

@dataclass
class Var(Expr):
    name: str = ""

@dataclass
class BinOp(Expr):
    op: str = ""
    left: Expr = field(default_factory=Expr)
    right: Expr = field(default_factory=Expr)

@dataclass
class UnaryOp(Expr):
    op: str = ""          # "-" or "not"
    operand: Expr = field(default_factory=Expr)

@dataclass
class Call(Expr):
    func: str = ""
    args: list[Expr] = field(default_factory=list)

@dataclass
class MethodCall(Expr):
    obj: Expr = field(default_factory=Expr)
    method: str = ""
    args: list[Expr] = field(default_factory=list)

@dataclass
class CastExpr(Expr):
    expr: Expr = field(default_factory=Expr)
    target_type: str = ""

@dataclass
class FieldAccess(Expr):
    obj: Expr = field(default_factory=Expr)
    field_name: str = ""


# --- Statements ---

@dataclass
class Stmt:
    line: int = 0

@dataclass
class LetStmt(Stmt):
    name: str = ""
    type_name: str | None = None   # explicit type annotation, or None for literal inference
    value: Expr = field(default_factory=Expr)

@dataclass
class AssignStmt(Stmt):
    target: Expr = field(default_factory=Expr)   # Var or FieldAccess
    value: Expr = field(default_factory=Expr)

@dataclass
class ReturnStmt(Stmt):
    value: Expr | None = None

@dataclass
class IfStmt(Stmt):
    condition: Expr = field(default_factory=Expr)
    body: list[Stmt] = field(default_factory=list)
    elif_clauses: list[tuple[Expr, list[Stmt]]] = field(default_factory=list)
    else_body: list[Stmt] = field(default_factory=list)

@dataclass
class WhileStmt(Stmt):
    condition: Expr = field(default_factory=Expr)
    body: list[Stmt] = field(default_factory=list)

@dataclass
class ForStmt(Stmt):
    var_name: str = ""
    start: Expr | None = None       # None means range(end) → start=0
    end: Expr = field(default_factory=Expr)
    body: list["Stmt"] = field(default_factory=list)

@dataclass
class ExprStmt(Stmt):
    expr: Expr = field(default_factory=Expr)


# --- Declarations ---

@dataclass
class Param:
    name: str
    type_name: str

@dataclass
class FnDecl:
    name: str = ""
    params: list[Param] = field(default_factory=list)
    ret_type: str = "Void"
    body: list[Stmt] = field(default_factory=list)
    line: int = 0

@dataclass
class ClassDecl:
    name: str = ""
    parent: str = "Object"
    methods: list[FnDecl] = field(default_factory=list)
    line: int = 0

@dataclass
class LinkDecl:
    lib_name: str = ""
    line: int = 0

@dataclass
class ExternFnDecl:
    name: str = ""
    params: list[Param] = field(default_factory=list)
    ret_type: str = "Void"
    line: int = 0

@dataclass
class ImportDecl:
    path: str = ""          # e.g. "math.sen"
    module_name: str = ""   # e.g. "math"
    line: int = 0

@dataclass
class Program:
    functions: list[FnDecl] = field(default_factory=list)
    classes: list[ClassDecl] = field(default_factory=list)
    imports: list[ImportDecl] = field(default_factory=list)
    extern_fns: list[ExternFnDecl] = field(default_factory=list)
    links: list[LinkDecl] = field(default_factory=list)
    class_info: dict = field(default_factory=dict)  # filled by type checker
