"""AST node definitions for Kouhai."""

from dataclasses import dataclass, field


@dataclass
class Expr:
    line: int = 0


@dataclass
class IntLit(Expr):
    value: int = 0


@dataclass
class FloatLit(Expr):
    value: float = 0.0
    is_float32: bool = False


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
class NilLit(Expr):
    pass


@dataclass
class Var(Expr):
    name: str = ""


@dataclass
class BinOp(Expr):
    op: str = ""
    left: "Expr" = field(default_factory=lambda: Expr())
    right: "Expr" = field(default_factory=lambda: Expr())


@dataclass
class UnaryOp(Expr):
    op: str = ""
    operand: "Expr" = field(default_factory=lambda: Expr())


@dataclass
class Call(Expr):
    func: str = ""
    args: "list[Expr]" = field(default_factory=list)


@dataclass
class MethodCall(Expr):
    obj: "Expr" = field(default_factory=lambda: Expr())
    method: str = ""
    args: "list[Expr]" = field(default_factory=list)


@dataclass
class CastExpr(Expr):
    expr: "Expr" = field(default_factory=lambda: Expr())
    target_type: str = ""


@dataclass
class FieldAccess(Expr):
    obj: "Expr" = field(default_factory=lambda: Expr())
    field_name: str = ""


@dataclass
class SizeofExpr(Expr):
    type_name: str = ""


@dataclass
class TernaryExpr(Expr):
    true_expr: "Expr" = field(default_factory=lambda: Expr())
    condition: "Expr" = field(default_factory=lambda: Expr())
    false_expr: "Expr" = field(default_factory=lambda: Expr())


@dataclass
class Stmt:
    line: int = 0


@dataclass
class LetStmt(Stmt):
    name: str = ""
    type_name: "str | None" = None
    value: Expr = field(default_factory=Expr)


@dataclass
class AssignStmt(Stmt):
    target: Expr = field(default_factory=Expr)
    value: Expr = field(default_factory=Expr)


@dataclass
class ReturnStmt(Stmt):
    value: "Expr | None" = None


@dataclass
class IfStmt(Stmt):
    condition: Expr = field(default_factory=Expr)
    body: "list[Stmt]" = field(default_factory=list)
    elif_clauses: "list[tuple[Expr, list[Stmt]]]" = field(default_factory=list)
    else_body: "list[Stmt]" = field(default_factory=list)


@dataclass
class WhileStmt(Stmt):
    condition: Expr = field(default_factory=Expr)
    body: "list[Stmt]" = field(default_factory=list)


@dataclass
class ForStmt(Stmt):
    var_name: str = ""
    start: "Expr | None" = None
    end: Expr = field(default_factory=Expr)
    body: "list[Stmt]" = field(default_factory=list)


@dataclass
class ExprStmt(Stmt):
    expr: Expr = field(default_factory=Expr)


@dataclass
class Param:
    name: str
    type_name: str


@dataclass
class FnDecl:
    name: str = ""
    params: "list[Param]" = field(default_factory=list)
    ret_type: str = "Void"
    body: "list[Stmt]" = field(default_factory=list)
    line: int = 0


@dataclass
class ClassDecl:
    name: str = ""
    parent: str = "Object"
    methods: "list[FnDecl]" = field(default_factory=list)
    line: int = 0


@dataclass
class StructField:
    name: str = ""
    type_name: str = ""
    line: int = 0


@dataclass
class StructDecl:
    name: str = ""
    fields: "list[StructField]" = field(default_factory=list)
    line: int = 0


@dataclass
class LinkDecl:
    lib_name: str = ""
    line: int = 0


@dataclass
class ExternFnDecl:
    name: str = ""
    params: "list[Param]" = field(default_factory=list)
    ret_type: str = "Void"
    line: int = 0


@dataclass
class ImportDecl:
    path: str = ""
    module_name: str = ""
    line: int = 0


@dataclass
class ConstDecl:
    """Top-level constant declaration: let NAME = value"""
    name: str = ""
    value: int = 0  # For now, only support I64 integer constants
    line: int = 0


@dataclass
class Program:
    functions: "list[FnDecl]" = field(default_factory=list)
    classes: "list[ClassDecl]" = field(default_factory=list)
    structs: "list[StructDecl]" = field(default_factory=list)
    imports: "list[ImportDecl]" = field(default_factory=list)
    extern_fns: "list[ExternFnDecl]" = field(default_factory=list)
    consts: "list[ConstDecl]" = field(default_factory=list)
    links: "list[LinkDecl]" = field(default_factory=list)
    class_info: dict = field(default_factory=dict)
    module_programs: dict = field(default_factory=dict)
    module_info: dict = field(default_factory=dict)
