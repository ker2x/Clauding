"""Type system and type checker for Senpai."""

from dataclasses import dataclass, field
from .ast_nodes import (
    Expr, IntLit, FloatLit, StrLit, BoolLit, NoneLit, Var,
    BinOp, UnaryOp, Call, MethodCall, FieldAccess, CastExpr,
    Stmt, LetStmt, AssignStmt, ReturnStmt, IfStmt, WhileStmt, ForStmt, ExprStmt,
    FnDecl, ClassDecl, Program,
)


class TypeError_(Exception):
    def __init__(self, line: int, msg: str):
        super().__init__(f"line {line}: {msg}")
        self.line = line


# Type names recognized by the type checker
PRIMITIVE_TYPES = {"I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64",
                   "Float", "Double", "Bool", "Str", "Void"}
INT_ALIAS = "I64"  # "Int" maps to "I64"

# Integer types and their signedness
SIGNED_INT_TYPES = {"I8", "I16", "I32", "I64"}
UNSIGNED_INT_TYPES = {"U8", "U16", "U32", "U64"}
ALL_INT_TYPES = SIGNED_INT_TYPES | UNSIGNED_INT_TYPES
FLOAT_TYPES = {"Float", "Double"}
NUMERIC_TYPES = ALL_INT_TYPES | FLOAT_TYPES


def resolve_type(name: str) -> str:
    """Resolve type aliases."""
    if name == "Int":
        return INT_ALIAS
    return name


@dataclass
class FnSig:
    param_types: list[str]
    ret_type: str


@dataclass
class ClassInfo:
    name: str
    parent_name: str  # "Object" for root classes, "" for Object itself
    fields: dict[str, str] = field(default_factory=dict)  # all fields, ordered (inherited first)
    methods: dict[str, FnSig] = field(default_factory=dict)  # all methods, sigs WITHOUT self
    vtable_order: list[str] = field(default_factory=list)  # method names in vtable slot order (no __init__)
    vtable_impl: dict[str, str] = field(default_factory=dict)  # method_name -> implementing class


@dataclass
class TypeEnv:
    """Type environment with scoping."""
    variables: dict[str, str] = field(default_factory=dict)
    functions: dict[str, FnSig] = field(default_factory=dict)
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    modules: dict[str, "ModuleInfo"] = field(default_factory=dict)
    parent: "TypeEnv | None" = None
    current_fn_ret: str = "Void"
    current_class: str | None = None

    def lookup_var(self, name: str) -> str | None:
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.lookup_var(name)
        return None

    def lookup_fn(self, name: str) -> FnSig | None:
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_fn(name)
        return None

    def lookup_class(self, name: str) -> ClassInfo | None:
        if name in self.classes:
            return self.classes[name]
        if self.parent:
            return self.parent.lookup_class(name)
        return None

    def lookup_module(self, name: str) -> "ModuleInfo | None":
        if name in self.modules:
            return self.modules[name]
        if self.parent:
            return self.parent.lookup_module(name)
        return None

    def child(self) -> "TypeEnv":
        return TypeEnv(parent=self, current_fn_ret=self.current_fn_ret,
                       current_class=self.current_class)


def _is_array_type(name: str) -> bool:
    """Check if a type name is Array[T]."""
    return name.startswith("Array[") and name.endswith("]")


def _array_elem_type(name: str) -> str:
    """Extract T from Array[T]."""
    return name[6:-1]


def _valid_type(name: str, env: TypeEnv) -> bool:
    """Check if a type name is valid (primitive or registered class or Array[T] or module type)."""
    if name in PRIMITIVE_TYPES:
        return True
    if name.startswith("Module:"):
        return True
    if env.lookup_class(name) is not None:
        return True
    if _is_array_type(name):
        return _valid_type(_array_elem_type(name), env)
    return False


def _literal_compatible(expr, from_type: str, to_type: str) -> bool:
    """Check if a literal expression can adapt to a target type."""
    if isinstance(expr, IntLit) and from_type in ALL_INT_TYPES and to_type in ALL_INT_TYPES:
        return True
    if isinstance(expr, FloatLit) and from_type in FLOAT_TYPES and to_type in FLOAT_TYPES:
        return True
    return False


# Operator → dunder method mapping
BINOP_METHODS = {
    "+": "__add__", "-": "__sub__", "*": "__mul__", "/": "__div__", "%": "__mod__",
    "==": "__eq__", "!=": "__eq__",  # != desugars to not __eq__
    "<": "__lt__", ">": "__gt__", "<=": "__le__", ">=": "__ge__",
}
UNARYOP_METHODS = {
    "-": "__neg__",
}


def _is_subtype(child: str, parent: str, env: TypeEnv) -> bool:
    """Check if child is a subtype of parent (same type or inherits from it)."""
    if child == parent:
        return True
    ci = env.lookup_class(child)
    if ci is None or not ci.parent_name:
        return False
    return _is_subtype(ci.parent_name, parent, env)


def _type_compatible(actual: str, expected: str, expr, env: TypeEnv) -> bool:
    """Check if actual type is compatible with expected type."""
    if actual == expected:
        return True
    if _literal_compatible(expr, actual, expected):
        return True
    if _is_subtype(actual, expected, env):
        return True
    return False


@dataclass
class ModuleInfo:
    """Stores type information about an imported module."""
    name: str
    functions: dict[str, FnSig] = field(default_factory=dict)
    classes: dict[str, ClassInfo] = field(default_factory=dict)


def check_program(prog: Program) -> None:
    """Type-check a program. Raises TypeError_ on errors."""
    env = TypeEnv()

    # Register built-in functions
    env.functions["print"] = FnSig(param_types=["I64"], ret_type="Void")
    env.functions["print_str"] = FnSig(param_types=["Str"], ret_type="Void")
    env.functions["print_bool"] = FnSig(param_types=["Bool"], ret_type="Void")

    # Register built-in root class
    env.classes["Object"] = ClassInfo(name="Object", parent_name="")

    # Process imports
    prog.module_info = {}
    if hasattr(prog, 'module_programs'):
        for mod_name, mod_prog in prog.module_programs.items():
            # Type-check the imported module
            check_program(mod_prog)
            # Collect its exported functions and classes
            mi = ModuleInfo(name=mod_name)
            for fn in mod_prog.functions:
                if fn.name == "main":
                    continue  # skip imported main()
                ret = resolve_type(fn.ret_type)
                params = [resolve_type(p.type_name) for p in fn.params]
                mi.functions[fn.name] = FnSig(param_types=params, ret_type=ret)
            # Collect all class names from this module for type qualification
            mod_class_names = {c for c in mod_prog.class_info if c != "Object"}
            for cls_name, ci in mod_prog.class_info.items():
                if cls_name == "Object":
                    continue
                # Create a copy with qualified types in method signatures
                qual_ci = ClassInfo(
                    name=f"{mod_name}.{cls_name}",
                    parent_name=ci.parent_name,
                    fields=dict(ci.fields),
                    methods={},
                    vtable_order=list(ci.vtable_order),
                    vtable_impl=dict(ci.vtable_impl),
                )
                # Qualify field types that reference module classes
                for fname, ftype in qual_ci.fields.items():
                    if ftype in mod_class_names:
                        qual_ci.fields[fname] = f"{mod_name}.{ftype}"
                # Qualify method signatures
                for mname, sig in ci.methods.items():
                    qual_params = []
                    for pt in sig.param_types:
                        if pt in mod_class_names:
                            qual_params.append(f"{mod_name}.{pt}")
                        else:
                            qual_params.append(pt)
                    qual_ret = f"{mod_name}.{sig.ret_type}" if sig.ret_type in mod_class_names else sig.ret_type
                    qual_ci.methods[mname] = FnSig(param_types=qual_params, ret_type=qual_ret)
                mi.classes[cls_name] = qual_ci
                env.classes[f"{mod_name}.{cls_name}"] = qual_ci
            prog.module_info[mod_name] = mi
            # Register module in env
            env.modules[mod_name] = mi
            env.variables[mod_name] = f"Module:{mod_name}"

    # Register class names first (so they can be used as types)
    for cls in prog.classes:
        env.classes[cls.name] = ClassInfo(name=cls.name, parent_name=cls.parent)

    # Register all user functions
    for fn in prog.functions:
        ret = resolve_type(fn.ret_type)
        params = [resolve_type(p.type_name) for p in fn.params]
        for t in params + [ret]:
            if not _valid_type(t, env):
                raise TypeError_(fn.line, f"unknown type: {t}")
        env.functions[fn.name] = FnSig(param_types=params, ret_type=ret)

    # Discover fields and register method signatures for each class
    for cls in prog.classes:
        _register_class(cls, env)

    # Check each function body
    for fn in prog.functions:
        fn_env = env.child()
        fn_env.current_fn_ret = resolve_type(fn.ret_type)
        for p in fn.params:
            fn_env.variables[p.name] = resolve_type(p.type_name)
        for stmt in fn.body:
            _check_stmt(stmt, fn_env)

    # Check each class's method bodies
    for cls in prog.classes:
        _check_class_body(cls, env)

    # Attach class info to program for codegen
    prog.class_info = dict(env.classes)


def _register_class(cls: ClassDecl, env: TypeEnv) -> None:
    """Register a class's fields and method signatures."""
    ci = env.classes[cls.name]

    # Inherit from parent
    if ci.parent_name and ci.parent_name in env.classes:
        parent = env.classes[ci.parent_name]
        ci.fields = dict(parent.fields)
        ci.methods = dict(parent.methods)
        ci.vtable_order = list(parent.vtable_order)
        ci.vtable_impl = dict(parent.vtable_impl)

    # Register method signatures (without self param)
    for m in cls.methods:
        ret = resolve_type(m.ret_type)
        params = []
        for p in m.params:
            if p.name == "self":
                continue
            pt = resolve_type(p.type_name)
            if not _valid_type(pt, env):
                raise TypeError_(m.line, f"unknown type: {pt}")
        for p in m.params:
            if p.name == "self":
                continue
            params.append(resolve_type(p.type_name))
        if not _valid_type(ret, env) and ret != "Void":
            raise TypeError_(m.line, f"unknown type: {ret}")
        ci.methods[m.name] = FnSig(param_types=params, ret_type=ret)

        # Update vtable (skip __init__)
        if m.name != "__init__":
            if m.name not in ci.vtable_order:
                ci.vtable_order.append(m.name)
            ci.vtable_impl[m.name] = cls.name

    # Discover fields from __init__
    init = next((m for m in cls.methods if m.name == "__init__"), None)
    if init:
        init_env = env.child()
        init_env.variables["self"] = cls.name
        for p in init.params:
            if p.name != "self":
                init_env.variables[p.name] = resolve_type(p.type_name)

        for stmt in init.body:
            if (isinstance(stmt, AssignStmt)
                    and isinstance(stmt.target, FieldAccess)
                    and isinstance(stmt.target.obj, Var)
                    and stmt.target.obj.name == "self"
                    and stmt.target.field_name not in ci.fields):
                val_type = _check_expr(stmt.value, init_env)
                ci.fields[stmt.target.field_name] = val_type


def _check_class_body(cls: ClassDecl, env: TypeEnv) -> None:
    """Type-check all method bodies in a class."""
    for method in cls.methods:
        m_env = env.child()
        m_env.current_class = cls.name
        m_env.current_fn_ret = resolve_type(method.ret_type)
        m_env.variables["self"] = cls.name
        for p in method.params:
            if p.name != "self":
                m_env.variables[p.name] = resolve_type(p.type_name)
        for stmt in method.body:
            _check_stmt(stmt, m_env)


def _check_stmt(stmt: Stmt, env: TypeEnv) -> None:
    if isinstance(stmt, LetStmt):
        val_type = _check_expr(stmt.value, env)
        if stmt.type_name:
            declared = resolve_type(stmt.type_name)
            if not _valid_type(declared, env):
                raise TypeError_(stmt.line, f"unknown type: {declared}")
            if not _type_compatible(val_type, declared, stmt.value, env):
                raise TypeError_(stmt.line,
                    f"type mismatch: declared {declared} but got {val_type}")
            env.variables[stmt.name] = declared
        else:
            env.variables[stmt.name] = val_type

    elif isinstance(stmt, AssignStmt):
        if isinstance(stmt.target, Var):
            existing = env.lookup_var(stmt.target.name)
            if existing is None:
                raise TypeError_(stmt.line, f"undefined variable: {stmt.target.name}")
            val_type = _check_expr(stmt.value, env)
            if val_type != existing:
                raise TypeError_(stmt.line,
                    f"cannot assign {val_type} to {existing} variable '{stmt.target.name}'")
        elif isinstance(stmt.target, FieldAccess):
            obj_type = _check_expr(stmt.target.obj, env)
            ci = env.lookup_class(obj_type)
            if ci is None:
                raise TypeError_(stmt.line, f"cannot set field on type {obj_type}")
            val_type = _check_expr(stmt.value, env)
            if stmt.target.field_name in ci.fields:
                expected = ci.fields[stmt.target.field_name]
                if not _type_compatible(val_type, expected, stmt.value, env):
                    raise TypeError_(stmt.line,
                        f"cannot assign {val_type} to field '{stmt.target.field_name}' of type {expected}")
            else:
                raise TypeError_(stmt.line,
                    f"'{obj_type}' has no field '{stmt.target.field_name}'")
        else:
            _check_expr(stmt.value, env)

    elif isinstance(stmt, ReturnStmt):
        if stmt.value is not None:
            val_type = _check_expr(stmt.value, env)
            if not _type_compatible(val_type, env.current_fn_ret, stmt.value, env):
                raise TypeError_(stmt.line,
                    f"return type mismatch: expected {env.current_fn_ret}, got {val_type}")
        elif env.current_fn_ret != "Void":
            raise TypeError_(stmt.line,
                f"return without value in function returning {env.current_fn_ret}")

    elif isinstance(stmt, IfStmt):
        cond_type = _check_expr(stmt.condition, env)
        if cond_type != "Bool":
            raise TypeError_(stmt.line, f"if condition must be Bool, got {cond_type}")
        block_env = env.child()
        for s in stmt.body:
            _check_stmt(s, block_env)
        for elif_cond, elif_body in stmt.elif_clauses:
            ec_type = _check_expr(elif_cond, env)
            if ec_type != "Bool":
                raise TypeError_(stmt.line, f"elif condition must be Bool, got {ec_type}")
            elif_env = env.child()
            for s in elif_body:
                _check_stmt(s, elif_env)
        if stmt.else_body:
            else_env = env.child()
            for s in stmt.else_body:
                _check_stmt(s, else_env)

    elif isinstance(stmt, WhileStmt):
        cond_type = _check_expr(stmt.condition, env)
        if cond_type != "Bool":
            raise TypeError_(stmt.line, f"while condition must be Bool, got {cond_type}")
        block_env = env.child()
        for s in stmt.body:
            _check_stmt(s, block_env)

    elif isinstance(stmt, ForStmt):
        end_type = _check_expr(stmt.end, env)
        if end_type not in ALL_INT_TYPES:
            raise TypeError_(stmt.line, f"range() expects integer, got {end_type}")
        if stmt.start is not None:
            start_type = _check_expr(stmt.start, env)
            if start_type not in ALL_INT_TYPES:
                raise TypeError_(stmt.line, f"range() expects integer, got {start_type}")
        block_env = env.child()
        block_env.variables[stmt.var_name] = "I64"
        for s in stmt.body:
            _check_stmt(s, block_env)

    elif isinstance(stmt, ExprStmt):
        _check_expr(stmt.expr, env)


def _check_expr(expr: Expr, env: TypeEnv) -> str:
    """Type-check an expression, return its type name."""
    if isinstance(expr, IntLit):
        return "I64"
    if isinstance(expr, FloatLit):
        return "Float" if expr.is_float32 else "Double"
    if isinstance(expr, StrLit):
        return "Str"
    if isinstance(expr, BoolLit):
        return "Bool"
    if isinstance(expr, NoneLit):
        return "Void"

    if isinstance(expr, Var):
        # super resolves to parent class type
        if expr.name == "super" and env.current_class:
            ci = env.lookup_class(env.current_class)
            if ci and ci.parent_name:
                return ci.parent_name
            raise TypeError_(expr.line, "super used in class with no parent")
        t = env.lookup_var(expr.name)
        if t is None:
            raise TypeError_(expr.line, f"undefined variable: {expr.name}")
        return t

    if isinstance(expr, FieldAccess):
        obj_type = _check_expr(expr.obj, env)
        ci = env.lookup_class(obj_type)
        if ci is None:
            raise TypeError_(expr.line, f"cannot access field on type {obj_type}")
        if expr.field_name not in ci.fields:
            raise TypeError_(expr.line, f"'{obj_type}' has no field '{expr.field_name}'")
        return ci.fields[expr.field_name]

    if isinstance(expr, MethodCall):
        obj_type = _check_expr(expr.obj, env)
        # Module-qualified call: module.func() or module.Class()
        if obj_type.startswith("Module:"):
            mod_name = obj_type[7:]  # strip "Module:" prefix
            # Find the module info — walk up the env chain to find root
            root = env
            while root.parent:
                root = root.parent
            # Check module_info through the program (stored during check_program)
            # We stored modules as variables, need to find the ModuleInfo
            # It's accessible via the class registry with "mod.Class" keys
            # and function registry... let's use a helper on env
            mi = env.lookup_module(mod_name)
            if mi is None:
                raise TypeError_(expr.line, f"unknown module: {mod_name}")
            # Check if it's a class constructor
            if expr.method in mi.classes:
                ci = mi.classes[expr.method]
                if "__init__" in ci.methods:
                    init_sig = ci.methods["__init__"]
                    if len(expr.args) != len(init_sig.param_types):
                        raise TypeError_(expr.line,
                            f"{expr.method}() expects {len(init_sig.param_types)} args, got {len(expr.args)}")
                    for i, (arg, expected) in enumerate(zip(expr.args, init_sig.param_types)):
                        at = _check_expr(arg, env)
                        if not _type_compatible(at, expected, arg, env):
                            raise TypeError_(expr.line,
                                f"{expr.method}() arg {i + 1}: expected {expected}, got {at}")
                elif len(expr.args) != 0:
                    raise TypeError_(expr.line, f"{expr.method}() takes no arguments")
                return f"{mod_name}.{expr.method}"
            # Check if it's a function
            if expr.method in mi.functions:
                sig = mi.functions[expr.method]
                if len(expr.args) != len(sig.param_types):
                    raise TypeError_(expr.line,
                        f"{expr.method}() expects {len(sig.param_types)} args, got {len(expr.args)}")
                for i, (arg, expected) in enumerate(zip(expr.args, sig.param_types)):
                    at = _check_expr(arg, env)
                    if not _type_compatible(at, expected, arg, env):
                        raise TypeError_(expr.line,
                            f"{expr.method}() arg {i + 1}: expected {expected}, got {at}")
                return sig.ret_type
            raise TypeError_(expr.line, f"module '{mod_name}' has no '{expr.method}'")

        # Built-in to_str() on primitive types
        if expr.method == "to_str" and len(expr.args) == 0:
            if obj_type in NUMERIC_TYPES or obj_type == "Bool":
                return "Str"
        # Array methods
        if _is_array_type(obj_type):
            elem_type = _array_elem_type(obj_type)
            if expr.method == "push":
                if len(expr.args) != 1:
                    raise TypeError_(expr.line, "push() expects 1 argument")
                at = _check_expr(expr.args[0], env)
                if not _type_compatible(at, elem_type, expr.args[0], env):
                    raise TypeError_(expr.line, f"push() expects {elem_type}, got {at}")
                return "Void"
            if expr.method == "get":
                if len(expr.args) != 1:
                    raise TypeError_(expr.line, "get() expects 1 argument")
                at = _check_expr(expr.args[0], env)
                if at not in ALL_INT_TYPES:
                    raise TypeError_(expr.line, f"get() index must be integer, got {at}")
                return elem_type
            if expr.method == "set":
                if len(expr.args) != 2:
                    raise TypeError_(expr.line, "set() expects 2 arguments")
                idx_t = _check_expr(expr.args[0], env)
                if idx_t not in ALL_INT_TYPES:
                    raise TypeError_(expr.line, f"set() index must be integer, got {idx_t}")
                val_t = _check_expr(expr.args[1], env)
                if not _type_compatible(val_t, elem_type, expr.args[1], env):
                    raise TypeError_(expr.line, f"set() expects {elem_type}, got {val_t}")
                return "Void"
            if expr.method == "len":
                if len(expr.args) != 0:
                    raise TypeError_(expr.line, "len() takes no arguments")
                return "I64"
            raise TypeError_(expr.line, f"Array has no method '{expr.method}'")
        ci = env.lookup_class(obj_type)
        if ci is None:
            raise TypeError_(expr.line, f"cannot call method on type {obj_type}")
        if expr.method not in ci.methods:
            raise TypeError_(expr.line, f"'{obj_type}' has no method '{expr.method}'")
        sig = ci.methods[expr.method]
        if len(expr.args) != len(sig.param_types):
            raise TypeError_(expr.line,
                f"{expr.method}() expects {len(sig.param_types)} args, got {len(expr.args)}")
        for i, (arg, expected) in enumerate(zip(expr.args, sig.param_types)):
            at = _check_expr(arg, env)
            if not _type_compatible(at, expected, arg, env):
                raise TypeError_(expr.line,
                    f"{expr.method}() arg {i + 1}: expected {expected}, got {at}")
        return sig.ret_type

    if isinstance(expr, BinOp):
        lt = _check_expr(expr.left, env)
        rt = _check_expr(expr.right, env)

        if expr.op in ("+", "-", "*", "/", "%"):
            if lt in NUMERIC_TYPES and rt in NUMERIC_TYPES:
                if lt != rt:
                    raise TypeError_(expr.line,
                        f"mismatched types in '{expr.op}': {lt} and {rt}")
                if expr.op == "/" and lt in ALL_INT_TYPES:
                    return lt
                return lt
            if expr.op == "+" and lt == "Str" and rt == "Str":
                return "Str"
            # Fall through to operator method check below

        elif expr.op in ("==", "!="):
            if lt in PRIMITIVE_TYPES and rt in PRIMITIVE_TYPES:
                if lt != rt:
                    raise TypeError_(expr.line,
                        f"cannot compare {lt} and {rt}")
                return "Bool"
            # Fall through to operator method check for class types

        elif expr.op in ("<", ">", "<=", ">="):
            if lt in NUMERIC_TYPES and rt in NUMERIC_TYPES and lt == rt:
                return "Bool"
            # Fall through to operator method check

        elif expr.op in ("and", "or"):
            if lt == "Bool" and rt == "Bool":
                return "Bool"
            raise TypeError_(expr.line,
                f"'{expr.op}' requires Bool operands, got {lt} and {rt}")

        # Operator methods on class types
        method_name = BINOP_METHODS.get(expr.op)
        if method_name:
            actual_method = "__eq__" if expr.op in ("==", "!=") else method_name
            ci = env.lookup_class(lt)
            if ci and actual_method in ci.methods:
                sig = ci.methods[actual_method]
                if len(sig.param_types) == 1:
                    if not _type_compatible(rt, sig.param_types[0], expr.right, env):
                        raise TypeError_(expr.line,
                            f"{actual_method}() expects {sig.param_types[0]}, got {rt}")
                    if expr.op in ("!=", "==") and actual_method == "__eq__":
                        return "Bool"
                    return sig.ret_type

        raise TypeError_(expr.line,
            f"cannot apply '{expr.op}' to {lt} and {rt}")

    if isinstance(expr, UnaryOp):
        ot = _check_expr(expr.operand, env)
        if expr.op == "-":
            if ot in NUMERIC_TYPES:
                return ot
            # Check for __neg__ on class types
            ci = env.lookup_class(ot)
            if ci and "__neg__" in ci.methods:
                return ci.methods["__neg__"].ret_type
            raise TypeError_(expr.line, f"cannot negate {ot}")
        if expr.op == "not":
            if ot == "Bool":
                return "Bool"
            raise TypeError_(expr.line, f"'not' requires Bool, got {ot}")
        raise TypeError_(expr.line, f"unknown unary operator: {expr.op}")

    if isinstance(expr, Call):
        # Array[T]() constructor
        if _is_array_type(expr.func):
            elem_type = _array_elem_type(expr.func)
            if not _valid_type(elem_type, env):
                raise TypeError_(expr.line, f"unknown type: {elem_type}")
            if len(expr.args) != 0:
                raise TypeError_(expr.line, f"Array constructor takes no arguments")
            return expr.func

        # Check for class constructor
        ci = env.lookup_class(expr.func)
        if ci is not None and expr.func != "Object":
            if "__init__" in ci.methods:
                init_sig = ci.methods["__init__"]
                if len(expr.args) != len(init_sig.param_types):
                    raise TypeError_(expr.line,
                        f"{expr.func}() expects {len(init_sig.param_types)} args, got {len(expr.args)}")
                for i, (arg, expected) in enumerate(zip(expr.args, init_sig.param_types)):
                    at = _check_expr(arg, env)
                    if not _type_compatible(at, expected, arg, env):
                        raise TypeError_(expr.line,
                            f"{expr.func}() arg {i + 1}: expected {expected}, got {at}")
            elif len(expr.args) != 0:
                raise TypeError_(expr.line, f"{expr.func}() takes no arguments")
            return expr.func

        # Check for print overloads
        if expr.func == "print" and len(expr.args) == 1:
            arg_type = _check_expr(expr.args[0], env)
            if arg_type in NUMERIC_TYPES or arg_type in ("Bool", "Str"):
                return "Void"
            raise TypeError_(expr.line, f"print() cannot print type {arg_type}")

        sig = env.lookup_fn(expr.func)
        if sig is None:
            raise TypeError_(expr.line, f"undefined function: {expr.func}")
        if len(expr.args) != len(sig.param_types):
            raise TypeError_(expr.line,
                f"{expr.func}() expects {len(sig.param_types)} args, got {len(expr.args)}")
        for i, (arg, expected) in enumerate(zip(expr.args, sig.param_types)):
            at = _check_expr(arg, env)
            if not _type_compatible(at, expected, arg, env):
                raise TypeError_(expr.line,
                    f"{expr.func}() arg {i + 1}: expected {expected}, got {at}")
        return sig.ret_type

    if isinstance(expr, CastExpr):
        src_type = _check_expr(expr.expr, env)
        target = resolve_type(expr.target_type)
        if not _valid_type(target, env):
            raise TypeError_(expr.line, f"unknown type: {target}")
        # Allow numeric-to-numeric casts
        if src_type in NUMERIC_TYPES and target in NUMERIC_TYPES:
            return target
        # Allow Bool to integer
        if src_type == "Bool" and target in ALL_INT_TYPES:
            return target
        raise TypeError_(expr.line, f"cannot cast {src_type} to {target}")

    raise TypeError_(expr.line, f"unsupported expression type: {type(expr).__name__}")
