"""LLVM IR text emitter for Senpai."""

from .ast_nodes import (
    Expr, IntLit, FloatLit, StrLit, BoolLit, NoneLit, NilLit, Var,
    BinOp, UnaryOp, Call, MethodCall, FieldAccess, CastExpr,
    SizeofExpr, TernaryExpr,
    Stmt, LetStmt, AssignStmt, ReturnStmt, IfStmt, WhileStmt, ForStmt, ExprStmt,
    FnDecl, ClassDecl, Program,
)
from .types import (resolve_type, ALL_INT_TYPES, SIGNED_INT_TYPES,
                    UNSIGNED_INT_TYPES, FLOAT_TYPES, NUMERIC_TYPES,
                    ClassInfo, PRIMITIVE_TYPES,
                    _is_array_type, _array_elem_type)


# LLVM type mapping
def _llvm_type(senpai_type: str) -> str:
    mapping = {
        "I8": "i8", "I16": "i16", "I32": "i32", "I64": "i64",
        "U8": "i8", "U16": "i16", "U32": "i32", "U64": "i64",
        "Float": "float", "Double": "double",
        "Bool": "i1",
        "Str": "ptr",
        "Ptr": "ptr",
        "Void": "void",
    }
    return mapping.get(senpai_type, "ptr")  # class types are pointers


def _zero_value(senpai_type: str) -> str:
    """Return the LLVM zero value for a Senpai type."""
    if senpai_type in ALL_INT_TYPES:
        return "0"
    if senpai_type == "Float":
        return "0.0"
    if senpai_type == "Double":
        return "0.0"
    if senpai_type == "Bool":
        return "0"
    if senpai_type in ("Str", "Ptr"):
        return "null"
    if senpai_type not in PRIMITIVE_TYPES:
        return "null"  # class types
    return "0"


class CodeGen:
    def __init__(self):
        self._lines: list[str] = []       # main body lines
        self._globals: list[str] = []      # global declarations (strings, etc.)
        self._tmp_counter = 0
        self._label_counter = 0
        self._str_counter = 0
        self._vars: dict[str, tuple[str, str]] = {}  # name -> (alloca_reg, senpai_type)
        self._fn_sigs: dict[str, str] = {} # fn name -> return senpai type
        self._fn_params: dict[str, list[str]] = {} # fn name -> param senpai types
        self._class_info: dict[str, ClassInfo] = {}  # class name -> ClassInfo
        self._current_class: str | None = None  # set when generating a method
        self._modules: dict = {}  # module_name -> module_programs entry
        self._current_module: str | None = None  # set when generating module code
        self._extern_fns: set[str] = set()  # extern function names (use real C name)

    def _tmp(self) -> str:
        self._tmp_counter += 1
        return f"%t{self._tmp_counter}"

    def _label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return f"{prefix}{self._label_counter}"

    def _emit(self, line: str):
        self._lines.append(line)

    def _emit_global(self, line: str):
        self._globals.append(line)

    def _coerce(self, reg: str, from_type: str, to_type: str) -> str:
        """Coerce a value between compatible types (e.g., I64 literal to I8)."""
        if from_type == to_type:
            return reg
        from_llvm = _llvm_type(from_type)
        to_llvm = _llvm_type(to_type)
        if from_llvm == to_llvm:
            return reg
        # Integer truncation/extension
        from_bits = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}
        if from_llvm in from_bits and to_llvm in from_bits:
            if from_bits[from_llvm] > from_bits[to_llvm]:
                result = self._tmp()
                self._emit(f'  {result} = trunc {from_llvm} {reg} to {to_llvm}')
                return result
            else:
                result = self._tmp()
                op = "sext" if from_type in SIGNED_INT_TYPES else "zext"
                self._emit(f'  {result} = {op} {from_llvm} {reg} to {to_llvm}')
                return result
        # Float/Double conversion
        if from_llvm == "double" and to_llvm == "float":
            result = self._tmp()
            self._emit(f'  {result} = fptrunc double {reg} to float')
            return result
        if from_llvm == "float" and to_llvm == "double":
            result = self._tmp()
            self._emit(f'  {result} = fpext float {reg} to double')
            return result
        return reg

    def _str_const(self, s: str) -> tuple[str, int]:
        """Emit a global string constant, return (global name, byte length including null)."""
        self._str_counter += 1
        name = f"@.str.{self._str_counter}"
        encoded = []
        for ch in s:
            o = ord(ch)
            if o == 10:
                encoded.append("\\0A")
            elif o == 9:
                encoded.append("\\09")
            elif o == 0:
                encoded.append("\\00")
            elif o == 92:
                encoded.append("\\5C")
            elif o == 34:
                encoded.append("\\22")
            elif 32 <= o < 127:
                encoded.append(ch)
            else:
                encoded.append(f"\\{o:02X}")
        ir_str = "".join(encoded) + "\\00"
        byte_len = len(s.encode("utf-8")) + 1
        self._emit_global(
            f'{name} = private unnamed_addr constant [{byte_len} x i8] c"{ir_str}"'
        )
        return name, byte_len

    # --- Class layout helpers ---

    def _resolve_class(self, class_name: str) -> str:
        """Resolve a class name, qualifying with current module if needed."""
        if class_name in self._class_info:
            return class_name
        if self._current_module:
            qual = f"{self._current_module}.{class_name}"
            if qual in self._class_info:
                return qual
        return class_name

    def _resolve_fn(self, fn_name: str) -> str:
        """Resolve a function name, qualifying with current module if needed."""
        if fn_name in self._fn_sigs:
            return fn_name
        if self._current_module:
            qual = f"{self._current_module}.{fn_name}"
            if qual in self._fn_sigs:
                return qual
        return fn_name

    def _struct_name(self, class_name: str) -> str:
        # "math.Vec2" -> "%struct.math_Vec2"
        return f"%struct.{class_name.replace('.', '_')}"

    def _field_index(self, class_name: str, field_name: str) -> int:
        """Get field index in struct (classes: +1 for vtable ptr, structs: no offset)."""
        ci = self._class_info[class_name]
        fields = list(ci.fields.keys())
        offset = 0 if ci.is_struct else 1  # structs have no vtable ptr
        return fields.index(field_name) + offset

    def _field_type(self, class_name: str, field_name: str) -> str:
        return self._class_info[class_name].fields[field_name]

    def _peek_type(self, expr: Expr) -> str | None:
        """Get the Senpai type of an expression without generating code."""
        if isinstance(expr, Var):
            if expr.name == "super" and self._current_class:
                return self._class_info[self._current_class].parent_name
            if expr.name in self._vars:
                return self._vars[expr.name][1]
            # Module names
            if hasattr(self, '_modules') and expr.name in self._modules:
                return f"Module:{expr.name}"
        if isinstance(expr, FieldAccess):
            obj_type = self._peek_type(expr.obj)
            if obj_type:
                resolved = self._resolve_class(obj_type)
                if resolved in self._class_info:
                    ci = self._class_info[resolved]
                    if expr.field_name in ci.fields:
                        return ci.fields[expr.field_name]
        if isinstance(expr, Call):
            if _is_array_type(expr.func):
                return expr.func
            resolved = self._resolve_class(expr.func)
            if resolved in self._class_info and expr.func != "Object":
                return resolved
            resolved_fn = self._resolve_fn(expr.func)
            if resolved_fn in self._fn_sigs:
                return self._fn_sigs[resolved_fn]
        if isinstance(expr, MethodCall):
            obj_type = self._peek_type(expr.obj)
            if obj_type and _is_array_type(obj_type):
                elem = _array_elem_type(obj_type)
                if expr.method == "get":
                    return elem
                if expr.method == "len":
                    return "I64"
                if expr.method in ("push", "set"):
                    return "Void"
            if obj_type and obj_type.startswith("Module:"):
                mod_name = obj_type[7:]
                qual_fn = f"{mod_name}.{expr.method}"
                if qual_fn in self._fn_sigs:
                    return self._fn_sigs[qual_fn]
                # Class constructor
                qual_cls = f"{mod_name}.{expr.method}"
                if qual_cls in self._class_info:
                    return qual_cls
            if obj_type:
                resolved = self._resolve_class(obj_type)
                if resolved in self._class_info:
                    ci = self._class_info[resolved]
                    if expr.method in ci.methods:
                        return ci.methods[expr.method].ret_type
        if isinstance(expr, CastExpr):
            return resolve_type(expr.target_type)
        if isinstance(expr, SizeofExpr):
            return "I64"
        if isinstance(expr, TernaryExpr):
            return self._peek_type(expr.true_expr)
        if isinstance(expr, NilLit):
            return "Ptr"
        return None

    def _vtable_index(self, class_name: str, method_name: str) -> int:
        ci = self._class_info[class_name]
        return ci.vtable_order.index(method_name)

    def _method_fn_name(self, class_name: str, method_name: str) -> str:
        # "math.Vec2" -> "@senpai_math_Vec2_method"
        return f"@senpai_{class_name.replace('.', '_')}_{method_name}"

    # --- Top-level generation ---

    def generate(self, prog: Program) -> str:
        """Generate LLVM IR for an entire program."""
        # Collect class info from type checker
        self._class_info = prog.class_info

        # Collect function signatures
        for fn in prog.functions:
            self._fn_sigs[fn.name] = resolve_type(fn.ret_type)
            self._fn_params[fn.name] = [resolve_type(p.type_name) for p in fn.params]

        # Collect method signatures
        for cls in prog.classes:
            for m in cls.methods:
                full_name = f"{cls.name}_{m.name}"
                ret = resolve_type(m.ret_type)
                self._fn_sigs[full_name] = ret
                params = []
                for p in m.params:
                    if p.name == "self":
                        params.append(cls.name)
                    else:
                        params.append(resolve_type(p.type_name))
                self._fn_params[full_name] = params

        # Process imported modules
        if hasattr(prog, 'module_info'):
            self._modules = prog.module_info
            for mod_name, mi in prog.module_info.items():
                # Register qualified class info from type checker
                for cls_name, ci in mi.classes.items():
                    qual_name = f"{mod_name}.{cls_name}"
                    self._class_info[qual_name] = ci
                # Register function sigs from module_info
                for fn_name, sig in mi.functions.items():
                    qual_name = f"{mod_name}.{fn_name}"
                    self._fn_sigs[qual_name] = sig.ret_type
                    self._fn_params[qual_name] = list(sig.param_types)
                # Register method sigs from qualified class info
                for cls_name, ci in mi.classes.items():
                    qual_cls = f"{mod_name}.{cls_name}"
                    for mname, sig in ci.methods.items():
                        full_name = f"{qual_cls}_{mname}"
                        self._fn_sigs[full_name] = sig.ret_type
                        self._fn_params[full_name] = [qual_cls] + list(sig.param_types)

        # External declarations — track what's been declared to avoid duplicates
        declared = set()

        def _declare(name: str, sig: str):
            if name not in declared:
                declared.add(name)
                self._emit_global(sig)

        self._emit_global('; External declarations')
        _declare('printf',  'declare i32 @printf(ptr, ...)')
        _declare('puts',    'declare i32 @puts(ptr)')
        _declare('malloc',  'declare ptr @malloc(i64)')
        _declare('strlen',  'declare i64 @strlen(ptr)')
        _declare('memcpy',  'declare ptr @memcpy(ptr, ptr, i64)')
        _declare('snprintf','declare i32 @snprintf(ptr, i64, ptr, ...)')
        _declare('realloc', 'declare ptr @realloc(ptr, i64)')
        self._emit_global('')
        # Array struct type: { i64 len, i64 cap, ptr data }
        self._emit_global('%struct.Array = type { i64, i64, ptr }')
        self._emit_global('')

        # Collect extern functions from imported modules too
        all_externs = list(prog.extern_fns)
        if hasattr(prog, 'module_programs'):
            for mod_prog in prog.module_programs.values():
                all_externs.extend(mod_prog.extern_fns)

        # Emit user extern function declarations (skip already declared)
        seen_externs = set()
        for ext in all_externs:
            if ext.name in seen_externs:
                continue
            seen_externs.add(ext.name)
            self._extern_fns.add(ext.name)
            ret = resolve_type(ext.ret_type)
            params = [resolve_type(p.type_name) for p in ext.params]
            self._fn_sigs[ext.name] = ret
            self._fn_params[ext.name] = params
            param_ir = ", ".join(_llvm_type(p) for p in params)
            _declare(ext.name, f'declare {_llvm_type(ret)} @{ext.name}({param_ir})')
        if seen_externs:
            self._emit_global('')

        # Emit struct types and vtables for imported classes
        if hasattr(prog, 'module_programs'):
            for mod_name, mod_prog in prog.module_programs.items():
                for cls in mod_prog.classes:
                    self._emit_class_types_module(cls, mod_name)

        # Emit struct types (no vtable)
        for st in prog.structs:
            ci = self._class_info[st.name]
            field_types = [_llvm_type(ft) for ft in ci.fields.values()]
            struct = self._struct_name(st.name)
            self._emit_global(f'{struct} = type {{ {", ".join(field_types)} }}')
            self._emit_global('')

        # Emit struct types and vtables for classes
        for cls in prog.classes:
            self._emit_class_types(cls)

        # Generate imported module functions and methods
        if hasattr(prog, 'module_programs'):
            for mod_name, mod_prog in prog.module_programs.items():
                for fn in mod_prog.functions:
                    if fn.name == "main":
                        continue
                    self._gen_fn_module(fn, mod_name)
                for cls in mod_prog.classes:
                    self._gen_class_methods_module(cls, mod_name)

        # Generate each free function
        for fn in prog.functions:
            self._gen_fn(fn)

        # Generate methods for each class
        for cls in prog.classes:
            self._gen_class_methods(cls)

        # Assemble final IR
        parts = [
            '; Generated by Senpai compiler',
            f'target triple = "arm64-apple-macosx"',
            '',
        ]
        parts.extend(self._globals)
        parts.append('')
        parts.extend(self._lines)

        # If there's a main(), emit a C-compatible entry point
        if "main" in self._fn_sigs:
            parts.append('')
            parts.append('; C entry point')
            parts.append('define i32 @main() {')
            parts.append('  call void @senpai_main()')
            parts.append('  ret i32 0')
            parts.append('}')

        return "\n".join(parts) + "\n"

    def _emit_class_types(self, cls: ClassDecl):
        """Emit struct type definition and vtable global for a class."""
        ci = self._class_info[cls.name]

        # Struct type: { ptr (vtable), field1_type, field2_type, ... }
        field_types = ["ptr"]  # vtable pointer
        for fname, ftype in ci.fields.items():
            field_types.append(_llvm_type(ftype))
        struct = self._struct_name(cls.name)
        self._emit_global(f'{struct} = type {{ {", ".join(field_types)} }}')

        # Vtable global: array of method function pointers
        if ci.vtable_order:
            vtable_entries = []
            for method_name in ci.vtable_order:
                impl_class = ci.vtable_impl[method_name]
                vtable_entries.append(f'ptr {self._method_fn_name(impl_class, method_name)}')
            n = len(vtable_entries)
            self._emit_global(
                f'@vtable.{cls.name} = global [{n} x ptr] [{", ".join(vtable_entries)}]'
            )
        else:
            self._emit_global(f'@vtable.{cls.name} = global [0 x ptr] zeroinitializer')

        self._emit_global('')

    def _emit_class_types_module(self, cls: ClassDecl, mod_name: str):
        """Emit struct type and vtable for an imported class."""
        qual_name = f"{mod_name}.{cls.name}"
        ci = self._class_info[qual_name]
        safe_name = qual_name.replace(".", "_")

        # Struct type
        field_types = ["ptr"]
        for fname, ftype in ci.fields.items():
            field_types.append(_llvm_type(ftype))
        self._emit_global(f'%struct.{safe_name} = type {{ {", ".join(field_types)} }}')

        # Vtable
        if ci.vtable_order:
            vtable_entries = []
            for method_name in ci.vtable_order:
                impl_class = ci.vtable_impl[method_name]
                # The impl_class is the local class name, prefix with module
                vtable_entries.append(f'ptr @senpai_{mod_name}_{impl_class}_{method_name}')
            n = len(vtable_entries)
            self._emit_global(
                f'@vtable.{safe_name} = global [{n} x ptr] [{", ".join(vtable_entries)}]'
            )
        else:
            self._emit_global(f'@vtable.{safe_name} = global [0 x ptr] zeroinitializer')
        self._emit_global('')

    def _gen_fn_module(self, fn: FnDecl, mod_name: str):
        """Generate a module function with module-prefixed name."""
        self._tmp_counter = 0
        self._vars = {}
        self._current_module = mod_name

        ret_type = resolve_type(fn.ret_type)
        llvm_ret = _llvm_type(ret_type)

        params = []
        for p in fn.params:
            pt = resolve_type(p.type_name)
            params.append(f"{_llvm_type(pt)} %param_{p.name}")

        fn_name = f"senpai_{mod_name}_{fn.name}"
        self._emit(f'define {llvm_ret} @{fn_name}({", ".join(params)}) {{')
        self._emit('entry:')

        for p in fn.params:
            pt = resolve_type(p.type_name)
            lt = _llvm_type(pt)
            alloca = self._tmp()
            self._emit(f'  {alloca} = alloca {lt}')
            self._emit(f'  store {lt} %param_{p.name}, ptr {alloca}')
            self._vars[p.name] = (alloca, pt)

        self._gen_stmts(fn.body, ret_type)

        if ret_type == "Void":
            self._emit('  ret void')
        else:
            zero = _zero_value(ret_type)
            self._emit(f'  ret {llvm_ret} {zero}')

        self._emit('}')
        self._emit('')
        self._current_module = None

    def _gen_class_methods_module(self, cls: ClassDecl, mod_name: str):
        """Generate methods for an imported class."""
        qual_name = f"{mod_name}.{cls.name}"
        self._current_module = mod_name
        for method in cls.methods:
            self._gen_method(qual_name, method)
        self._current_module = None

    def _gen_class_methods(self, cls: ClassDecl):
        """Generate LLVM IR for all methods of a class."""
        for method in cls.methods:
            self._gen_method(cls.name, method)

    def _gen_method(self, class_name: str, method: FnDecl):
        """Generate a single method as a function with self as first param."""
        self._tmp_counter = 0
        self._vars = {}
        self._current_class = class_name

        ret_type = resolve_type(method.ret_type)
        llvm_ret = _llvm_type(ret_type)

        # Build parameter list (self is ptr)
        params = []
        for p in method.params:
            if p.name == "self":
                params.append("ptr %param_self")
            else:
                pt = resolve_type(p.type_name)
                params.append(f"{_llvm_type(pt)} %param_{p.name}")

        fn_name = self._method_fn_name(class_name, method.name).lstrip("@")
        self._emit(f'define {llvm_ret} @{fn_name}({", ".join(params)}) {{')
        self._emit('entry:')

        # Allocate space for parameters and copy them
        for p in method.params:
            if p.name == "self":
                alloca = self._tmp()
                self._emit(f'  {alloca} = alloca ptr')
                self._emit(f'  store ptr %param_self, ptr {alloca}')
                self._vars["self"] = (alloca, class_name)
            else:
                pt = resolve_type(p.type_name)
                lt = _llvm_type(pt)
                alloca = self._tmp()
                self._emit(f'  {alloca} = alloca {lt}')
                self._emit(f'  store {lt} %param_{p.name}, ptr {alloca}')
                self._vars[p.name] = (alloca, pt)

        # Generate body
        self._gen_stmts(method.body, ret_type)

        # Ensure function always terminates
        if ret_type == "Void":
            self._emit('  ret void')
        else:
            zero = _zero_value(ret_type)
            self._emit(f'  ret {llvm_ret} {zero}')

        self._emit('}')
        self._emit('')
        self._current_class = None

    # --- Function generation (unchanged for free functions) ---

    def _gen_fn(self, fn: FnDecl):
        self._tmp_counter = 0
        self._vars = {}

        ret_type = resolve_type(fn.ret_type)
        llvm_ret = _llvm_type(ret_type)

        params = []
        for p in fn.params:
            pt = resolve_type(p.type_name)
            params.append(f"{_llvm_type(pt)} %param_{p.name}")

        fn_name = f"senpai_{fn.name}"
        self._emit(f'define {llvm_ret} @{fn_name}({", ".join(params)}) {{')
        self._emit('entry:')

        for p in fn.params:
            pt = resolve_type(p.type_name)
            lt = _llvm_type(pt)
            alloca = self._tmp()
            self._emit(f'  {alloca} = alloca {lt}')
            self._emit(f'  store {lt} %param_{p.name}, ptr {alloca}')
            self._vars[p.name] = (alloca, pt)

        self._gen_stmts(fn.body, ret_type)

        if ret_type == "Void":
            self._emit('  ret void')
        else:
            zero = _zero_value(ret_type)
            self._emit(f'  ret {llvm_ret} {zero}')

        self._emit('}')
        self._emit('')

    # --- Statement generation ---

    def _gen_stmts(self, stmts: list[Stmt], fn_ret: str):
        for stmt in stmts:
            self._gen_stmt(stmt, fn_ret)

    def _gen_stmt(self, stmt: Stmt, fn_ret: str):
        if isinstance(stmt, LetStmt):
            val_reg, val_type = self._gen_expr(stmt.value)
            st = resolve_type(stmt.type_name) if stmt.type_name else val_type
            lt = _llvm_type(st)
            val_reg = self._coerce(val_reg, val_type, st)
            alloca = self._tmp()
            self._emit(f'  {alloca} = alloca {lt}')
            self._emit(f'  store {lt} {val_reg}, ptr {alloca}')
            self._vars[stmt.name] = (alloca, st)

        elif isinstance(stmt, AssignStmt):
            if isinstance(stmt.target, Var):
                val_reg, val_type = self._gen_expr(stmt.value)
                alloca, st = self._vars[stmt.target.name]
                lt = _llvm_type(st)
                self._emit(f'  store {lt} {val_reg}, ptr {alloca}')
            elif isinstance(stmt.target, FieldAccess):
                self._gen_field_assign(stmt)

        elif isinstance(stmt, ReturnStmt):
            if stmt.value is not None:
                val_reg, val_type = self._gen_expr(stmt.value)
                self._emit(f'  ret {_llvm_type(val_type)} {val_reg}')
            else:
                self._emit('  ret void')
            dead = self._label("dead")
            self._emit(f'{dead}:')

        elif isinstance(stmt, IfStmt):
            self._gen_if(stmt, fn_ret)

        elif isinstance(stmt, WhileStmt):
            self._gen_while(stmt, fn_ret)

        elif isinstance(stmt, ForStmt):
            self._gen_for(stmt, fn_ret)

        elif isinstance(stmt, ExprStmt):
            self._gen_expr(stmt.expr)

    def _gen_field_assign(self, stmt: AssignStmt):
        """Generate field assignment: obj.field = value."""
        obj_reg, obj_type = self._gen_expr(stmt.target.obj)
        obj_type = self._resolve_class(obj_type)
        val_reg, val_type = self._gen_expr(stmt.value)
        field_name = stmt.target.field_name
        field_idx = self._field_index(obj_type, field_name)
        field_stype = self._field_type(obj_type, field_name)
        field_ltype = _llvm_type(field_stype)
        struct = self._struct_name(obj_type)

        ptr = self._tmp()
        self._emit(f'  {ptr} = getelementptr {struct}, ptr {obj_reg}, i32 0, i32 {field_idx}')
        val_reg = self._coerce(val_reg, val_type, field_stype)
        self._emit(f'  store {field_ltype} {val_reg}, ptr {ptr}')

    def _gen_if(self, stmt: IfStmt, fn_ret: str):
        then_label = self._label("if.then")
        end_label = self._label("if.end")

        branches = [(stmt.condition, stmt.body)] + list(stmt.elif_clauses)

        for i, (cond, body) in enumerate(branches):
            next_label = self._label("if.elif") if i < len(branches) - 1 else (
                self._label("if.else") if stmt.else_body else end_label
            )
            cond_reg, _ = self._gen_expr(cond)
            this_label = self._label("if.then")
            self._emit(f'  br i1 {cond_reg}, label %{this_label}, label %{next_label}')
            self._emit(f'{this_label}:')
            self._gen_stmts(body, fn_ret)
            self._emit(f'  br label %{end_label}')
            if i < len(branches) - 1 or stmt.else_body:
                self._emit(f'{next_label}:')

        if stmt.else_body:
            self._gen_stmts(stmt.else_body, fn_ret)
            self._emit(f'  br label %{end_label}')

        self._emit(f'{end_label}:')

    def _gen_while(self, stmt: WhileStmt, fn_ret: str):
        cond_label = self._label("while.cond")
        body_label = self._label("while.body")
        end_label = self._label("while.end")

        self._emit(f'  br label %{cond_label}')
        self._emit(f'{cond_label}:')
        cond_reg, _ = self._gen_expr(stmt.condition)
        self._emit(f'  br i1 {cond_reg}, label %{body_label}, label %{end_label}')

        self._emit(f'{body_label}:')
        self._gen_stmts(stmt.body, fn_ret)
        self._emit(f'  br label %{cond_label}')

        self._emit(f'{end_label}:')

    def _gen_for(self, stmt: ForStmt, fn_ret: str):
        """Generate for x in range(start, end): as a counted loop."""
        # Evaluate start and end
        if stmt.start is not None:
            start_reg, _ = self._gen_expr(stmt.start)
        else:
            start_reg = "0"
        end_reg, _ = self._gen_expr(stmt.end)

        # Allocate loop variable
        alloca = self._tmp()
        self._emit(f'  {alloca} = alloca i64')
        self._emit(f'  store i64 {start_reg}, ptr {alloca}')
        self._vars[stmt.var_name] = (alloca, "I64")

        cond_label = self._label("for.cond")
        body_label = self._label("for.body")
        end_label = self._label("for.end")

        self._emit(f'  br label %{cond_label}')
        self._emit(f'{cond_label}:')
        cur = self._tmp()
        self._emit(f'  {cur} = load i64, ptr {alloca}')
        cmp = self._tmp()
        self._emit(f'  {cmp} = icmp slt i64 {cur}, {end_reg}')
        self._emit(f'  br i1 {cmp}, label %{body_label}, label %{end_label}')

        self._emit(f'{body_label}:')
        self._gen_stmts(stmt.body, fn_ret)
        # Increment
        cur2 = self._tmp()
        self._emit(f'  {cur2} = load i64, ptr {alloca}')
        inc = self._tmp()
        self._emit(f'  {inc} = add i64 {cur2}, 1')
        self._emit(f'  store i64 {inc}, ptr {alloca}')
        self._emit(f'  br label %{cond_label}')

        self._emit(f'{end_label}:')

    # --- Expression generation ---

    def _gen_expr(self, expr: Expr) -> tuple[str, str]:
        """Generate code for an expression. Returns (register, senpai_type)."""
        if isinstance(expr, IntLit):
            return str(expr.value), "I64"

        if isinstance(expr, FloatLit):
            import struct
            if expr.is_float32:
                d = expr.value
                hex_val = "0x%016X" % struct.unpack('<Q', struct.pack('<d', d))[0]
                result = self._tmp()
                self._emit(f'  {result} = fptrunc double {hex_val} to float')
                return result, "Float"
            else:
                d = expr.value
                hex_val = "0x%016X" % struct.unpack('<Q', struct.pack('<d', d))[0]
                return hex_val, "Double"

        if isinstance(expr, BoolLit):
            return ("1" if expr.value else "0"), "Bool"

        if isinstance(expr, NilLit):
            return "null", "Ptr"

        if isinstance(expr, StrLit):
            name, byte_len = self._str_const(expr.value)
            reg = self._tmp()
            self._emit(
                f'  {reg} = getelementptr [{byte_len} x i8], '
                f'ptr {name}, i64 0, i64 0'
            )
            return reg, "Str"

        if isinstance(expr, Var):
            if expr.name == "super" and self._current_class:
                # super returns self pointer with parent class type
                alloca, _ = self._vars["self"]
                reg = self._tmp()
                self._emit(f'  {reg} = load ptr, ptr {alloca}')
                parent = self._class_info[self._current_class].parent_name
                return reg, parent
            alloca, st = self._vars[expr.name]
            lt = _llvm_type(st)
            reg = self._tmp()
            self._emit(f'  {reg} = load {lt}, ptr {alloca}')
            return reg, st

        if isinstance(expr, FieldAccess):
            return self._gen_field_access(expr)

        if isinstance(expr, MethodCall):
            return self._gen_method_call(expr)

        if isinstance(expr, BinOp):
            return self._gen_binop(expr)

        if isinstance(expr, UnaryOp):
            return self._gen_unaryop(expr)

        if isinstance(expr, Call):
            return self._gen_call(expr)

        if isinstance(expr, CastExpr):
            return self._gen_cast(expr)

        if isinstance(expr, SizeofExpr):
            target = resolve_type(expr.type_name)
            struct = self._struct_name(target)
            size_ptr = self._tmp()
            self._emit(f'  {size_ptr} = getelementptr {struct}, ptr null, i32 1')
            size = self._tmp()
            self._emit(f'  {size} = ptrtoint ptr {size_ptr} to i64')
            return size, "I64"

        if isinstance(expr, TernaryExpr):
            cond_reg, _ = self._gen_expr(expr.condition)
            true_reg, true_type = self._gen_expr(expr.true_expr)
            false_reg, _ = self._gen_expr(expr.false_expr)
            llvm_t = _llvm_type(true_type)
            result = self._tmp()
            self._emit(f'  {result} = select i1 {cond_reg}, {llvm_t} {true_reg}, {llvm_t} {false_reg}')
            return result, true_type

        raise RuntimeError(f"unsupported expr: {type(expr).__name__}")

    def _gen_field_access(self, expr: FieldAccess) -> tuple[str, str]:
        """Generate field read: obj.field."""
        obj_reg, obj_type = self._gen_expr(expr.obj)
        obj_type = self._resolve_class(obj_type)
        field_name = expr.field_name
        field_idx = self._field_index(obj_type, field_name)
        field_stype = self._field_type(obj_type, field_name)
        field_ltype = _llvm_type(field_stype)
        struct = self._struct_name(obj_type)

        ptr = self._tmp()
        self._emit(f'  {ptr} = getelementptr {struct}, ptr {obj_reg}, i32 0, i32 {field_idx}')
        result = self._tmp()
        self._emit(f'  {result} = load {field_ltype}, ptr {ptr}')
        return result, field_stype

    def _gen_method_call(self, expr: MethodCall) -> tuple[str, str]:
        """Generate method call with vtable dispatch (or direct for super)."""
        # Built-in to_str() on primitive types
        if expr.method == "to_str" and len(expr.args) == 0:
            obj_type = self._peek_type(expr.obj)
            if obj_type and (obj_type in NUMERIC_TYPES or obj_type == "Bool"):
                return self._gen_to_str(expr.obj, obj_type)

        # Array methods
        obj_type = self._peek_type(expr.obj)
        if obj_type and _is_array_type(obj_type):
            return self._gen_array_method(expr, obj_type)

        # Module-qualified call: module.func() or module.Class()
        if obj_type and obj_type.startswith("Module:"):
            return self._gen_module_call(expr, obj_type[7:])

        # Check if this is a super call (direct dispatch, not vtable)
        is_super = isinstance(expr.obj, Var) and expr.obj.name == "super"

        obj_reg, obj_type = self._gen_expr(expr.obj)
        obj_type = self._resolve_class(obj_type)
        ci = self._class_info[obj_type]
        method_name = expr.method
        sig = ci.methods[method_name]
        ret_type = sig.ret_type
        llvm_ret = _llvm_type(ret_type)

        # Build args: self first, then actual args
        args_ir = [f"ptr {obj_reg}"]
        for i, arg in enumerate(expr.args):
            ar, at = self._gen_expr(arg)
            if i < len(sig.param_types):
                ar = self._coerce(ar, at, sig.param_types[i])
                at = sig.param_types[i]
            args_ir.append(f"{_llvm_type(at)} {ar}")

        if is_super:
            # Direct call to parent's method implementation
            # For __init__ and other non-vtable methods, use the class that defines it
            impl_class = ci.vtable_impl.get(method_name, obj_type)
            # __init__ is always on the defining class
            if method_name == "__init__":
                impl_class = obj_type
            impl_class = self._resolve_class(impl_class)
            fn_name = self._method_fn_name(impl_class, method_name)
            if ret_type == "Void":
                self._emit(f'  call void {fn_name}({", ".join(args_ir)})')
                return "void", "Void"
            else:
                result = self._tmp()
                self._emit(f'  {result} = call {llvm_ret} {fn_name}({", ".join(args_ir)})')
                # For super calls returning self-type, return current class type
                actual_ret = self._current_class if ret_type == obj_type and self._current_class else ret_type
                return result, actual_ret
        else:
            # Vtable dispatch
            vtable_idx = self._vtable_index(obj_type, method_name)
            vtable_ptr_ptr = self._tmp()
            struct = self._struct_name(obj_type)
            self._emit(f'  {vtable_ptr_ptr} = getelementptr {struct}, ptr {obj_reg}, i32 0, i32 0')
            vtable_ptr = self._tmp()
            self._emit(f'  {vtable_ptr} = load ptr, ptr {vtable_ptr_ptr}')

            method_ptr_ptr = self._tmp()
            n = len(ci.vtable_order)
            self._emit(f'  {method_ptr_ptr} = getelementptr [{n} x ptr], ptr {vtable_ptr}, i32 0, i32 {vtable_idx}')
            method_ptr = self._tmp()
            self._emit(f'  {method_ptr} = load ptr, ptr {method_ptr_ptr}')

            param_types_ir = ["ptr"]
            for pt in sig.param_types:
                param_types_ir.append(_llvm_type(pt))
            fn_type = f"{llvm_ret} ({', '.join(param_types_ir)})"

            if ret_type == "Void":
                self._emit(f'  call {fn_type} {method_ptr}({", ".join(args_ir)})')
                return "void", "Void"
            else:
                result = self._tmp()
                self._emit(f'  {result} = call {fn_type} {method_ptr}({", ".join(args_ir)})')
                return result, ret_type

    # --- Binary/Unary ops ---

    def _gen_binop(self, expr: BinOp) -> tuple[str, str]:
        # Check if left operand is a class type — if so, use operator methods
        # (peek at type without generating code yet)
        left_type = self._peek_type(expr.left)
        if left_type:
            left_type = self._resolve_class(left_type)
        if left_type and left_type in self._class_info:
            from .types import BINOP_METHODS
            method_name = BINOP_METHODS.get(expr.op)
            if method_name:
                ci = self._class_info[left_type]
                actual_method = "__eq__" if expr.op in ("==", "!=") else method_name
                if actual_method in ci.methods:
                    method_call = MethodCall(
                        obj=expr.left, method=actual_method,
                        args=[expr.right], line=expr.line
                    )
                    call_result, call_type = self._gen_method_call(method_call)
                    if expr.op == "!=":
                        neg = self._tmp()
                        self._emit(f'  {neg} = xor i1 {call_result}, 1')
                        return neg, "Bool"
                    return call_result, call_type

        lr, lt = self._gen_expr(expr.left)
        rr, rt = self._gen_expr(expr.right)
        result = self._tmp()

        # Integer arithmetic
        if lt in ALL_INT_TYPES and rt in ALL_INT_TYPES:
            llvm_t = _llvm_type(lt)
            is_signed = lt in SIGNED_INT_TYPES

            if expr.op == "+":
                self._emit(f'  {result} = add {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "-":
                self._emit(f'  {result} = sub {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "*":
                self._emit(f'  {result} = mul {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "/":
                div_op = "sdiv" if is_signed else "udiv"
                self._emit(f'  {result} = {div_op} {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "%":
                rem_op = "srem" if is_signed else "urem"
                self._emit(f'  {result} = {rem_op} {llvm_t} {lr}, {rr}')
                return result, lt

            # Bitwise ops
            if expr.op == "&":
                self._emit(f'  {result} = and {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "|":
                self._emit(f'  {result} = or {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "^":
                self._emit(f'  {result} = xor {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "<<":
                rr_coerced = self._coerce(rr, rt, lt)
                self._emit(f'  {result} = shl {llvm_t} {lr}, {rr_coerced}')
                return result, lt
            if expr.op == ">>":
                rr_coerced = self._coerce(rr, rt, lt)
                shr_op = "ashr" if is_signed else "lshr"
                self._emit(f'  {result} = {shr_op} {llvm_t} {lr}, {rr_coerced}')
                return result, lt

            cmp_ops = {
                "==": "eq", "!=": "ne",
                "<": "slt" if is_signed else "ult",
                ">": "sgt" if is_signed else "ugt",
                "<=": "sle" if is_signed else "ule",
                ">=": "sge" if is_signed else "uge",
            }
            if expr.op in cmp_ops:
                self._emit(f'  {result} = icmp {cmp_ops[expr.op]} {llvm_t} {lr}, {rr}')
                return result, "Bool"

        # Float/Double arithmetic
        if lt in FLOAT_TYPES and rt in FLOAT_TYPES:
            llvm_t = _llvm_type(lt)
            if expr.op == "+":
                self._emit(f'  {result} = fadd {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "-":
                self._emit(f'  {result} = fsub {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "*":
                self._emit(f'  {result} = fmul {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "/":
                self._emit(f'  {result} = fdiv {llvm_t} {lr}, {rr}')
                return result, lt
            if expr.op == "%":
                self._emit(f'  {result} = frem {llvm_t} {lr}, {rr}')
                return result, lt

            cmp_ops = {
                "==": "oeq", "!=": "one",
                "<": "olt", ">": "ogt",
                "<=": "ole", ">=": "oge",
            }
            if expr.op in cmp_ops:
                self._emit(f'  {result} = fcmp {cmp_ops[expr.op]} {llvm_t} {lr}, {rr}')
                return result, "Bool"

        # String concatenation
        if lt == "Str" and rt == "Str" and expr.op == "+":
            len_a = self._tmp()
            len_b = self._tmp()
            total = self._tmp()
            total1 = self._tmp()
            buf = self._tmp()
            dst = self._tmp()
            self._emit(f'  {len_a} = call i64 @strlen(ptr {lr})')
            self._emit(f'  {len_b} = call i64 @strlen(ptr {rr})')
            self._emit(f'  {total} = add i64 {len_a}, {len_b}')
            self._emit(f'  {total1} = add i64 {total}, 1')
            self._emit(f'  {buf} = call ptr @malloc(i64 {total1})')
            self._emit(f'  call ptr @memcpy(ptr {buf}, ptr {lr}, i64 {len_a})')
            self._emit(f'  {dst} = getelementptr i8, ptr {buf}, i64 {len_a}')
            len_b1 = self._tmp()
            self._emit(f'  {len_b1} = add i64 {len_b}, 1')
            self._emit(f'  call ptr @memcpy(ptr {dst}, ptr {rr}, i64 {len_b1})')
            return buf, "Str"

        # Pointer comparison
        if lt == "Ptr" and rt == "Ptr" and expr.op in ("==", "!="):
            cmp = "eq" if expr.op == "==" else "ne"
            self._emit(f'  {result} = icmp {cmp} ptr {lr}, {rr}')
            return result, "Bool"

        # Boolean logic
        if lt == "Bool" and rt == "Bool":
            if expr.op == "and":
                self._emit(f'  {result} = and i1 {lr}, {rr}')
                return result, "Bool"
            if expr.op == "or":
                self._emit(f'  {result} = or i1 {lr}, {rr}')
                return result, "Bool"
            if expr.op == "==":
                self._emit(f'  {result} = icmp eq i1 {lr}, {rr}')
                return result, "Bool"
            if expr.op == "!=":
                self._emit(f'  {result} = icmp ne i1 {lr}, {rr}')
                return result, "Bool"

        raise RuntimeError(f"unsupported binop: {expr.op} on {lt}, {rt}")

    def _gen_unaryop(self, expr: UnaryOp) -> tuple[str, str]:
        # Check for class operator method first (avoid double-eval)
        if expr.op == "-":
            op_type = self._peek_type(expr.operand)
            if op_type:
                op_type = self._resolve_class(op_type)
            if op_type and op_type in self._class_info:
                ci = self._class_info[op_type]
                if "__neg__" in ci.methods:
                    method_call = MethodCall(
                        obj=expr.operand, method="__neg__",
                        args=[], line=expr.line
                    )
                    return self._gen_method_call(method_call)

        or_, ot = self._gen_expr(expr.operand)
        result = self._tmp()

        if expr.op == "-" and ot in ALL_INT_TYPES:
            self._emit(f'  {result} = sub {_llvm_type(ot)} 0, {or_}')
            return result, ot

        if expr.op == "-" and ot in FLOAT_TYPES:
            self._emit(f'  {result} = fneg {_llvm_type(ot)} {or_}')
            return result, ot

        if expr.op == "not" and ot == "Bool":
            self._emit(f'  {result} = xor i1 {or_}, 1')
            return result, "Bool"

        if expr.op == "~" and ot in ALL_INT_TYPES:
            self._emit(f'  {result} = xor {_llvm_type(ot)} {or_}, -1')
            return result, ot

        raise RuntimeError(f"unsupported unary: {expr.op} on {ot}")

    # --- Call generation ---

    def _gen_call(self, expr: Call) -> tuple[str, str]:
        # Built-in print
        if expr.func == "print" and len(expr.args) == 1:
            return self._gen_print(expr.args[0])

        # Array constructor
        if _is_array_type(expr.func):
            return self._gen_array_constructor(expr.func, expr)

        # Struct or class constructor call
        resolved_cls = self._resolve_class(expr.func)
        if resolved_cls in self._class_info and expr.func != "Object":
            ci = self._class_info[resolved_cls]
            if ci.is_struct:
                return self._gen_struct_constructor(expr, resolved_cls)
            return self._gen_constructor(expr, resolved_cls)

        # User function
        resolved_fn = self._resolve_fn(expr.func)
        ret_type = self._fn_sigs.get(resolved_fn, "Void")
        llvm_ret = _llvm_type(ret_type)

        param_types = self._fn_params.get(resolved_fn, [])
        args_ir = []
        for i, arg in enumerate(expr.args):
            ar, at = self._gen_expr(arg)
            if i < len(param_types):
                ar = self._coerce(ar, at, param_types[i])
                at = param_types[i]
            args_ir.append(f"{_llvm_type(at)} {ar}")

        # Extern functions use their real C name; others get senpai_ prefix
        if resolved_fn in self._extern_fns:
            ir_name = resolved_fn
        else:
            ir_name = f"senpai_{resolved_fn.replace('.', '_')}"
        if ret_type == "Void":
            self._emit(f'  call void @{ir_name}({", ".join(args_ir)})')
            return "void", "Void"
        else:
            result = self._tmp()
            self._emit(f'  {result} = call {llvm_ret} @{ir_name}({", ".join(args_ir)})')
            return result, ret_type

    def _gen_struct_constructor(self, expr: Call, struct_name: str) -> tuple[str, str]:
        """Generate struct constructor: malloc + zero-init."""
        struct = self._struct_name(struct_name)

        # Calculate size via GEP trick
        size_ptr = self._tmp()
        self._emit(f'  {size_ptr} = getelementptr {struct}, ptr null, i32 1')
        size = self._tmp()
        self._emit(f'  {size} = ptrtoint ptr {size_ptr} to i64')

        # Allocate and zero-initialize
        obj = self._tmp()
        self._emit(f'  {obj} = call ptr @malloc(i64 {size})')
        ci = self._class_info[struct_name]
        for i, (fname, ftype) in enumerate(ci.fields.items()):
            flt = _llvm_type(ftype)
            ptr = self._tmp()
            self._emit(f'  {ptr} = getelementptr {struct}, ptr {obj}, i32 0, i32 {i}')
            self._emit(f'  store {flt} {_zero_value(ftype)}, ptr {ptr}')

        return obj, struct_name

    def _gen_constructor(self, expr: Call, class_name: str | None = None) -> tuple[str, str]:
        """Generate constructor: malloc + store vtable + call __init__."""
        if class_name is None:
            class_name = expr.func
        ci = self._class_info[class_name]
        struct = self._struct_name(class_name)

        # Calculate struct size using GEP trick
        size_ptr = self._tmp()
        self._emit(f'  {size_ptr} = getelementptr {struct}, ptr null, i32 1')
        size = self._tmp()
        self._emit(f'  {size} = ptrtoint ptr {size_ptr} to i64')

        # Allocate
        obj = self._tmp()
        self._emit(f'  {obj} = call ptr @malloc(i64 {size})')

        # Store vtable pointer at offset 0
        vtable_ptr = self._tmp()
        self._emit(f'  {vtable_ptr} = getelementptr {struct}, ptr {obj}, i32 0, i32 0')
        n = len(ci.vtable_order)
        safe_name = class_name.replace(".", "_")
        self._emit(f'  store ptr @vtable.{safe_name}, ptr {vtable_ptr}')

        # Call __init__ if it exists
        if "__init__" in ci.methods:
            init_sig = ci.methods["__init__"]
            args_ir = [f"ptr {obj}"]
            for i, arg in enumerate(expr.args):
                ar, at = self._gen_expr(arg)
                if i < len(init_sig.param_types):
                    ar = self._coerce(ar, at, init_sig.param_types[i])
                    at = init_sig.param_types[i]
                args_ir.append(f"{_llvm_type(at)} {ar}")
            self._emit(f'  call void {self._method_fn_name(class_name, "__init__")}({", ".join(args_ir)})')

        return obj, class_name

    # --- Module calls ---

    def _gen_module_call(self, expr: MethodCall, mod_name: str) -> tuple[str, str]:
        """Generate a module-qualified function call or constructor."""
        qual_cls = f"{mod_name}.{expr.method}"
        qual_fn = f"{mod_name}.{expr.method}"

        # Class constructor
        if qual_cls in self._class_info:
            ci = self._class_info[qual_cls]
            struct = self._struct_name(qual_cls)

            # Calculate struct size
            size_ptr = self._tmp()
            self._emit(f'  {size_ptr} = getelementptr {struct}, ptr null, i32 1')
            size = self._tmp()
            self._emit(f'  {size} = ptrtoint ptr {size_ptr} to i64')

            # Allocate
            obj = self._tmp()
            self._emit(f'  {obj} = call ptr @malloc(i64 {size})')

            # Store vtable pointer
            vtable_ptr = self._tmp()
            self._emit(f'  {vtable_ptr} = getelementptr {struct}, ptr {obj}, i32 0, i32 0')
            safe_name = qual_cls.replace(".", "_")
            self._emit(f'  store ptr @vtable.{safe_name}, ptr {vtable_ptr}')

            # Call __init__
            if "__init__" in ci.methods:
                init_sig = ci.methods["__init__"]
                args_ir = [f"ptr {obj}"]
                for i, arg in enumerate(expr.args):
                    ar, at = self._gen_expr(arg)
                    if i < len(init_sig.param_types):
                        ar = self._coerce(ar, at, init_sig.param_types[i])
                        at = init_sig.param_types[i]
                    args_ir.append(f"{_llvm_type(at)} {ar}")
                self._emit(f'  call void {self._method_fn_name(qual_cls, "__init__")}({", ".join(args_ir)})')

            return obj, qual_cls

        # Module function call
        if qual_fn in self._fn_sigs:
            ret_type = self._fn_sigs[qual_fn]
            llvm_ret = _llvm_type(ret_type)
            param_types = self._fn_params.get(qual_fn, [])

            args_ir = []
            for i, arg in enumerate(expr.args):
                ar, at = self._gen_expr(arg)
                if i < len(param_types):
                    ar = self._coerce(ar, at, param_types[i])
                    at = param_types[i]
                args_ir.append(f"{_llvm_type(at)} {ar}")

            fn_ir_name = f"@senpai_{mod_name}_{expr.method}"
            if ret_type == "Void":
                self._emit(f'  call void {fn_ir_name}({", ".join(args_ir)})')
                return "void", "Void"
            else:
                result = self._tmp()
                self._emit(f'  {result} = call {llvm_ret} {fn_ir_name}({", ".join(args_ir)})')
                return result, ret_type

        raise RuntimeError(f"module '{mod_name}' has no '{expr.method}'")

    # --- Array ---

    def _elem_size(self, elem_type: str) -> int:
        """Get element size in bytes."""
        sizes = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "float": 4, "double": 8, "i1": 1, "ptr": 8}
        return sizes.get(_llvm_type(elem_type), 8)

    def _gen_array_constructor(self, array_type: str, expr: Call) -> tuple[str, str]:
        """Generate Array[T]() — allocates struct + initial data buffer."""
        # Allocate the struct {i64 len, i64 cap, ptr data}
        arr = self._tmp()
        self._emit(f'  {arr} = call ptr @malloc(i64 24)')  # 3 * i64 = 24 bytes

        # Store len = 0
        len_ptr = self._tmp()
        self._emit(f'  {len_ptr} = getelementptr %struct.Array, ptr {arr}, i32 0, i32 0')
        self._emit(f'  store i64 0, ptr {len_ptr}')

        # Store cap = 8 (initial capacity)
        cap_ptr = self._tmp()
        self._emit(f'  {cap_ptr} = getelementptr %struct.Array, ptr {arr}, i32 0, i32 1')
        self._emit(f'  store i64 8, ptr {cap_ptr}')

        # Allocate initial data buffer
        elem_type = _array_elem_type(array_type)
        elem_sz = self._elem_size(elem_type)
        data = self._tmp()
        self._emit(f'  {data} = call ptr @malloc(i64 {elem_sz * 8})')

        # Store data pointer
        data_ptr = self._tmp()
        self._emit(f'  {data_ptr} = getelementptr %struct.Array, ptr {arr}, i32 0, i32 2')
        self._emit(f'  store ptr {data}, ptr {data_ptr}')

        return arr, array_type

    def _gen_array_method(self, expr: MethodCall, array_type: str) -> tuple[str, str]:
        """Generate array method calls."""
        elem_type = _array_elem_type(array_type)
        elem_llvm = _llvm_type(elem_type)
        elem_sz = self._elem_size(elem_type)
        obj_reg, _ = self._gen_expr(expr.obj)

        if expr.method == "len":
            len_ptr = self._tmp()
            self._emit(f'  {len_ptr} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 0')
            result = self._tmp()
            self._emit(f'  {result} = load i64, ptr {len_ptr}')
            return result, "I64"

        if expr.method == "get":
            idx_reg, idx_type = self._gen_expr(expr.args[0])
            idx_reg = self._coerce(idx_reg, idx_type, "I64")
            # Load data pointer
            data_ptr_ptr = self._tmp()
            self._emit(f'  {data_ptr_ptr} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 2')
            data_ptr = self._tmp()
            self._emit(f'  {data_ptr} = load ptr, ptr {data_ptr_ptr}')
            # GEP to element
            elem_ptr = self._tmp()
            self._emit(f'  {elem_ptr} = getelementptr {elem_llvm}, ptr {data_ptr}, i64 {idx_reg}')
            result = self._tmp()
            self._emit(f'  {result} = load {elem_llvm}, ptr {elem_ptr}')
            return result, elem_type

        if expr.method == "set":
            idx_reg, idx_type = self._gen_expr(expr.args[0])
            idx_reg = self._coerce(idx_reg, idx_type, "I64")
            val_reg, val_type = self._gen_expr(expr.args[1])
            val_reg = self._coerce(val_reg, val_type, elem_type)
            # Load data pointer
            data_ptr_ptr = self._tmp()
            self._emit(f'  {data_ptr_ptr} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 2')
            data_ptr = self._tmp()
            self._emit(f'  {data_ptr} = load ptr, ptr {data_ptr_ptr}')
            # GEP to element
            elem_ptr = self._tmp()
            self._emit(f'  {elem_ptr} = getelementptr {elem_llvm}, ptr {data_ptr}, i64 {idx_reg}')
            self._emit(f'  store {elem_llvm} {val_reg}, ptr {elem_ptr}')
            return "void", "Void"

        if expr.method == "push":
            val_reg, val_type = self._gen_expr(expr.args[0])
            val_reg = self._coerce(val_reg, val_type, elem_type)

            # Load len and cap
            len_ptr = self._tmp()
            self._emit(f'  {len_ptr} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 0')
            cur_len = self._tmp()
            self._emit(f'  {cur_len} = load i64, ptr {len_ptr}')

            cap_ptr = self._tmp()
            self._emit(f'  {cap_ptr} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 1')
            cur_cap = self._tmp()
            self._emit(f'  {cur_cap} = load i64, ptr {cap_ptr}')

            # Check if we need to grow
            need_grow = self._tmp()
            self._emit(f'  {need_grow} = icmp sge i64 {cur_len}, {cur_cap}')

            grow_label = self._label("arr.grow")
            store_label = self._label("arr.store")

            self._emit(f'  br i1 {need_grow}, label %{grow_label}, label %{store_label}')

            # Grow path: double capacity, realloc
            self._emit(f'{grow_label}:')
            new_cap = self._tmp()
            self._emit(f'  {new_cap} = mul i64 {cur_cap}, 2')
            self._emit(f'  store i64 {new_cap}, ptr {cap_ptr}')
            new_bytes = self._tmp()
            self._emit(f'  {new_bytes} = mul i64 {new_cap}, {elem_sz}')
            data_ptr_ptr_g = self._tmp()
            self._emit(f'  {data_ptr_ptr_g} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 2')
            old_data_g = self._tmp()
            self._emit(f'  {old_data_g} = load ptr, ptr {data_ptr_ptr_g}')
            new_data = self._tmp()
            self._emit(f'  {new_data} = call ptr @realloc(ptr {old_data_g}, i64 {new_bytes})')
            self._emit(f'  store ptr {new_data}, ptr {data_ptr_ptr_g}')
            self._emit(f'  br label %{store_label}')

            # Store path: store element at arr[len], increment len
            self._emit(f'{store_label}:')
            data_ptr_ptr_s = self._tmp()
            self._emit(f'  {data_ptr_ptr_s} = getelementptr %struct.Array, ptr {obj_reg}, i32 0, i32 2')
            data_ptr_s = self._tmp()
            self._emit(f'  {data_ptr_s} = load ptr, ptr {data_ptr_ptr_s}')
            # Reload len (could have been the same, but need to load after phi)
            cur_len2 = self._tmp()
            self._emit(f'  {cur_len2} = load i64, ptr {len_ptr}')
            elem_ptr = self._tmp()
            self._emit(f'  {elem_ptr} = getelementptr {elem_llvm}, ptr {data_ptr_s}, i64 {cur_len2}')
            self._emit(f'  store {elem_llvm} {val_reg}, ptr {elem_ptr}')
            # Increment len
            new_len = self._tmp()
            self._emit(f'  {new_len} = add i64 {cur_len2}, 1')
            self._emit(f'  store i64 {new_len}, ptr {len_ptr}')

            return "void", "Void"

        raise RuntimeError(f"unknown array method: {expr.method}")

    # --- to_str ---

    def _gen_to_str(self, obj_expr: Expr, obj_type: str) -> tuple[str, str]:
        """Generate to_str() for primitive types using snprintf."""
        obj_reg, _ = self._gen_expr(obj_expr)

        # Allocate a buffer (64 bytes is plenty for any number)
        buf = self._tmp()
        self._emit(f'  {buf} = call ptr @malloc(i64 64)')

        if obj_type in SIGNED_INT_TYPES:
            llvm_t = _llvm_type(obj_type)
            if obj_type != "I64":
                ext = self._tmp()
                self._emit(f'  {ext} = sext {llvm_t} {obj_reg} to i64')
                obj_reg = ext
            fmt_name, fmt_len = self._str_const("%lld")
            fmt_reg = self._tmp()
            self._emit(f'  {fmt_reg} = getelementptr [{fmt_len} x i8], ptr {fmt_name}, i64 0, i64 0')
            discard = self._tmp()
            self._emit(f'  {discard} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {buf}, i64 64, ptr {fmt_reg}, i64 {obj_reg})')

        elif obj_type in UNSIGNED_INT_TYPES:
            llvm_t = _llvm_type(obj_type)
            if obj_type != "U64":
                ext = self._tmp()
                self._emit(f'  {ext} = zext {llvm_t} {obj_reg} to i64')
                obj_reg = ext
            fmt_name, fmt_len = self._str_const("%llu")
            fmt_reg = self._tmp()
            self._emit(f'  {fmt_reg} = getelementptr [{fmt_len} x i8], ptr {fmt_name}, i64 0, i64 0')
            discard = self._tmp()
            self._emit(f'  {discard} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {buf}, i64 64, ptr {fmt_reg}, i64 {obj_reg})')

        elif obj_type == "Float":
            ext = self._tmp()
            self._emit(f'  {ext} = fpext float {obj_reg} to double')
            fmt_name, fmt_len = self._str_const("%g")
            fmt_reg = self._tmp()
            self._emit(f'  {fmt_reg} = getelementptr [{fmt_len} x i8], ptr {fmt_name}, i64 0, i64 0')
            discard = self._tmp()
            self._emit(f'  {discard} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {buf}, i64 64, ptr {fmt_reg}, double {ext})')

        elif obj_type == "Double":
            fmt_name, fmt_len = self._str_const("%g")
            fmt_reg = self._tmp()
            self._emit(f'  {fmt_reg} = getelementptr [{fmt_len} x i8], ptr {fmt_name}, i64 0, i64 0')
            discard = self._tmp()
            self._emit(f'  {discard} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {buf}, i64 64, ptr {fmt_reg}, double {obj_reg})')

        elif obj_type == "Bool":
            true_name, true_len = self._str_const("true")
            false_name, false_len = self._str_const("false")
            true_reg = self._tmp()
            self._emit(f'  {true_reg} = getelementptr [{true_len} x i8], ptr {true_name}, i64 0, i64 0')
            false_reg = self._tmp()
            self._emit(f'  {false_reg} = getelementptr [{false_len} x i8], ptr {false_name}, i64 0, i64 0')
            sel = self._tmp()
            self._emit(f'  {sel} = select i1 {obj_reg}, ptr {true_reg}, ptr {false_reg}')
            # For Bool, just return the string constant directly (no need for snprintf)
            # Free the unused buffer... or just return the constant
            return sel, "Str"

        return buf, "Str"

    # --- Cast ---

    def _gen_cast(self, expr: CastExpr) -> tuple[str, str]:
        """Generate type cast: expr as Type."""
        src_reg, src_type = self._gen_expr(expr.expr)
        target = resolve_type(expr.target_type)
        src_llvm = _llvm_type(src_type)
        tgt_llvm = _llvm_type(target)

        if src_llvm == tgt_llvm and src_type != "Bool":
            return src_reg, target  # same LLVM type, just reinterpret

        result = self._tmp()

        # Bool to integer: zext i1 to target
        if src_type == "Bool" and target in ALL_INT_TYPES:
            self._emit(f'  {result} = zext i1 {src_reg} to {tgt_llvm}')
            return result, target

        # Int to float
        if src_type in SIGNED_INT_TYPES and target in FLOAT_TYPES:
            self._emit(f'  {result} = sitofp {src_llvm} {src_reg} to {tgt_llvm}')
            return result, target
        if src_type in UNSIGNED_INT_TYPES and target in FLOAT_TYPES:
            self._emit(f'  {result} = uitofp {src_llvm} {src_reg} to {tgt_llvm}')
            return result, target

        # Float to int
        if src_type in FLOAT_TYPES and target in SIGNED_INT_TYPES:
            self._emit(f'  {result} = fptosi {src_llvm} {src_reg} to {tgt_llvm}')
            return result, target
        if src_type in FLOAT_TYPES and target in UNSIGNED_INT_TYPES:
            self._emit(f'  {result} = fptoui {src_llvm} {src_reg} to {tgt_llvm}')
            return result, target

        # Ptr to integer
        if src_type == "Ptr" and target in ALL_INT_TYPES:
            self._emit(f'  {result} = ptrtoint ptr {src_reg} to {tgt_llvm}')
            return result, target
        # Integer to Ptr
        if src_type in ALL_INT_TYPES and target == "Ptr":
            self._emit(f'  {result} = inttoptr {src_llvm} {src_reg} to ptr')
            return result, target

        # Int width changes and float width changes — use _coerce
        return self._coerce(src_reg, src_type, target), target

    # --- Print ---

    def _gen_print(self, arg: Expr) -> tuple[str, str]:
        ar, at = self._gen_expr(arg)

        if at in SIGNED_INT_TYPES:
            llvm_t = _llvm_type(at)
            if at != "I64":
                ext = self._tmp()
                self._emit(f'  {ext} = sext {llvm_t} {ar} to i64')
                ar = ext
            fmt_name, fmt_len = self._str_const("%lld\n")
            fmt_reg = self._tmp()
            self._emit(
                f'  {fmt_reg} = getelementptr [{fmt_len} x i8], '
                f'ptr {fmt_name}, i64 0, i64 0'
            )
            result = self._tmp()
            self._emit(f'  {result} = call i32 (ptr, ...) @printf(ptr {fmt_reg}, i64 {ar})')

        elif at in UNSIGNED_INT_TYPES:
            llvm_t = _llvm_type(at)
            if at != "U64":
                ext = self._tmp()
                self._emit(f'  {ext} = zext {llvm_t} {ar} to i64')
                ar = ext
            fmt_name, fmt_len = self._str_const("%llu\n")
            fmt_reg = self._tmp()
            self._emit(
                f'  {fmt_reg} = getelementptr [{fmt_len} x i8], '
                f'ptr {fmt_name}, i64 0, i64 0'
            )
            result = self._tmp()
            self._emit(f'  {result} = call i32 (ptr, ...) @printf(ptr {fmt_reg}, i64 {ar})')

        elif at == "Float":
            ext = self._tmp()
            self._emit(f'  {ext} = fpext float {ar} to double')
            fmt_name, fmt_len = self._str_const("%g\n")
            fmt_reg = self._tmp()
            self._emit(
                f'  {fmt_reg} = getelementptr [{fmt_len} x i8], '
                f'ptr {fmt_name}, i64 0, i64 0'
            )
            result = self._tmp()
            self._emit(f'  {result} = call i32 (ptr, ...) @printf(ptr {fmt_reg}, double {ext})')

        elif at == "Double":
            fmt_name, fmt_len = self._str_const("%g\n")
            fmt_reg = self._tmp()
            self._emit(
                f'  {fmt_reg} = getelementptr [{fmt_len} x i8], '
                f'ptr {fmt_name}, i64 0, i64 0'
            )
            result = self._tmp()
            self._emit(f'  {result} = call i32 (ptr, ...) @printf(ptr {fmt_reg}, double {ar})')

        elif at == "Bool":
            true_name, true_len = self._str_const("true")
            false_name, false_len = self._str_const("false")
            true_reg = self._tmp()
            self._emit(
                f'  {true_reg} = getelementptr [{true_len} x i8], '
                f'ptr {true_name}, i64 0, i64 0'
            )
            false_reg = self._tmp()
            self._emit(
                f'  {false_reg} = getelementptr [{false_len} x i8], '
                f'ptr {false_name}, i64 0, i64 0'
            )
            sel = self._tmp()
            self._emit(f'  {sel} = select i1 {ar}, ptr {true_reg}, ptr {false_reg}')
            result = self._tmp()
            self._emit(f'  {result} = call i32 @puts(ptr {sel})')

        elif at == "Str":
            result = self._tmp()
            self._emit(f'  {result} = call i32 @puts(ptr {ar})')

        elif at == "Ptr":
            fmt_name, fmt_len = self._str_const("%p\n")
            fmt_reg = self._tmp()
            self._emit(
                f'  {fmt_reg} = getelementptr [{fmt_len} x i8], '
                f'ptr {fmt_name}, i64 0, i64 0'
            )
            result = self._tmp()
            self._emit(f'  {result} = call i32 (ptr, ...) @printf(ptr {fmt_reg}, ptr {ar})')

        else:
            raise RuntimeError(f"print: unsupported type {at}")

        return "void", "Void"
