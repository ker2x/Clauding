from typing import Optional
"""Lightweight LLVM IR builder helpers for Kouhai."""


class IRModule:
    def __init__(self):
        self.globals: list[str] = []
        self._str_counter = 0

    def declare(self, name, ret_type, param_types, vararg=False):
        params = list(param_types)
        if vararg:
            params.append("...")
        self.globals.append(f"declare {ret_type} @{name}({', '.join(params)})")

    def define(self, name, ret_type, params: list[tuple[str, str]], linkage=""):
        sig = ", ".join([f"{ty} {p}" for p, ty in params])
        prefix = f"{linkage} " if linkage else ""
        return IRFunction(header=f"define {prefix}{ret_type} @{name}({sig}) {{")

    def global_string(self, s: str) -> tuple[str, int]:
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
        self.globals.append(
            f'{name} = private unnamed_addr constant [{byte_len} x i8] c"{ir_str}"'
        )
        return name, byte_len

    def global_array(self, name, elem_type, values: list[str]) -> str:
        n = len(values)
        line = f"{name} = global [{n} x {elem_type}] [{', '.join(values)}]"
        self.globals.append(line)
        return name

    def type_def(self, name, fields: list[str]):
        self.globals.append(f"{name} = type {{ {', '.join(fields)} }}")

    def raw_global(self, line: str):
        self.globals.append(line)

    def emit(self) -> str:
        return "\n".join(self.globals)


class IRFunction:
    def __init__(self, header: str = ""):
        self.header = header
        self.blocks: dict[str, list[str]] = {"entry": []}
        self.order: list[str] = ["entry"]
        self.block = "entry"
        self._tmp_counter = 0
        self._label_counter = 0

    def new_block(self, name="") -> str:
        label = name or self._label()
        if label not in self.blocks:
            self.blocks[label] = []
            self.order.append(label)
        return label

    def set_block(self, label: str):
        if label not in self.blocks:
            self.new_block(label)
        self.block = label

    def current_block(self) -> str:
        return self.block

    def _tmp(self) -> str:
        self._tmp_counter += 1
        return f"%t{self._tmp_counter}"

    def _label(self, prefix="L") -> str:
        self._label_counter += 1
        return f"{prefix}{self._label_counter}"

    def _emit(self, line: str):
        self.blocks[self.block].append(f"  {line}")

    def _binop(self, op, ty, a, b) -> str:
        t = self._tmp()
        self._emit(f"{t} = {op} {ty} {a}, {b}")
        return t

    def add(self, ty, a, b) -> str:
        return self._binop("add", ty, a, b)

    def sub(self, ty, a, b) -> str:
        return self._binop("sub", ty, a, b)

    def mul(self, ty, a, b) -> str:
        return self._binop("mul", ty, a, b)

    def sdiv(self, ty, a, b) -> str:
        return self._binop("sdiv", ty, a, b)

    def udiv(self, ty, a, b) -> str:
        return self._binop("udiv", ty, a, b)

    def srem(self, ty, a, b) -> str:
        return self._binop("srem", ty, a, b)

    def urem(self, ty, a, b) -> str:
        return self._binop("urem", ty, a, b)

    def and_(self, ty, a, b) -> str:
        return self._binop("and", ty, a, b)

    def or_(self, ty, a, b) -> str:
        return self._binop("or", ty, a, b)

    def xor(self, ty, a, b) -> str:
        return self._binop("xor", ty, a, b)

    def shl(self, ty, a, b) -> str:
        return self._binop("shl", ty, a, b)

    def lshr(self, ty, a, b) -> str:
        return self._binop("lshr", ty, a, b)

    def ashr(self, ty, a, b) -> str:
        return self._binop("ashr", ty, a, b)

    def fadd(self, ty, a, b) -> str:
        return self._binop("fadd", ty, a, b)

    def fsub(self, ty, a, b) -> str:
        return self._binop("fsub", ty, a, b)

    def fmul(self, ty, a, b) -> str:
        return self._binop("fmul", ty, a, b)

    def fdiv(self, ty, a, b) -> str:
        return self._binop("fdiv", ty, a, b)

    def icmp(self, pred, ty, a, b) -> str:
        return self._binop(f"icmp {pred}", ty, a, b)

    def fcmp(self, pred, ty, a, b) -> str:
        return self._binop(f"fcmp {pred}", ty, a, b)

    def alloca(self, ty, name="") -> str:
        t = name or self._tmp()
        self._emit(f"{t} = alloca {ty}")
        return t

    def load(self, ty, ptr) -> str:
        t = self._tmp()
        self._emit(f"{t} = load {ty}, ptr {ptr}")
        return t

    def store(self, ty, val, ptr):
        self._emit(f"store {ty} {val}, ptr {ptr}")

    def gep(self, ty, ptr, *indices) -> str:
        t = self._tmp()
        self._emit(
            f"{t} = getelementptr inbounds {ty}, ptr {ptr}, "
            + ", ".join([f"i64 {i}" for i in indices])
        )
        return t

    def trunc(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = trunc {from_ty} {val} to {to_ty}")
        return t

    def zext(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = zext {from_ty} {val} to {to_ty}")
        return t

    def sext(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = sext {from_ty} {val} to {to_ty}")
        return t

    def fptrunc(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = fptrunc {from_ty} {val} to {to_ty}")
        return t

    def fpext(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = fpext {from_ty} {val} to {to_ty}")
        return t

    def fptosi(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = fptosi {from_ty} {val} to {to_ty}")
        return t

    def fptoui(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = fptoui {from_ty} {val} to {to_ty}")
        return t

    def sitofp(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = sitofp {from_ty} {val} to {to_ty}")
        return t

    def uitofp(self, from_ty, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = uitofp {from_ty} {val} to {to_ty}")
        return t

    def ptrtoint(self, val, to_ty) -> str:
        t = self._tmp()
        self._emit(f"{t} = ptrtoint ptr {val} to {to_ty}")
        return t

    def inttoptr(self, val) -> str:
        t = self._tmp()
        self._emit(f"{t} = inttoptr i64 {val} to ptr")
        return t

    def ret(self, ty="void", val=None):
        if ty == "void" or val is None:
            self._emit("ret void")
        else:
            self._emit(f"ret {ty} {val}")

    def br(self, label):
        self._emit(f"br label %{label}")

    def cond_br(self, cond, true_label, false_label):
        self._emit(f"br i1 {cond}, label %{true_label}, label %{false_label}")

    def phi(self, ty, *incoming: tuple[str, str]) -> str:
        t = self._tmp()
        pairs = ", ".join([f"[ {v}, %{lbl} ]" for v, lbl in incoming])
        self._emit(f"{t} = phi {ty} {pairs}")
        return t

    def select(self, cond, ty, true_val, false_val) -> str:
        t = self._tmp()
        self._emit(f"{t} = select i1 {cond}, {ty} {true_val}, {ty} {false_val}")
        return t

    def call(self, ret_ty, fn_ref, args: list[str], arg_types: list[str] = None) -> str:
        if arg_types is None:
            arg_types = ["ptr"] * len(args)
        arg_ir = ", ".join([f"{t} {a}" for t, a in zip(arg_types, args, strict=False)])
        if ret_ty == "void":
            self._emit(f"call void {fn_ref}({arg_ir})")
            return ""
        t = self._tmp()
        self._emit(f"{t} = call {ret_ty} {fn_ref}({arg_ir})")
        return t

    def call_indirect(self, ret_ty, fn_ptr, fn_type, args: list[str]) -> str:
        arg_ir = ", ".join(args)
        if ret_ty == "void":
            self._emit(f"call {ret_ty} {fn_type} {fn_ptr}({arg_ir})")
            return ""
        t = self._tmp()
        self._emit(f"{t} = call {ret_ty} {fn_type} {fn_ptr}({arg_ir})")
        return t

    def unreachable(self):
        self._emit("unreachable")

    def emit(self) -> str:
        lines = []
        if self.header:
            lines.append(self.header)
        for i, label in enumerate(self.order):
            if i == 0 and label == "entry":
                lines.append("entry:")
            else:
                lines.append(f"{label}:")
            lines.extend(self.blocks[label])
        if self.header:
            lines.append("}")
        return "\n".join(lines)
