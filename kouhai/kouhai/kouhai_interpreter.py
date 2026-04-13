"""Kouhai interpreter - executes Kouhai AST directly."""

from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class KouhaiValue:
    """A value in the Kouhai interpreter."""
    value: Any = None
    type_name: str = "Void"


@dataclass  
class KouhaiEnv:
    """Execution environment."""
    variables: dict = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    parent: Optional["KouhaiEnv"] = None


class KouhaiInterpreter:
    """Simple interpreter for Kouhai AST."""
    
    def __init__(self):
        self.env = KouhaiEnv()
        self.setup_builtins()
    
    def setup_builtins(self):
        """Setup built-in functions."""
        # Print function
        def builtin_print(args):
            result = ""
            for i, arg in enumerate(args):
                if i > 0:
                    result += " "
                result += str(arg)
            print(result)
            return KouhaiValue(value=None, type_name="Void")
        
        self.env.functions["print"] = builtin_print
        
        # Array functions
        def array_len(arr):
            if isinstance(arr, list):
                return KouhaiValue(value=len(arr), type_name="I64")
            return KouhaiValue(value=0, type_name="I64")
        self.env.functions["len"] = array_len
    
    def interpret_program(self, program) -> None:
        """Interpret a Kouhai Program AST."""
        # Collect function definitions
        for fn in program.functions:
            self.env.functions[fn.name] = fn
        
        # Execute main if present
        if "main" in self.env.functions:
            self.call_function("main", [])
    
    def call_function(self, name: str, args: list) -> KouhaiValue:
        """Call a function by name."""
        if name not in self.env.functions:
            return KouhaiValue(value=None, type_name="Void")
        
        fn = self.env.functions[name]
        return self.interpret_fn(fn, args)
    
    def interpret_fn(self, fn, args: list) -> KouhaiValue:
        """Interpret a function definition."""
        local_env = KouhaiEnv(parent=self.env)
        
        # Bind parameters
        for i, param in enumerate(fn.params):
            if i < len(args):
                local_env.variables[param.name] = args[i]
            else:
                local_env.variables[param.name] = KouhaiValue()
        
        # Execute body
        old_env = self.env
        self.env = local_env
        
        for stmt in fn.body:
            result = self.interpret_stmt(stmt)
            if hasattr(stmt, 'is_return') and stmt.is_return:
                self.env = old_env
                return result
        
        self.env = old_env
        return KouhaiValue(value=None, type_name="Void")
    
    def interpret_stmt(self, stmt) -> KouhaiValue:
        """Interpret a statement."""
        # Handle different statement types
        if hasattr(stmt, 'name') and hasattr(stmt, 'value'):
            # LetStmt
            value = self.interpret_expr(stmt.value)
            self.env.variables[stmt.name] = value
            return KouhaiValue()
        
        if hasattr(stmt, 'value'):
            # ReturnStmt
            return self.interpret_expr(stmt.value)
        
        if hasattr(stmt, 'expr'):
            # ExprStmt
            return self.interpret_expr(stmt.expr)
        
        if hasattr(stmt, 'cond'):
            # IfStmt
            cond = self.interpret_expr(stmt.cond)
            if cond.value:
                for s in stmt.body:
                    self.interpret_stmt(s)
            return KouhaiValue()
        
        return KouhaiValue()
    
    def interpret_expr(self, expr) -> KouhaiValue:
        """Interpret an expression."""
        if expr is None:
            return KouhaiValue(value=None, type_name="Void")
        
        # IntLit
        if hasattr(expr, 'value') and hasattr(expr, 'line') and not hasattr(expr, 'op'):
            return KouhaiValue(value=expr.value, type_name="I64")
        
        # StrLit
        if hasattr(expr, 'value') and hasattr(expr, 'line') and not hasattr(expr, 'line_number'):
            if isinstance(expr.value, str):
                return KouhaiValue(value=expr.value, type_name="Str")
        
        # Var
        if hasattr(expr, 'name') and hasattr(expr, 'line') and not hasattr(expr, 'op'):
            if expr.name in self.env.variables:
                return self.env.variables[expr.name]
            return KouhaiValue(value=0, type_name="I64")
        
        # BinOp
        if hasattr(expr, 'op'):
            left = self.interpret_expr(expr.left)
            right = self.interpret_expr(expr.right)
            
            if expr.op == "+":
                return KouhaiValue(value=left.value + right.value, type_name="I64")
            elif expr.op == "-":
                return KouhaiValue(value=left.value - right.value, type_name="I64")
            elif expr.op == "*":
                return KouhaiValue(value=left.value * right.value, type_name="I64")
            elif expr.op == "==":
                return KouhaiValue(value=left.value == right.value, type_name="Bool")
            
            return KouhaiValue(value=left.value, type_name="I64")
        
        # Call
        if hasattr(expr, 'func'):
            args = [self.interpret_expr(a) for a in expr.args]
            # expr.func can be string or Var object
            if isinstance(expr.func, str):
                name = expr.func
            elif hasattr(expr.func, 'value'):
                name = expr.func.value
            elif hasattr(expr.func, 'name'):
                name = expr.func.name
            else:
                name = str(expr.func)
            return self.call_function(name, args)
        
        return KouhaiValue(value=0, type_name="I64")


def interpret_program(program) -> None:
    """Main entry point to interpret a Kouhai program."""
    interpreter = KouhaiInterpreter()
    interpreter.interpret_program(program)
