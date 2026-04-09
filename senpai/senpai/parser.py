"""Recursive descent parser for Senpai."""

from .tokens import TT, Token
from .ast_nodes import (
    Expr, IntLit, FloatLit, StrLit, BoolLit, NoneLit, Var,
    BinOp, UnaryOp, Call, MethodCall, FieldAccess,
    Stmt, LetStmt, AssignStmt, ReturnStmt, IfStmt, WhileStmt, ExprStmt,
    Param, FnDecl, ClassDecl, Program,
)


class ParseError(Exception):
    def __init__(self, token: Token, msg: str):
        super().__init__(f"line {token.line}, col {token.col}: {msg}")
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # --- Helpers ---

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _at(self, *types: TT) -> bool:
        return self._peek().type in types

    def _eat(self, tt: TT) -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise ParseError(tok, f"expected {tt.name}, got {tok.type.name}")
        self.pos += 1
        return tok

    def _eat_newlines(self):
        while self._at(TT.NEWLINE):
            self.pos += 1

    def _skip_newlines(self):
        while self._at(TT.NEWLINE):
            self.pos += 1

    # --- Top level ---

    def parse(self) -> Program:
        prog = Program()
        self._skip_newlines()
        while not self._at(TT.EOF):
            if self._at(TT.FN):
                prog.functions.append(self._parse_fn())
            elif self._at(TT.CLASS):
                prog.classes.append(self._parse_class())
            else:
                raise ParseError(self._peek(), "expected 'fn' or 'class' at top level")
            self._skip_newlines()
        return prog

    # --- Function ---

    def _parse_fn(self) -> FnDecl:
        tok = self._eat(TT.FN)
        name_tok = self._eat(TT.IDENT)
        self._eat(TT.LPAREN)
        params = self._parse_params()
        self._eat(TT.RPAREN)

        ret_type = "Void"
        if self._at(TT.ARROW):
            self._eat(TT.ARROW)
            ret_type = self._eat(TT.IDENT).value

        self._eat(TT.COLON)
        self._eat(TT.NEWLINE)
        body = self._parse_block()

        return FnDecl(name=name_tok.value, params=params, ret_type=ret_type,
                      body=body, line=tok.line)

    def _parse_params(self) -> list[Param]:
        params = []
        if self._at(TT.RPAREN):
            return params
        params.append(self._parse_param())
        while self._at(TT.COMMA):
            self._eat(TT.COMMA)
            params.append(self._parse_param())
        return params

    def _parse_param(self) -> Param:
        # Handle 'self' as a parameter
        if self._at(TT.SELF):
            self._eat(TT.SELF)
            return Param(name="self", type_name="Self")
        name = self._eat(TT.IDENT).value
        self._eat(TT.COLON)
        type_name = self._eat(TT.IDENT).value
        return Param(name=name, type_name=type_name)

    # --- Class ---

    def _parse_class(self) -> ClassDecl:
        tok = self._eat(TT.CLASS)
        name = self._eat(TT.IDENT).value
        parent = "Object"
        if self._at(TT.LPAREN):
            self._eat(TT.LPAREN)
            parent = self._eat(TT.IDENT).value
            self._eat(TT.RPAREN)
        self._eat(TT.COLON)
        self._eat(TT.NEWLINE)

        self._eat(TT.INDENT)
        methods = []
        self._skip_newlines()
        while not self._at(TT.DEDENT):
            if self._at(TT.FN):
                methods.append(self._parse_fn())
            else:
                raise ParseError(self._peek(), "expected 'fn' in class body")
            self._skip_newlines()
        self._eat(TT.DEDENT)

        return ClassDecl(name=name, parent=parent, methods=methods, line=tok.line)

    # --- Block ---

    def _parse_block(self) -> list[Stmt]:
        self._eat(TT.INDENT)
        stmts = []
        self._skip_newlines()
        while not self._at(TT.DEDENT):
            stmts.append(self._parse_stmt())
            self._skip_newlines()
        self._eat(TT.DEDENT)
        return stmts

    # --- Statements ---

    def _parse_stmt(self) -> Stmt:
        if self._at(TT.LET):
            return self._parse_let()
        if self._at(TT.RETURN):
            return self._parse_return()
        if self._at(TT.IF):
            return self._parse_if()
        if self._at(TT.WHILE):
            return self._parse_while()

        # Expression statement, or assignment
        line = self._peek().line
        expr = self._parse_expr()

        if self._at(TT.EQ):
            self._eat(TT.EQ)
            value = self._parse_expr()
            self._eat(TT.NEWLINE)
            return AssignStmt(target=expr, value=value, line=line)

        self._eat(TT.NEWLINE)
        return ExprStmt(expr=expr, line=line)

    def _parse_let(self) -> LetStmt:
        tok = self._eat(TT.LET)
        name = self._eat(TT.IDENT).value
        type_name = None
        if self._at(TT.COLON):
            self._eat(TT.COLON)
            type_name = self._eat(TT.IDENT).value
        self._eat(TT.EQ)
        value = self._parse_expr()
        self._eat(TT.NEWLINE)
        return LetStmt(name=name, type_name=type_name, value=value, line=tok.line)

    def _parse_return(self) -> ReturnStmt:
        tok = self._eat(TT.RETURN)
        value = None
        if not self._at(TT.NEWLINE):
            value = self._parse_expr()
        self._eat(TT.NEWLINE)
        return ReturnStmt(value=value, line=tok.line)

    def _parse_if(self) -> IfStmt:
        tok = self._eat(TT.IF)
        cond = self._parse_expr()
        self._eat(TT.COLON)
        self._eat(TT.NEWLINE)
        body = self._parse_block()

        elif_clauses = []
        self._skip_newlines()
        while self._at(TT.ELIF):
            self._eat(TT.ELIF)
            elif_cond = self._parse_expr()
            self._eat(TT.COLON)
            self._eat(TT.NEWLINE)
            elif_body = self._parse_block()
            elif_clauses.append((elif_cond, elif_body))
            self._skip_newlines()

        else_body = []
        if self._at(TT.ELSE):
            self._eat(TT.ELSE)
            self._eat(TT.COLON)
            self._eat(TT.NEWLINE)
            else_body = self._parse_block()

        return IfStmt(condition=cond, body=body, elif_clauses=elif_clauses,
                      else_body=else_body, line=tok.line)

    def _parse_while(self) -> WhileStmt:
        tok = self._eat(TT.WHILE)
        cond = self._parse_expr()
        self._eat(TT.COLON)
        self._eat(TT.NEWLINE)
        body = self._parse_block()
        return WhileStmt(condition=cond, body=body, line=tok.line)

    # --- Expressions (precedence climbing) ---

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._at(TT.OR):
            self._eat(TT.OR)
            right = self._parse_and()
            left = BinOp(op="or", left=left, right=right, line=left.line)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self._at(TT.AND):
            self._eat(TT.AND)
            right = self._parse_not()
            left = BinOp(op="and", left=left, right=right, line=left.line)
        return left

    def _parse_not(self) -> Expr:
        if self._at(TT.NOT):
            tok = self._eat(TT.NOT)
            operand = self._parse_not()
            return UnaryOp(op="not", operand=operand, line=tok.line)
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        left = self._parse_add()
        if self._at(TT.EQEQ, TT.NEQ, TT.LT, TT.GT, TT.LE, TT.GE):
            tok = self._peek()
            self.pos += 1
            right = self._parse_add()
            left = BinOp(op=tok.value, left=left, right=right, line=left.line)
        return left

    def _parse_add(self) -> Expr:
        left = self._parse_mul()
        while self._at(TT.PLUS, TT.MINUS):
            tok = self._peek()
            self.pos += 1
            right = self._parse_mul()
            left = BinOp(op=tok.value, left=left, right=right, line=left.line)
        return left

    def _parse_mul(self) -> Expr:
        left = self._parse_unary()
        while self._at(TT.STAR, TT.SLASH, TT.PERCENT):
            tok = self._peek()
            self.pos += 1
            right = self._parse_unary()
            left = BinOp(op=tok.value, left=left, right=right, line=left.line)
        return left

    def _parse_unary(self) -> Expr:
        if self._at(TT.MINUS):
            tok = self._eat(TT.MINUS)
            operand = self._parse_unary()
            return UnaryOp(op="-", operand=operand, line=tok.line)
        return self._parse_postfix()

    def _parse_postfix(self) -> Expr:
        expr = self._parse_primary()
        while True:
            if self._at(TT.DOT):
                self._eat(TT.DOT)
                name = self._eat(TT.IDENT).value
                if self._at(TT.LPAREN):
                    self._eat(TT.LPAREN)
                    args = self._parse_args()
                    self._eat(TT.RPAREN)
                    expr = MethodCall(obj=expr, method=name, args=args, line=expr.line)
                else:
                    expr = FieldAccess(obj=expr, field_name=name, line=expr.line)
            elif self._at(TT.LPAREN) and isinstance(expr, Var):
                # Function call: name(args)
                self._eat(TT.LPAREN)
                args = self._parse_args()
                self._eat(TT.RPAREN)
                expr = Call(func=expr.name, args=args, line=expr.line)
            else:
                break
        return expr

    def _parse_args(self) -> list[Expr]:
        args = []
        if self._at(TT.RPAREN):
            return args
        args.append(self._parse_expr())
        while self._at(TT.COMMA):
            self._eat(TT.COMMA)
            args.append(self._parse_expr())
        return args

    def _parse_primary(self) -> Expr:
        tok = self._peek()

        if tok.type == TT.INT_LIT:
            self.pos += 1
            return IntLit(value=tok.value, line=tok.line)

        if tok.type == TT.FLOAT_LIT:
            self.pos += 1
            value, is_float32 = tok.value
            return FloatLit(value=value, is_float32=is_float32, line=tok.line)

        if tok.type == TT.STR_LIT:
            self.pos += 1
            return StrLit(value=tok.value, line=tok.line)

        if tok.type == TT.BOOL_LIT:
            self.pos += 1
            return BoolLit(value=tok.value, line=tok.line)

        if tok.type == TT.NONE:
            self.pos += 1
            return NoneLit(line=tok.line)

        if tok.type == TT.SELF:
            self.pos += 1
            return Var(name="self", line=tok.line)

        if tok.type == TT.SUPER:
            self.pos += 1
            return Var(name="super", line=tok.line)

        if tok.type == TT.IDENT:
            self.pos += 1
            return Var(name=tok.value, line=tok.line)

        if tok.type == TT.LPAREN:
            self._eat(TT.LPAREN)
            expr = self._parse_expr()
            self._eat(TT.RPAREN)
            return expr

        raise ParseError(tok, f"unexpected token: {tok.type.name}")
