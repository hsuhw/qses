import antlr4

import lenc
import prob
import we

from typing import List, Tuple, Union, Optional

from generated.SMTLIB26Lexer import SMTLIB26Lexer
from generated.SMTLIB26Parser import SMTLIB26Parser
from generated.SMTLIB26ParserListener import SMTLIB26ParserListener
from prob import Problem, ValueType
from we import WordEquation
from lenc import LengthConstraint

TermContext = SMTLIB26Parser.TermContext


def term_operator(term: TermContext) -> Optional[str]:
    try:
        symbol = term.qual_identifier().identifier().symbol()
        return symbol.SIMPLE_SYMBOL().getText()
    except AttributeError:
        return None


class Syntax:
    @classmethod
    def is_disjunction(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'or'

    @classmethod
    def is_conjunction(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'and'

    @classmethod
    def is_negation(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'not'

    @classmethod
    def is_equality(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == '='

    @classmethod
    def is_minus(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == '-'

    @classmethod
    def is_plus(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == '+'

    @classmethod
    def is_string_concat(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_string_length(cls, term: SMTLIB26Parser.TermContext):
        pass


class Z3Str2Syntax(Syntax):
    @classmethod
    def is_string_concat(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'Concat'

    @classmethod
    def is_string_length(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'Length'


class Z3Str3Syntax(Syntax):
    @classmethod
    def is_string_concat(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'str.++'

    @classmethod
    def is_string_length(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == 'str.len'


Z3STR2_SYNTAX = Z3Str2Syntax()
Z3STR3_SYNTAX = Z3Str3Syntax()

YetKnown = str
BuiltTerm = Union[List['BuiltTerm'], we.Expression, lenc.Expression, YetKnown]
Operands = List[BuiltTerm]
TypedBuiltTerm = Tuple[BuiltTerm, ValueType]
TypedOperands = List[TypedBuiltTerm]


def string_to_characters(string: str) -> List[we.Character]:
    return list(map(lambda ch: we.Character(ch), [*string]))


def without_type(operands: TypedOperands) -> Operands:
    return [operand for operand, typ in operands]


class BasicProblemBuilder(SMTLIB26ParserListener):
    def __init__(self, src_path: str, syntax: Syntax):
        self.src_path: str = src_path
        self.syntax: Syntax = syntax
        self.problem: Problem = Problem()

    def src_pos(self, ctx: antlr4.ParserRuleContext):
        return f'at "{self.src_path}" {ctx.start.line}:{ctx.start.column}'

    def ensure_operands_same_type(self, term: TermContext,
                                  operands: TypedOperands) -> ValueType:
        tgt_type = ValueType.unknown
        for index, (_, typ) in enumerate(operands):
            if typ is not ValueType.unknown:
                if tgt_type is ValueType.unknown:
                    tgt_type = typ
                elif typ is not tgt_type:
                    raise prob.InvalidTypeError(self.src_pos(term.term(index)))
        return tgt_type

    def set_operands_type(self, term: TermContext, operands: TypedOperands,
                          tgt_type: ValueType) -> Operands:
        assert tgt_type is ValueType.int or tgt_type is ValueType.string
        result = []
        for index, (built_term, typ) in enumerate(operands):
            if typ is ValueType.unknown:
                assert isinstance(built_term, YetKnown)
                self.problem.declare_variable(built_term, tgt_type)
                if tgt_type is ValueType.int:
                    result.append(([lenc.Variable(built_term)], tgt_type))
                elif tgt_type is ValueType.string:
                    result.append(([we.Variable(built_term)], tgt_type))
            elif typ is not tgt_type:
                raise prob.InvalidTypeError(self.src_pos(term.term(index)))
            else:
                result.append((built_term, tgt_type))
        return without_type(result)

    def declare_variable(self, symbol: SMTLIB26Parser.SymbolContext,
                         sort: SMTLIB26Parser.SortContext):
        name = symbol.SIMPLE_SYMBOL().getText()
        typ = sort.identifier().getText().lower()
        self.problem.declare_variable(name, ValueType[typ])

    def handle_atomic_term(self, term: TermContext) -> TypedBuiltTerm:
        if term.spec_constant():
            if term.spec_constant().NUMERAL():
                num = int(term.spec_constant().NUMERAL().getText())
                return [lenc.Constant(num)], ValueType.int
            elif term.spec_constant().STRING():
                string = term.spec_constant().STRING().getText()[1:-1]
                chars = string_to_characters(string)
                return chars, ValueType.string
            # other types of `spec_constant` not handled
        elif term.qual_identifier():
            assert len(term.term()) == 0
            symbol = term.qual_identifier().identifier().symbol()
            name = symbol.SIMPLE_SYMBOL().getText()
            known_type = self.problem.variables.get(name)
            if known_type is ValueType.int:
                return [lenc.Variable(name)], ValueType.int
            elif known_type is ValueType.string:
                return [we.Variable(name)], ValueType.string
            elif not known_type:
                return name, ValueType.unknown
            # `ValueType.bool` not handled
        raise prob.UnsupportedConstructError(self.src_pos(term))

    def handle_conjunction(self, term: TermContext,
                           operands: TypedOperands) -> TypedBuiltTerm:
        """ Because the way how we handle equalities, conjunction terms
            cannot be negated right now.
        """
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.bool)
        # TODO: the construction of Boolean formula is not handled
        # currently we do assertions right in the equality handler
        return ops, ValueType.bool  # TODO: lack operator info

    def handle_string_equality(self, term: TermContext,
                               operands: TypedOperands) -> TypedBuiltTerm:
        """ Equality terms cannot be negated right now; they currently cause
            the assertions right away.
        """
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.string)
        for index in range(1, len(ops)):
            constraint = WordEquation(ops[index - 1], ops[index])
            self.problem.add_word_equation(constraint)  # add assertion
        return ops, ValueType.bool  # TODO: lack operator info

    def handle_string_concat(self, term: TermContext,
                             operands: TypedOperands) -> TypedBuiltTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.string)
        return [e for str_exp in ops for e in str_exp], ValueType.string

    def handle_string_length(self, term: TermContext,
                             operands: TypedOperands) -> TypedBuiltTerm:
        assert len(operands) == 1
        if len(operands[0][0]) != 1:
            raise prob.InvalidConstructError()
        ops: List[we.Expression] = self.set_operands_type(term, operands,
                                                          ValueType.string)
        return [e.length() for str_exp in ops for e in str_exp], ValueType.int

    def handle_int_equality(self, term: TermContext,
                            operands: TypedOperands) -> TypedBuiltTerm:
        """ Equality terms cannot be negated right now; they currently cause
            the assertions right away.
        """
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.int)
        for index in range(1, len(ops)):
            constraint = LengthConstraint(ops[index - 1], ops[index])
            self.problem.add_length_constraint(constraint)  # add assertion
        return ops, ValueType.bool  # TODO: lack operator info

    def handle_int_opposite(self, term: SMTLIB26Parser.TermContext,
                            operands: TypedOperands) -> TypedBuiltTerm:
        assert len(operands) == 1
        op: lenc.Expression = self.set_operands_type(term, operands,
                                                     ValueType.int)[0]
        return [e.opposite() for e in op], ValueType.int

    def handle_int_plus(self, term: SMTLIB26Parser.TermContext,
                        operands: TypedOperands) -> TypedBuiltTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.int)
        return [e for len_exp in ops for e in len_exp], ValueType.int

    def handle_int_minus(self, term: SMTLIB26Parser.TermContext,
                         operands: TypedOperands) -> TypedBuiltTerm:
        assert len(operands) == 2
        ops: List[lenc.Expression] = self.set_operands_type(term, operands,
                                                            ValueType.int)
        result = [e for e in ops[0]] + [e.opposite() for e in ops[1]]
        return result, ValueType.int

    def handle_term(self, term: TermContext) -> TypedBuiltTerm:
        if not term.OPEN_PAR():
            return self.handle_atomic_term(term)
        elif term.qual_identifier():
            if self.syntax.is_disjunction(term):
                return [], ValueType.bool  # not handled
            operands = self.handle_terms(term.term())
            operand_num = len(operands)
            if operand_num == 1:
                if self.syntax.is_minus(term):
                    return self.handle_int_opposite(term, operands)
                elif self.syntax.is_string_length(term):
                    return self.handle_string_length(term, operands)
            elif operand_num > 1:
                if self.syntax.is_conjunction(term):
                    return self.handle_conjunction(term, operands)
                elif self.syntax.is_equality(term):
                    typ = self.ensure_operands_same_type(term, operands)
                    if typ is ValueType.int:
                        return self.handle_int_equality(term, operands)
                    elif typ is ValueType.string:
                        return self.handle_string_equality(term, operands)
                elif self.syntax.is_plus(term):
                    return self.handle_int_plus(term, operands)
                elif self.syntax.is_minus(term) and operand_num == 2:
                    return self.handle_int_minus(term, operands)
                elif self.syntax.is_string_concat(term):
                    return self.handle_string_concat(term, operands)
        raise prob.UnsupportedConstructError(self.src_pos(term))

    def handle_terms(self, terms: List[TermContext]) -> TypedOperands:
        return list(map(lambda t: self.handle_term(t), terms))

    def enterCommand(self, ctx: SMTLIB26Parser.CommandContext):
        if ctx.TOKEN_CMD_DECLARE_FUN():
            self.declare_variable(ctx.symbol(0), ctx.sort(0))
        elif ctx.TOKEN_CMD_ASSERT():
            self.handle_term(ctx.term(0))


def parse_file(file_path: str, syntax: Syntax = Z3STR3_SYNTAX):
    lexer = SMTLIB26Lexer(antlr4.FileStream(file_path))
    parser = SMTLIB26Parser(antlr4.CommonTokenStream(lexer))
    ast_script_part = parser.script()
    builder = BasicProblemBuilder(file_path, syntax)
    walker = antlr4.ParseTreeWalker()
    walker.walk(builder, ast_script_part)
    return builder.problem
