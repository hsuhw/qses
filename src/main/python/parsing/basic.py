import antlr4

import fsa
import lenc
import prob
import tok

from functools import reduce
from typing import List, Tuple, Union, Optional

from generated.SMTLIB26Lexer import SMTLIB26Lexer
from generated.SMTLIB26Parser import SMTLIB26Parser
from generated.SMTLIB26ParserListener import SMTLIB26ParserListener
from lenc import LengthConstraint, IntConstant, IntVariable, IntExpression, \
    Relation
from prob import Problem, Part, Literal, Term, ValueType, Connective, YetTyped
from regc import RegularConstraint, RegExpression
from we import WordEquation, Character, StrVariable, StrExpression

TermContext = SMTLIB26Parser.TermContext


def term_operator(term: TermContext) -> Optional[str]:
    try:
        symbol = term.qual_identifier().identifier().symbol()
        return symbol.SIMPLE_SYMBOL().getText()
    except AttributeError:
        return None


class Syntax:
    @classmethod
    def is_conjunction(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_AND

    @classmethod
    def is_disjunction(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_OR

    @classmethod
    def is_negation(cls, term: SMTLIB26Parser.TermContext):
        try:
            symbol = term.qual_identifier().identifier().symbol()
            return symbol.SYMBOL_NOT()
        except AttributeError:
            return False

    @classmethod
    def is_equality(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_EQ

    @classmethod
    def is_greater(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_GT

    @classmethod
    def is_greater_equal(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_GEQ

    @classmethod
    def is_less(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_LT

    @classmethod
    def is_less_equal(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_LEQ

    @classmethod
    def is_plus(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_PLUS

    @classmethod
    def is_minus(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_MINUS

    @classmethod
    def is_times(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_TIMES

    @classmethod
    def is_string_concat(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_string_length(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_string_contains(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_regex_membership(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_regex_from_string(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_regex_concat(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_regex_union(cls, term: SMTLIB26Parser.TermContext):
        pass

    @classmethod
    def is_regex_closure(cls, term: SMTLIB26Parser.TermContext):
        pass


class Z3Str2Syntax(Syntax):
    @classmethod
    def is_string_concat(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_CONCAT_V1

    @classmethod
    def is_string_length(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_LENGTH_V1

    @classmethod
    def is_string_contains(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_CONTAINS_V1

    @classmethod
    def is_regex_membership(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_IN_RE_V1

    @classmethod
    def is_regex_from_string(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_FROM_STR_V1

    @classmethod
    def is_regex_concat(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_CONCAT_V1

    @classmethod
    def is_regex_union(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_UNION_V1

    @classmethod
    def is_regex_closure(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_CLOSURE_V1


class Z3Str3Syntax(Syntax):
    @classmethod
    def is_string_concat(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_CONCAT_V2

    @classmethod
    def is_string_length(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_LENGTH_V2

    @classmethod
    def is_string_contains(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_CONTAINS_V2

    @classmethod
    def is_regex_membership(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_STR_IN_RE_V2

    @classmethod
    def is_regex_from_string(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_FROM_STR_V2

    @classmethod
    def is_regex_concat(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_CONCAT_V2

    @classmethod
    def is_regex_union(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_UNION_V2

    @classmethod
    def is_regex_closure(cls, term: SMTLIB26Parser.TermContext):
        return term_operator(term) == tok.STR_THEORY_RE_CLOSURE_V2


Z3STR2_SYNTAX = Z3Str2Syntax()
Z3STR3_SYNTAX = Z3Str3Syntax()
SMTLIBTerm = Union[Part, Literal, Term]
TypedSMTLIBTerm = Tuple[SMTLIBTerm, ValueType]
Operands = List[SMTLIBTerm]
TypedOperands = List[TypedSMTLIBTerm]


def string_to_characters(string: str) -> List[Character]:
    return list(map(lambda ch: Character(ch), [*string]))


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
        assert tgt_type is not ValueType.unknown
        result = []
        for index, (operand, typ) in enumerate(operands):
            if typ is ValueType.unknown:
                assert isinstance(operand, YetTyped)
                self.problem.declare_variable(operand, tgt_type)
                if tgt_type is ValueType.int:
                    result.append(([IntVariable(operand)], tgt_type))
                elif tgt_type is ValueType.string:
                    result.append(([StrVariable(operand)], tgt_type))
                elif tgt_type is ValueType.bool:
                    pos = self.src_pos(term.term(index))
                    raise prob.UnsupportedConstructError(pos)
                elif tgt_type is ValueType.regex:
                    pos = self.src_pos(term.term(index))
                    raise prob.InvalidConstructError(pos)
            elif typ is not tgt_type:
                raise prob.InvalidTypeError(self.src_pos(term.term(index)))
            else:
                result.append((operand, tgt_type))
        return without_type(result)

    def declare_variable(self, symbol: SMTLIB26Parser.SymbolContext,
                         sort: SMTLIB26Parser.SortContext):
        name = symbol.SIMPLE_SYMBOL().getText()
        typ = sort.identifier().getText().lower()
        self.problem.declare_variable(name, ValueType[typ])

    def handle_base_case(self, term: TermContext) -> Tuple[Part, ValueType]:
        if term.spec_constant():
            if term.spec_constant().NUMERAL():
                num = int(term.spec_constant().NUMERAL().getText())
                return [IntConstant(num)], ValueType.int
            elif term.spec_constant().STRING():
                string = term.spec_constant().STRING().getText()[1:-1]
                chars = string_to_characters(string)
                map(lambda x: self.problem.alphabet.take(x.value), chars)
                return chars, ValueType.string
            # other types of `spec_constant` not handled
        elif term.qual_identifier():
            assert len(term.term()) == 0
            symbol = term.qual_identifier().identifier().symbol()
            name = symbol.SIMPLE_SYMBOL().getText()
            known_type = self.problem.variables.get(name)
            if known_type is ValueType.int:
                return [IntVariable(name)], ValueType.int
            elif known_type is ValueType.string:
                return [StrVariable(name)], ValueType.string
            elif known_type is ValueType.bool:
                return None, ValueType.bool  # neglecting
            elif not known_type:
                return name, ValueType.unknown
            # `ValueType.bool` not handled
        raise prob.UnsupportedConstructError(self.src_pos(term))

    def handle_conjunction(self, term: TermContext,
                           operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.bool)
        terms = []
        for op in ops:
            if isinstance(op, Term) and op.connective is Connective.logic_and:
                terms += op.items
            else:
                terms.append(op)
        return Term(terms, Connective.logic_and), ValueType.bool

    def handle_negation(self, term: TermContext,
                        operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 1
        t: Term = self.set_operands_type(term, operands, ValueType.bool)[0]
        return t.negate() if t else t, ValueType.bool

    def handle_string_equality(self, term: TermContext,
                               operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.string)
        literals = []
        for index in range(1, len(ops)):
            literals.append(WordEquation(ops[index - 1], ops[index]))
        if len(literals) == 1:
            return literals[0], ValueType.bool
        return Term(literals, Connective.logic_and), ValueType.bool

    def handle_string_concat(self, term: TermContext,
                             operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.string)
        return [e for str_exp in ops for e in str_exp], ValueType.string

    def handle_string_length(self, term: TermContext,
                             operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 1
        se: StrExpression = self.set_operands_type(term, operands,
                                                   ValueType.string)[0]
        return [e.length() for e in se], ValueType.int

    def handle_int_equality(self, term: TermContext,
                            operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.int)
        literals = []
        for index in range(1, len(ops)):
            literals.append(LengthConstraint(ops[index - 1], ops[index]))
        if len(literals) == 1:
            return literals[0], ValueType.bool
        return Term(literals, Connective.logic_and), ValueType.bool

    def handle_int_inequality(self, term: TermContext, operands: TypedOperands,
                              rel: Relation) -> TypedSMTLIBTerm:
        assert len(operands) == 2
        ops: List[IntExpression] = self.set_operands_type(term, operands,
                                                          ValueType.int)
        return LengthConstraint(ops[0], ops[1], rel), ValueType.bool

    def handle_int_opposite(self, term: SMTLIB26Parser.TermContext,
                            operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 1
        ie: IntExpression = self.set_operands_type(term, operands,
                                                   ValueType.int)[0]
        return [e.opposite() for e in ie], ValueType.int

    def handle_int_plus(self, term: SMTLIB26Parser.TermContext,
                        operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) > 1
        ops = self.set_operands_type(term, operands, ValueType.int)
        return [e for len_exp in ops for e in len_exp], ValueType.int

    def handle_int_minus(self, term: SMTLIB26Parser.TermContext,
                         operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 2
        ops: List[IntExpression] = self.set_operands_type(term, operands,
                                                          ValueType.int)
        result = [e for e in ops[0]] + [e.opposite() for e in ops[1]]
        return result, ValueType.int

    def handle_int_times(self, term: SMTLIB26Parser.TermContext,
                         operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 2
        ops: List[IntExpression] = self.set_operands_type(term, operands,
                                                          ValueType.int)
        op1: IntExpression = lenc.reduce_in_arithmetic(ops[0])
        op2: IntExpression = lenc.reduce_in_arithmetic(ops[1])
        if lenc.is_const_expr(op1):
            return [e.multiply(op1[0].value) for e in op2], ValueType.int
        elif lenc.is_const_expr(op2):
            return [e.multiply(op2[0].value) for e in op1], ValueType.int
        else:
            raise prob.InvalidConstructError(self.src_pos(term.term(1)))

    def handle_regex_from_string(self, term: SMTLIB26Parser.TermContext) \
            -> TypedSMTLIBTerm:
        assert len(term.term()) == 1
        op = term.term(0)
        if op.spec_constant() and op.spec_constant().STRING():
            string = op.spec_constant().STRING().getText()[1:-1]
            return fsa.from_str(string, self.problem.alphabet), ValueType.regex
        raise prob.UnsupportedConstructError(self.src_pos(op))

    def handle_regex_concat(self, term: SMTLIB26Parser.TermContext,
                            operands: TypedOperands) -> TypedSMTLIBTerm:
        ops: List[RegExpression] = self.set_operands_type(term, operands,
                                                          ValueType.regex)
        return reduce(lambda x, y: x.concat(y), ops), ValueType.regex

    def handle_regex_union(self, term: SMTLIB26Parser.TermContext,
                           operands: TypedOperands) -> TypedSMTLIBTerm:
        ops: List[RegExpression] = self.set_operands_type(term, operands,
                                                          ValueType.regex)
        return reduce(lambda x, y: x.union(y), ops), ValueType.regex

    def handle_regex_closure(self, term: SMTLIB26Parser.TermContext,
                             operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 1
        op: RegExpression = self.set_operands_type(term, operands,
                                                   ValueType.regex)[0]
        return op.closure(), ValueType.regex

    def handle_regex_membership(self, term: SMTLIB26Parser.TermContext,
                                operands: TypedOperands) -> TypedSMTLIBTerm:
        assert len(operands) == 2
        [(op1, typ1), (op2, typ2), *_] = operands
        if (typ1 is not ValueType.string
                or len(op1) != 1 or not isinstance(op1[0], StrVariable)):
            raise prob.InvalidConstructError(self.src_pos(term.term(0)))
        if typ2 is not ValueType.regex:
            raise prob.InvalidConstructError(self.src_pos(term.term(1)))
        dfa = op2.determinize().minimize()
        return RegularConstraint(op1[0].value, dfa), ValueType.bool

    def handle_term(self, term: TermContext) -> TypedSMTLIBTerm:
        if not term.OPEN_PAR():
            return self.handle_base_case(term)
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
                elif self.syntax.is_negation(term):
                    return self.handle_negation(term, operands)
                elif self.syntax.is_regex_from_string(term):
                    return self.handle_regex_from_string(term)
                elif self.syntax.is_regex_closure(term):
                    return self.handle_regex_closure(term, operands)
            elif operand_num > 1:
                if self.syntax.is_conjunction(term):
                    return self.handle_conjunction(term, operands)
                elif self.syntax.is_equality(term):
                    typ = self.ensure_operands_same_type(term, operands)
                    if typ is ValueType.int:
                        return self.handle_int_equality(term, operands)
                    elif typ is ValueType.string:
                        return self.handle_string_equality(term, operands)
                elif self.syntax.is_greater(term) and operand_num == 2:
                    return self.handle_int_inequality(term, operands,
                                                      Relation.greater)
                elif self.syntax.is_greater_equal(term) and operand_num == 2:
                    return self.handle_int_inequality(term, operands,
                                                      Relation.greater_equal)
                elif self.syntax.is_less(term) and operand_num == 2:
                    return self.handle_int_inequality(term, operands,
                                                      Relation.less)
                elif self.syntax.is_less_equal(term) and operand_num == 2:
                    return self.handle_int_inequality(term, operands,
                                                      Relation.greater_equal)
                elif self.syntax.is_plus(term):
                    return self.handle_int_plus(term, operands)
                elif self.syntax.is_minus(term) and operand_num == 2:
                    return self.handle_int_minus(term, operands)
                elif self.syntax.is_times(term) and operand_num == 2:
                    return self.handle_int_times(term, operands)
                elif self.syntax.is_string_concat(term):
                    return self.handle_string_concat(term, operands)
                elif self.syntax.is_regex_concat(term):
                    return self.handle_regex_concat(term, operands)
                elif self.syntax.is_regex_union(term):
                    return self.handle_regex_union(term, operands)
                elif self.syntax.is_regex_membership(term) and operand_num == 2:
                    return self.handle_regex_membership(term, operands)
        raise prob.UnsupportedConstructError(self.src_pos(term))

    def handle_terms(self, terms: List[TermContext]) -> TypedOperands:
        return list(map(lambda t: self.handle_term(t), terms))

    def assert_literal(self, item: Literal, ctx: SMTLIB26Parser.TermContext):
        if not item:
            return  # neglecting
        if isinstance(item, LengthConstraint):
            self.problem.add_length_constraint(item)
        elif isinstance(item, WordEquation):
            self.problem.add_word_equation(item)
        elif isinstance(item, RegularConstraint):
            self.problem.add_regular_constraint(item)
        else:
            raise prob.InvalidConstructError(self.src_pos(ctx))

    def assert_term(self, ctx: SMTLIB26Parser.TermContext):
        term, typ = self.handle_term(ctx)
        if typ is not ValueType.bool:
            raise prob.InvalidConstructError(self.src_pos(ctx))
        if not isinstance(term, Term):
            self.assert_literal(term, ctx.term(0))
        else:
            for index, item in enumerate(term.items):
                self.assert_literal(item, ctx.term(index))

    def enterCommand(self, ctx: SMTLIB26Parser.CommandContext):
        if ctx.TOKEN_CMD_DECLARE_FUN():
            self.declare_variable(ctx.symbol(0), ctx.sort(0))
        elif ctx.TOKEN_CMD_ASSERT():
            self.assert_term(ctx.term(0))


def parse_file(file_path: str, syntax: Syntax = Z3STR3_SYNTAX):
    lexer = SMTLIB26Lexer(antlr4.FileStream(file_path))
    parser = SMTLIB26Parser(antlr4.CommonTokenStream(lexer))
    ast_script_part = parser.script()
    builder = BasicProblemBuilder(file_path, syntax)
    walker = antlr4.ParseTreeWalker()
    walker.walk(builder, ast_script_part)
    return builder.problem
