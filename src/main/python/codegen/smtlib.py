import lenc
import tok
import we

from typing import TextIO

from lenc import IntExpression, Relation
from prob import Problem, ValueType
from we import StrExpression


class Syntax:
    type_label = {
        ValueType.bool: 'Bool',
        ValueType.string: 'String',
        ValueType.int: 'Int',
    }

    relation_label = {
        Relation.equal: tok.STR_THEORY_EQ,
        Relation.greater: tok.STR_THEORY_GT,
        Relation.greater_equal: tok.STR_THEORY_GEQ,
        Relation.less: tok.STR_THEORY_LT,
        Relation.less_equal: tok.STR_THEORY_LEQ,
    }

    @classmethod
    def conjunction(cls):
        return tok.STR_THEORY_AND

    @classmethod
    def disjunction(cls):
        return tok.STR_THEORY_OR

    @classmethod
    def negation(cls):
        return tok.STR_THEORY_NOT

    @classmethod
    def equality(cls):
        return tok.STR_THEORY_EQ

    @classmethod
    def greater(cls):
        return tok.STR_THEORY_GT

    @classmethod
    def greater_equal(cls):
        return tok.STR_THEORY_GEQ

    @classmethod
    def less(cls):
        return tok.STR_THEORY_LT

    @classmethod
    def less_equal(cls):
        return tok.STR_THEORY_LEQ

    @classmethod
    def plus(cls):
        return tok.STR_THEORY_PLUS

    @classmethod
    def minus(cls):
        return tok.STR_THEORY_MINUS

    @classmethod
    def times(cls):
        return tok.STR_THEORY_TIMES

    @classmethod
    def string_concat(cls):
        pass

    @classmethod
    def string_length(cls):
        pass

    @classmethod
    def string_contains(cls):
        pass

    @classmethod
    def regex_membership(cls):
        pass

    @classmethod
    def regex_from_string(cls):
        pass

    @classmethod
    def regex_concat(cls):
        pass

    @classmethod
    def regex_union(cls):
        pass

    @classmethod
    def regex_closure(cls):
        pass


class Z3Str2Syntax(Syntax):
    @classmethod
    def string_concat(cls):
        return tok.STR_THEORY_STR_CONCAT_V1

    @classmethod
    def string_length(cls):
        return tok.STR_THEORY_STR_LENGTH_V1

    @classmethod
    def string_contains(cls):
        return tok.STR_THEORY_STR_CONTAINS_V1

    @classmethod
    def regex_membership(cls):
        return tok.STR_THEORY_STR_IN_RE_V1

    @classmethod
    def regex_from_string(cls):
        return tok.STR_THEORY_RE_FROM_STR_V1

    @classmethod
    def regex_concat(cls):
        return tok.STR_THEORY_RE_CONCAT_V1

    @classmethod
    def regex_union(cls):
        return tok.STR_THEORY_RE_UNION_V1

    @classmethod
    def regex_closure(cls):
        return tok.STR_THEORY_RE_CLOSURE_V1


class Z3Str3Syntax(Syntax):
    @classmethod
    def string_concat(cls):
        return tok.STR_THEORY_STR_CONCAT_V2

    @classmethod
    def string_length(cls):
        return tok.STR_THEORY_STR_LENGTH_V2

    @classmethod
    def string_contains(cls):
        return tok.STR_THEORY_STR_CONTAINS_V2

    @classmethod
    def regex_membership(cls):
        return tok.STR_THEORY_STR_IN_RE_V2

    @classmethod
    def regex_from_string(cls):
        return tok.STR_THEORY_RE_FROM_STR_V2

    @classmethod
    def regex_concat(cls):
        return tok.STR_THEORY_RE_CONCAT_V2

    @classmethod
    def regex_union(cls):
        return tok.STR_THEORY_RE_UNION_V2

    @classmethod
    def regex_closure(cls):
        return tok.STR_THEORY_RE_CLOSURE_V2


Z3STR2_SYNTAX = Z3Str2Syntax()
Z3STR3_SYNTAX = Z3Str3Syntax()


class SMTLIBLayout:
    def __init__(self, problem: Problem, syntax: Syntax):
        self.syntax: Syntax = syntax
        self.problem: Problem = problem

    def print_variable_declarations(self, dest: TextIO):
        for name, typ in self.problem.variables.items():
            if (typ is ValueType.bool
                    or lenc.internal_len_var_name.match(name)):
                continue
            t = self.syntax.type_label[typ]
            print(f'(declare-fun {name} () {t})', file=dest)

    def render_negative_int(self, num: int) -> str:
        return f'({self.syntax.minus()} {-num})'

    def render_int_expression(self, expr: IntExpression) -> str:
        result = []
        for e in expr:
            if lenc.is_var(e):
                c = e.coefficient
                origin_value = lenc.length_origin_name(e)
                value = e.value if not origin_value \
                    else f'({self.syntax.string_length()} {origin_value})'
                if c == 1:
                    result.append(value)
                elif c > 0:
                    result.append(f'({self.syntax.times()} {c} {value})')
                elif c < 0:
                    c = self.render_negative_int(c)
                    result.append(f'({self.syntax.times()} {c} {value}))')
            elif lenc.is_const(e):
                c = e.value
                if c < 0:
                    result.append(self.render_negative_int(c))
                else:
                    result.append(c)
        if len(result) <= 1:
            return result[0]
        else:
            return f'({self.syntax.plus()} {" ".join(map(str, result))})'

    def print_len_constraints(self, dest: TextIO):
        for lc in self.problem.len_constraints:
            lhs = self.render_int_expression(lc.lhs)
            rhs = self.render_int_expression(lc.rhs)
            if lhs == rhs and (lc.relation is Relation.equal or
                               lc.relation is Relation.greater_equal or
                               lc.relation is Relation.less_equal):
                continue
            if lc.relation is Relation.unequal:
                neg = self.syntax.negation()
                eq = self.syntax.equality()
                c = f'({neg} ({eq} {lhs} {rhs}))'
            else:
                c = f'({self.syntax.relation_label[lc.relation]} {lhs} {rhs})'
            print(f'(assert {c})', file=dest)

    def render_str_expression(self, expr: StrExpression):
        result = []
        curr_str = ''
        in_str = False
        for e in expr:
            if in_str and we.is_var(e):
                in_str = False
                result.append(f'"{curr_str}"')
                curr_str = ''
            if we.is_var(e):
                result.append(e.value)
            if we.is_char(e):
                in_str = True
                curr_str += e.value
        if len(result) <= 1:
            return result[0]
        else:
            return f'({self.syntax.string_concat()} {" ".join(result)})'

    def print_word_equations(self, dest: TextIO):
        wes = self.problem.word_equations
        wes = self.problem.word_equations[0].split() if len(wes) == 1 else wes
        for w in wes:
            lhs = self.render_str_expression(w.lhs)
            rhs = self.render_str_expression(w.rhs)
            if w.negation:
                neg = self.syntax.negation()
                eq = self.syntax.equality()
                c = f'({neg} ({eq} {lhs} {rhs}))'
            else:
                c = f'({self.syntax.equality()} {lhs} {rhs})'
            print(f'(assert {c})', file=dest)

    def print(self, dest: TextIO):
        self.print_variable_declarations(dest)
        self.print_len_constraints(dest)
        self.print_word_equations(dest)
        print(f'(check-sat)', file=dest)


def to_file(prob: Problem, file_path: str, syntax: Syntax = Z3STR3_SYNTAX):
    with open(file_path, 'w') as dest:
        SMTLIBLayout(prob, syntax).print(dest)
