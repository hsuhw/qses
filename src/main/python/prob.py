from enum import Enum, unique, auto
from functools import reduce
from typing import Dict, List, Union

from fsa import Alphabet
from lenc import LengthConstraint, IntExpression
from regc import RegularConstraint, RegExpression
from tok import INTERNAL_VAR_PREFIX
from we import WordEquation, StrVariable, StrExpression


@unique
class ValueType(Enum):
    bool = auto()
    int = auto()
    string = auto()
    regex = auto()
    unknown = auto()


@unique
class Connective(Enum):
    logic_and = auto()
    logic_or = auto()


YetTyped = str
Part = Union[StrExpression, IntExpression, RegExpression, YetTyped]
Literal = Union[WordEquation, LengthConstraint, RegularConstraint]


class Term:
    def __init__(self, items: List[Union['Term', Literal]], conn: Connective):
        assert len(items) != 1
        self.items: List[Union['Term', Literal]] = items
        self.connective: Connective = conn

    def __repr__(self):
        return f'({self.connective.name} {" ".join(map(str, self.items))})'

    def is_clause(self):
        return reduce((lambda acc, i: acc and not isinstance(i, Term)),
                      self.items, True)

    def negate(self):
        pass  # TODO


class MultiDeclarationError(Exception):
    pass


class UnknownVariableError(Exception):
    pass


class UnsupportedConstructError(Exception):
    pass


class InvalidTypeError(Exception):
    pass


class InvalidConstructError(Exception):
    pass


class Problem:
    def __init__(self):
        self.variables: Dict[str, ValueType] = {}
        self.internal_var_count: int = 0
        self.alphabet = Alphabet()
        self.word_equations: List[WordEquation] = []
        self.we_inequality_memo: List[bool] = []
        self.reg_constraints: Dict[str, List[RegularConstraint]] = {}
        self.len_constraints: List[LengthConstraint] = []

    def declare_variable(self, name: str, typ: ValueType):
        if name in self.variables:
            raise MultiDeclarationError(f'variable: {name}')
        self.variables[name] = typ
        if typ is ValueType.string:
            length_var_name = StrVariable(name).length().value
            self.declare_variable(length_var_name, ValueType.int)

    def new_variable(self, typ: ValueType, note: str = None) -> str:
        n = f'_{note}' if note else ''
        name = f'{INTERNAL_VAR_PREFIX}{typ.name}{self.internal_var_count}{n}'
        self.declare_variable(name, typ)
        self.internal_var_count += 1
        return name

    def ensure_variable_known(self, var: str, typ: ValueType):
        if self.variables.get(var) is not typ:
            raise UnknownVariableError(f'variable: {var} type: {typ.name}')

    def add_word_equation(self, we: WordEquation):
        for var in we.variables():
            self.ensure_variable_known(var.value, ValueType.string)
        self.word_equations.append(we)
        self.we_inequality_memo.append(we.negation)

    def merge_all_word_equations(self):
        self.word_equations = [reduce(lambda x, y: x.merge(y),
                                      self.word_equations)]

    def add_regular_constraint(self, cons: RegularConstraint):
        assert cons.fsa.alphabet == self.alphabet
        self.ensure_variable_known(cons.str_var, ValueType.string)
        if cons.str_var in self.reg_constraints:
            self.reg_constraints[cons.str_var].append(cons)
        else:
            self.reg_constraints[cons.str_var] = [cons]

    def add_length_constraint(self, cons: LengthConstraint):
        for var in cons.variables():
            self.ensure_variable_known(var.value, ValueType.int)
        self.len_constraints.append(cons)
