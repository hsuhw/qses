from enum import Enum, unique, auto
from functools import reduce
from typing import Dict, List, Union, Optional

from lenc import LengthConstraint, IntExpression
from regc import RegularConstraint
from we import WordEquation, StrVariable, StrExpression


@unique
class ValueType(Enum):
    bool = auto()
    int = auto()
    string = auto()
    unknown = auto()


@unique
class Connective(Enum):
    logic_and = auto()
    logic_or = auto()


YetTyped = str
Part = Union[StrExpression, IntExpression, YetTyped]
Literal = Union[WordEquation, LengthConstraint, RegularConstraint]


class Term:
    def __init__(self, items: List[Union['Term', Literal]],
                 conn: Optional[Connective] = None):
        assert conn or (len(items) == 1 and not isinstance(items[0], Term))
        self.items: List[Union['Term', Literal]] = items
        self.connective: Optional[Connective] = conn

    def __repr__(self):
        ts = ' '.join(map(str, self.items))
        return f'({self.connective.name} {ts})' if self.connective else ts

    def is_clause(self):
        return reduce((lambda acc, i: acc and not isinstance(i, Term)),
                      self.items, True)

    def negate(self) -> 'Term':
        assert len(self.items) == 1
        return self.__class__([self.items[0].negate()])

    def normalize(self):
        # TODO: only correctly handle conjunction
        if not self.connective:
            return self
        result = []
        for i in self.items:
            if not isinstance(i, Term):
                result.append(i)
            elif not i.connective:
                result.append(i.items[0])
            else:
                result.append(i)
        return Term(result, self.connective)


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
        self.word_equations: List[WordEquation] = []
        self.reg_constraints: List[RegularConstraint] = []
        self.len_constraints: List[LengthConstraint] = []

    def declare_variable(self, name: str, typ: ValueType):
        if name in self.variables:
            raise MultiDeclarationError(f'variable: {name}')
        self.variables[name] = typ
        if typ is ValueType.string:
            length_var_name = StrVariable(name).length().value
            self.declare_variable(length_var_name, ValueType.int)

    def new_variable(self, typ: ValueType) -> str:
        name = f'xx_{typ.name}{self.internal_var_count}_'
        self.variables[name] = typ
        self.internal_var_count += 1
        return name

    def ensure_variable_known(self, var: str, typ: ValueType):
        if self.variables.get(var) is not typ:
            raise UnknownVariableError(f'variable: {var} type: {typ.name}')

    def add_word_equation(self, we: WordEquation):
        for var in we.variables():
            self.ensure_variable_known(var.value, ValueType.string)
        self.word_equations.append(we)

    def add_regular_constraint(self, cons: RegularConstraint):
        self.ensure_variable_known(cons.tgt_var.value, ValueType.string)
        self.reg_constraints.append(cons)

    def add_length_constraint(self, cons: LengthConstraint):
        for var in cons.variables():
            self.ensure_variable_known(var.value, ValueType.int)
        self.len_constraints.append(cons)
