from typing import Dict, List
from enum import Enum, unique, auto
from we import WordEquation
from constr import RegularConstraint, LengthConstraint


@unique
class VariableType(Enum):
    bool = auto()
    int = auto()
    string = auto()


class MultiDeclarationError(Exception):
    pass


class UnknownVariableError(Exception):
    pass


class InvalidProblemError(Exception):
    pass


class Problem:
    def __init__(self):
        self.variables: Dict[str, VariableType] = {}
        self.word_equations: List[WordEquation] = []
        self.reg_constraints: List[RegularConstraint] = []
        self.len_constraints: List[LengthConstraint] = []

    def declare_variable(self, name: str, var_type: VariableType):
        if name in self.variables:
            raise MultiDeclarationError(f'variable: {name}')
        else:
            self.variables[name] = var_type

    def ensure_variable_known(self, var: str, var_type: VariableType):
        if (var not in self.variables or
                self.variables[var] is not var_type):
            raise UnknownVariableError(f'variable: {var} type: {var_type.name}')

    def add_word_equation(self, we: WordEquation):
        for var in we.variables():
            self.ensure_variable_known(var.value, VariableType.string)
        self.word_equations.append(we)

    def add_regular_constraint(self, constr: RegularConstraint):
        self.ensure_variable_known(constr.tgt_var.value, VariableType.string)
        self.reg_constraints.append(constr)

    def add_length_constraint(self, constr: LengthConstraint):
        for var in constr.variables():
            self.ensure_variable_known(var, VariableType.int)
        self.len_constraints.append(constr)
