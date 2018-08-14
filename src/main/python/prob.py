from typing import Dict, List
from enum import Enum, unique, auto
from we import WordEquation, Element, Variable, Character, WordExpression
from constr import RegularConstraint, LengthConstraint
import re

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


class ReadProb():
    def __init__(self):
        self.word_equations: List[WordEquation] = []
        self.length_constraints: List[str] = []
        self.membership_constraints: List[str] = []
        
    def string_to_elements(self, str):
        l = list(str)
        ret = []
        for s in l:
            ret.append(Character(s))
        return ret

    def is_comment_line(self, line):
        return re.match(r"^#.*$", line)

    def is_empty_line(self, line):
        return re.match(r"^\s*$", line)

    def is_word_equation_line(self, line):
        return re.match(r"^.*( = ).*$", line)

    def is_length_constraint_line(self, line):  # todo
        return re.match(r"^\(.*\)$", line)

    def is_membership_constraint_line(self, line):  # todo
        return re.match(r"^\[.*\]$", line)

    def is_string(self, str):
        return re.match(r"^\".*\"$",str) or re.match(r"^\'.*\'$",str)

    def is_variable(self, str):
        return re.match(r"^[_$a-zA-z].*",str)

    def process_elements(self, str_list: List[str]) -> List[Element]:
        elem_list = []
        #print(str_list)
        for s in str_list:
            if self.is_string(s):  # string
                elem_list += self.string_to_elements(s[1:len(s)-1])
            elif self.is_variable(s):
                elem_list.append(Variable(s))
            else:
                assert(False)  # fail, should not reach this line
        return elem_list

    # Process a line of input and return a word equation
    def process_line(self, line: str) -> WordEquation:
        str_pair = line.split(' = ')  # separate left and right of a word equation
        #print str_pair[0], ', ', strPair[1]
        assert(len(str_pair)==2)
        return WordEquation(self.process_elements(str_pair[0].split()),self.process_elements(str_pair[1].split()))
 
    def read_input_file(self, filename: str):
        with open(filename) as fp:
            lines = fp.readlines()
        for line in lines:
            ln = line.strip()
            if self.is_comment_line(ln) or self.is_empty_line(ln):
                pass
            elif self.is_length_constraint_line(ln):
                self.length_constraints.append(ln)
            elif self.is_membership_constraint_line(ln):
                self.membership_constraints.append(ln)
            elif self.is_word_equation_line(ln):
                self.word_equations.append(self.process_line(ln))
#
#
def safely_add_word_equation_to_problem(we: WordEquation, prob: Problem):
    for v in we.variables():
        if (v.value not in prob.variables):
            prob.declare_variable(v.value,VariableType.string)
    prob.add_word_equation(we)
#

