import re

from typing import List

from prob import Problem, ValueType
from we import WordEquation, StrVariable, Character, StrExpression


def string_to_elements(ss):
    ret = []
    for s in list(ss):
        ret.append(Character(s))
    return ret


def is_comment_line(line):
    return re.match(r"^#.*$", line)


def is_empty_line(line):
    return re.match(r"^\s*$", line)


def is_word_equation_line(line):
    return re.match(r"^.*( = ).*$", line)


def is_length_constraint_line(line):  # TODO
    return re.match(r"^\(.*\)$", line)


def is_membership_constraint_line(line):  # TODO
    return re.match(r"^\[.*\]$", line)


def is_string(s):
    return re.match(r"^\".*\"$", s) or re.match(r"^\'.*\'$", s)


def is_variable(s):
    return re.match(r"^[_$a-zA-z].*", s)


def process_elements(str_list: List[str]) -> StrExpression:
    elem_list = []
    for s in str_list:
        if is_string(s):
            elem_list += string_to_elements(s[1:len(s) - 1])
        elif is_variable(s):
            elem_list.append(StrVariable(s))
        else:
            assert False
    return elem_list


# Process a line of input and return a word equation
def process_line(line: str) -> WordEquation:
    str_pair = line.split(' = ')
    # separate left and right of a word equation
    # print str_pair[0], ', ', strPair[1]
    assert (len(str_pair) == 2)
    return WordEquation(process_elements(str_pair[0].split()),
                        process_elements(str_pair[1].split()))


class ReadProb:
    def __init__(self):
        self.word_equations: List[WordEquation] = []
        self.length_constraints: List[str] = []
        self.membership_constraints: List[str] = []

    def read_input_file(self, filename: str):
        with open(filename) as fp:
            lines = fp.readlines()
        for line in lines:
            ln = line.strip()
            if is_comment_line(ln) or is_empty_line(ln):
                pass
            elif is_length_constraint_line(ln):
                self.length_constraints.append(ln)
            elif is_membership_constraint_line(ln):
                self.membership_constraints.append(ln)
            elif is_word_equation_line(ln):
                self.word_equations.append(process_line(ln))


def safely_add_word_equation_to_problem(w: WordEquation, prob: Problem):
    for v in w.variables():
        if v.value not in prob.variables:
            prob.declare_variable(v.value, ValueType.string)
    prob.add_word_equation(w)
