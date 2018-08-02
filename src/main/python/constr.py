from typing import List
from we import Variable

LengthExpression = List[str]


class LengthConstraint:
    def __init__(self, lhs: LengthExpression, rhs: LengthExpression):
        self.lhs: LengthExpression = lhs
        self.rhs: LengthExpression = rhs

    def __repr__(self):
        return f'{self.lhs} = {self.rhs}'

    def variables(self) -> List[str]:
        return [x for x in self.lhs + self.rhs if not x.isdigit()]


class RegularConstraint:
    def __init__(self, var: Variable, nfa):
        self.tgt_var: Variable = var
        self.nfa = nfa
