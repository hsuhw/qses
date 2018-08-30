from we import StrVariable


class RegularConstraint:
    def __init__(self, var: StrVariable, nfa):
        self.tgt_var: StrVariable = var
        self.nfa = nfa
