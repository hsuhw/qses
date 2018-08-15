import we


class RegularConstraint:
    def __init__(self, var: we.Variable, nfa):
        self.tgt_var: we.Variable = var
        self.nfa = nfa
