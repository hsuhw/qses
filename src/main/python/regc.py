from fsa import FSA

RegExpression = FSA


class RegularConstraint:
    def __init__(self, str_var: str, nfa: RegExpression):
        self.str_var: str = str_var
        self.nfa: RegExpression = nfa

    def negate(self) -> 'RegularConstraint':
        return self.__class__(self.str_var, self.nfa.complement())
