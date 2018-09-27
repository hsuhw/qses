from fsa import FSA

RegExpression = FSA


class RegularConstraint:
    def __init__(self, str_var: str, fsa: RegExpression, fsa_src: str,
                 neg: bool):
        self.str_var: str = str_var
        self.fsa: RegExpression = fsa
        self.fsa_src: str = fsa_src
        self.negation: bool = neg

    def __str__(self):
        var = f'var: {self.str_var}'
        neg = f'negation: {self.negation}'
        src = f'src: {self.fsa_src}'
        fsa = f'fsa: {{\n{self.fsa}\n}}'
        return '\n'.join([var, neg, src, fsa])

    def negate(self) -> 'RegularConstraint':
        return self.__class__(self.str_var, self.fsa, self.fsa_src,
                              not self.negation)
