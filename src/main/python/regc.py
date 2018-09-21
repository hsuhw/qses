from typing import Tuple

from fsa import FSA

RegExpression = FSA
RegOrigin = Tuple[str, bool]  # origin of the regex, negated or not


class RegularConstraint:
    def __init__(self, str_var: str, nfa: RegExpression, origin: RegOrigin):
        self.str_var: str = str_var
        self.nfa: RegExpression = nfa
        self.origin: RegOrigin = origin

    def negate(self) -> 'RegularConstraint':
        origin = self.origin[0], not self.origin[1]
        return self.__class__(self.str_var, self.nfa.complement(), origin)
