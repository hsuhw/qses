import lenc

from typing import List, Tuple, Optional
from collections import Counter


class Element:
    def __init__(self, value: str):
        self.value: str = value

    def __repr__(self):
        return f'{self.__class__.__name__[0]}({self.value})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash(str(self))

    def length(self):  # TODO: do it symbolically or concretely?
        pass


class Character(Element):
    def length(self):
        return lenc.Constant(1)


class Variable(Element):
    def length(self):
        return lenc.Variable(f'{self.value}_len_')


class Delimiter(Element):
    def __init__(self):
        super().__init__('')

    def length(self):
        return lenc.Constant(1)


DELIMITER = Delimiter()
Expression = List[Element]


def heads_or_none(lhs, rhs) -> Tuple[Optional[Element], Optional[Element]]:
    [lh], [rh] = lhs[:1] or [None], rhs[:1] or [None]
    return lh, rh


def is_var(e: Element):
    return isinstance(e, Variable)


def not_var(e: Element):
    return not is_var(e)


def is_char(e: Element):
    return isinstance(e, Character)


def not_char(e: Element):
    return not is_char(e)


def is_del(e: Element):
    return e is DELIMITER


def not_del(e: Element):
    return not is_del(e)


class WordEquation:
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs: Expression = lhs
        self.rhs: Expression = rhs

    def __repr__(self):
        return f'{self.lhs} = {self.rhs}'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __hash__(self):
        return hash(str(self))

    def variables(self) -> List[Variable]:
        return [e for e in self.lhs + self.rhs if isinstance(e, Variable)]

    def peek(self) -> Tuple[Optional[Element], Optional[Element]]:
        return heads_or_none(self.lhs, self.rhs)

    def copy_expressions(self) -> Tuple[Expression, Expression]:
        return self.lhs[:], self.rhs[:]  # a faster way to make list copies

    def is_quadratic(self):
        count = Counter(self.lhs + self.rhs)
        for e in self.variables():
            if count[e] > 2:
                return False
        return True

    def is_simply_unsolvable(self):
        lh, rh = self.peek()
        return ((is_char(lh) and is_char(rh) and lh != rh) or
                (not lh and rh and not_var(rh)) or
                (not rh and lh and not_var(lh)) or
                (is_del(lh) and not_del(rh) and not_var(rh)) or
                (is_del(rh) and not_del(lh) and not_var(lh)))

    def is_both_var_headed(self):
        lh, rh = self.peek()
        return is_var(lh) and is_var(rh)

    def is_char_var_headed(self):
        lh, rh = self.peek()
        return is_char(lh) and is_var(rh)

    def is_var_char_headed(self):
        lh, rh = self.peek()
        return is_var(lh) and is_char(rh)

    def has_emptiness(self):
        lh, rh = self.peek()
        return not lh or not rh or is_del(lh) or is_del(rh)

    def merge(self, other: 'WordEquation') -> 'WordEquation':
        lhs, rhs = self.copy_expressions()
        lhs.append(DELIMITER)
        rhs.append(DELIMITER)
        return WordEquation(lhs + other.lhs, rhs + other.rhs)

    def remove_heads(self) -> 'WordEquation':
        return WordEquation(self.lhs[1:], self.rhs[1:])

    def remove_trivial_prefix(self) -> 'WordEquation':
        lhs, rhs = self.copy_expressions()
        lh, rh = heads_or_none(lhs, rhs)
        while ((is_char(lh) and is_char(rh) and lh == rh) or
               (is_del(lh) and is_del(rh))):
            lhs.pop(0)
            rhs.pop(0)
            lh, rh = heads_or_none(lhs, rhs)
        return WordEquation(lhs, rhs)

    def remove_left_head_from_all(self) -> 'WordEquation':
        lh = self.lhs[0]
        return WordEquation([e for e in self.lhs if e != lh],
                            [e for e in self.rhs if e != lh])

    def remove_right_head_from_all(self) -> 'WordEquation':
        rh = self.rhs[0]
        return WordEquation([e for e in self.lhs if e != rh],
                            [e for e in self.rhs if e != rh])

    def replace(self, tgt: Element, subst: Element) -> 'WordEquation':
        return WordEquation([(subst if e == tgt else e) for e in self.lhs],
                            [(subst if e == tgt else e) for e in self.rhs])

    def replace_with(self, tgt: Element, subst: List[Element]) \
            -> 'WordEquation':
        lhs_hoisted = [(subst if e == tgt else [e]) for e in self.lhs]
        rhs_hoisted = [(subst if e == tgt else [e]) for e in self.rhs]
        return WordEquation([e for sublist in lhs_hoisted for e in sublist],
                            [e for sublist in rhs_hoisted for e in sublist])
