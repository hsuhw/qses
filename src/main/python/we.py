from typing import List, Tuple, Set, Optional, Iterator
from collections import Counter

from lenc import IntConstant, IntVariable
from tok import INTERNAL_VAR_PREFIX, INTERNAL_LEN_VAR_POSTFIX


class StrElement:
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
        return IntVariable(0)  # dummy value


class Character(StrElement):
    def length(self):
        return IntConstant(1)


class StrVariable(StrElement):
    def __repr__(self):
        return f'V({self.value})'

    def length(self):
        n = f'{INTERNAL_VAR_PREFIX}{self.value}{INTERNAL_LEN_VAR_POSTFIX}'
        return IntVariable(n)


class Delimiter(StrElement):
    def __init__(self):
        super().__init__('')

    def length(self):
        return IntConstant(1)


DELIMITER = Delimiter()
StrExpression = List[StrElement]
HeadPair = Tuple[Optional[StrElement], Optional[StrElement]]


def heads_or_none(lhs, rhs, reverse: bool = False) -> HeadPair:
    if reverse:
        [lh], [rh] = lhs[-1:] or [None], rhs[-1:] or [None]
    else:
        [lh], [rh] = lhs[:1] or [None], rhs[:1] or [None]
    return lh, rh


def is_var(e: StrElement):
    return isinstance(e, StrVariable)


def not_var(e: StrElement):
    return not is_var(e)


def is_char(e: StrElement):
    return isinstance(e, Character)


def not_char(e: StrElement):
    return not is_char(e)


def is_del(e: StrElement):
    return e is DELIMITER


def not_del(e: StrElement):
    return not is_del(e)


def split_expression(expr: StrExpression) -> List[StrExpression]:
    result = []
    curr_list = []
    for e in expr:
        if is_del(e):
            result.append(curr_list)
            curr_list = []
            continue
        curr_list.append(e)
    result.append(curr_list) if curr_list else None
    return result


class WordEquation:
    def __init__(self, lhs: StrExpression, rhs: StrExpression,
                 neg: bool = False):
        self.lhs: StrExpression = lhs
        self.rhs: StrExpression = rhs
        self.negation: bool = neg

    def __str__(self):
        lhs = ' '.join(map(str, self.lhs))
        rhs = ' '.join(map(str, self.rhs))
        return f'wc[{lhs} {"!=" if self.negation else "=="} {rhs}]'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __hash__(self):
        return hash(str(self))

    def __len__(self):
        return len(self.lhs) + len(self.rhs)

    def min_side_len(self):
        return min(len(self.lhs), len(self.rhs))

    def __bool__(self):
        return True

    def variables(self) -> Set[StrVariable]:
        return {e for e in self.lhs + self.rhs if isinstance(e, StrVariable)}

    def var_occurrence(self, elem: Optional[StrVariable] = None):
        if elem:  # if elem is specified, count occurrences of elem
            return sum([1 for e in self.lhs + self.rhs if e == elem])
        else:  # if elem is not specified, count occurrences of all string variables
            return sum([1 for e in self.lhs + self.rhs if is_var(e)])

    def negate(self) -> 'WordEquation':
        return self.__class__(self.lhs, self.rhs, not self.negation)

    def peek(self) -> HeadPair:
        return heads_or_none(self.lhs, self.rhs)

    def peek_rev(self) -> HeadPair:
        return heads_or_none(self.lhs, self.rhs, reverse=True)

    def copy_expressions(self) -> Tuple[StrExpression, StrExpression]:
        return self.lhs[:], self.rhs[:]  # a faster way to make list copies

    def is_quadratic(self):
        count = Counter(self.lhs + self.rhs)
        for e in self.variables():
            if count[e] > 2:
                return False
        return True

    def is_simply_unequal(self):
        lh, rh = self.peek()
        lh_r, rh_r = self.peek_rev()
        ret1 = ((is_char(lh) and is_char(rh) and lh != rh) or
                (not lh and rh and not_var(rh)) or
                (not rh and lh and not_var(lh)) or
                (is_del(lh) and not_del(rh) and not_var(rh)) or
                (is_del(rh) and not_del(lh) and not_var(lh)))
        ret2 = ((is_char(lh_r) and is_char(rh_r) and lh_r != rh_r) or
                (not lh_r and rh_r and not_var(rh_r)) or
                (not rh_r and lh_r and not_var(lh_r)) or
                (is_del(lh_r) and not_del(rh_r) and not_var(rh_r)) or
                (is_del(rh_r) and not_del(lh_r) and not_var(lh_r)))
        return ret1 or ret2
        # return ret1

    def is_empty_constant(self):  # check if one side empty with another side has constants (unsolvable)
        lh, rh = self.peek()
        if (not lh and not rh) or (lh and rh):
            return False
        elif not lh and rh:
            return sum([1 for e in self.rhs if is_char(e)]) > 0
        elif lh and not rh:
            return sum([1 for e in self.lhs if is_char(e)]) > 0
        else:
            assert False

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
        # assert not other.negation and not self.negation
        lhs, rhs = self.copy_expressions()
        lhs.append(DELIMITER)
        rhs.append(DELIMITER)
        return WordEquation(lhs + other.lhs, rhs + other.rhs)

    def split(self) -> Iterator[Tuple[StrExpression, StrExpression]]:
        left_parts = split_expression(self.lhs)
        right_parts = split_expression(self.rhs)
        assert len(left_parts) == len(right_parts)
        return zip(left_parts, right_parts)

    def remove_heads(self) -> 'WordEquation':
        return WordEquation(self.lhs[1:], self.rhs[1:], self.negation)

    def remove_left_head_from_all(self) -> 'WordEquation':
        lh = self.lhs[0]
        return WordEquation([e for e in self.lhs if e != lh],
                            [e for e in self.rhs if e != lh], self.negation)

    def remove_right_head_from_all(self) -> 'WordEquation':
        rh = self.rhs[0]
        return WordEquation([e for e in self.lhs if e != rh],
                            [e for e in self.rhs if e != rh], self.negation)

    def trim_prefix(self) -> 'WordEquation':
        lhs, rhs = self.copy_expressions()
        lh, rh = heads_or_none(lhs, rhs)
        while ((is_char(lh) and is_char(rh) and lh == rh) or
               (is_var(lh) and is_var(rh) and lh == rh) or
               (is_del(lh) and is_del(rh))):
            lhs.pop(0)
            rhs.pop(0)
            lh, rh = heads_or_none(lhs, rhs)
        return WordEquation(lhs, rhs, self.negation)

    def replace(self, tgt: StrElement, subst: StrElement) -> 'WordEquation':
        return WordEquation([(subst if e == tgt else e) for e in self.lhs],
                            [(subst if e == tgt else e) for e in self.rhs],
                            self.negation)

    def replace_with(self, tgt: StrElement, subst: List[StrElement]) \
            -> 'WordEquation':
        lhs_hoisted = [(subst if e == tgt else [e]) for e in self.lhs]
        rhs_hoisted = [(subst if e == tgt else [e]) for e in self.rhs]
        return WordEquation([e for sublist in lhs_hoisted for e in sublist],
                            [e for sublist in rhs_hoisted for e in sublist],
                            self.negation)

    def remove_element_from_all(self, elem: StrElement) -> 'WordEquation':
        return WordEquation([e for e in self.lhs if e != elem],
                            [e for e in self.rhs if e != elem], self.negation)
