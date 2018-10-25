import tok

from enum import Enum, unique
from re import compile
from typing import List, Tuple, Set, Union, Optional
from tok import INTERNAL_VAR_PREFIX, INTERNAL_LEN_VAR_POSTFIX


class IntElement:
    def __init__(self, value: Union[str, int], coefficient: int = 1):
        assert coefficient != 0
        self.value: Union[str, int] = value
        self.coefficient: int = coefficient

    def __repr__(self):
        c = self.coefficient
        if self.coefficient == 1:
            c = ''
        elif self.coefficient == -1:
            c = '-'
        return f'{c}{self.__class__.__name__[3]}({self.value})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.value == other.value and
                    self.coefficient == other.coefficient)

    def __hash__(self):
        return hash(str(self))

    def opposite(self):
        return self.__class__('undefined')

    def multiply(self, multiplier: int):
        pass


class IntVariable(IntElement):
    def __init__(self, value: Union[str, int], coefficient: int = 1):
        assert isinstance(value, str)
        super().__init__(value, coefficient)

    def opposite(self):
        return self.__class__(self.value, -self.coefficient)

    def multiply(self, multiplier: int):
        if multiplier == 0:
            return IntConstant(0)
        return self.__class__(self.value, self.coefficient * multiplier)


class IntConstant(IntElement):
    def __init__(self, value: Union[str, int]):
        assert isinstance(value, int)
        super().__init__(value)

    def opposite(self):
        return self.__class__(-self.value)

    def multiply(self, multiplier: int):
        return self.__class__(self.value * multiplier)


IntExpression = List[IntElement]


def is_var(e: IntElement):
    return isinstance(e, IntVariable)


def not_var(e: IntElement):
    return not is_var(e)


def is_const(e: IntElement):
    return isinstance(e, IntConstant)


def not_const(e: IntElement):
    return not is_const(e)


def is_const_expr(e: IntExpression):
    return len(e) == 1 and isinstance(e[0], IntConstant)


def not_const_expr(e: IntExpression):
    return not is_const_expr(e)


internal_len_var_name = compile(
    f'{tok.INTERNAL_VAR_PREFIX}(.*){tok.INTERNAL_LEN_VAR_POSTFIX}')


def length_origin_name(e: IntElement) -> Optional[str]:
    result = internal_len_var_name.match(e.value)
    return result.group(1) if result else None


@unique
class Relation(Enum):
    equal = '=='
    unequal = '!='
    greater = '>'
    greater_equal = '>='
    less = '<'
    less_equal = '<='


negation = {
    Relation.equal: Relation.unequal,
    Relation.unequal: Relation.equal,
    Relation.greater: Relation.less_equal,
    Relation.greater_equal: Relation.less,
    Relation.less: Relation.greater_equal,
    Relation.less_equal: Relation.greater,
}

CONST_TERM_KEY = '1'  # a number would never be a variable name


def reduce_in_arithmetic(expr: IntExpression) -> IntExpression:
    """ Constant (if any) will be the last element in the returned list. """
    acc = {CONST_TERM_KEY: 0}
    for e in expr:
        if is_const(e):
            acc[CONST_TERM_KEY] += e.value
        else:
            v = e.value
            acc[v] = acc[v] + e.coefficient if v in acc else e.coefficient
    result = []
    for var, c in acc.items():
        if var != CONST_TERM_KEY and c != 0:
            result.append(IntVariable(var, c))
    if acc[CONST_TERM_KEY] != 0:
        result.append(IntConstant(acc[CONST_TERM_KEY]))
    return result or [IntConstant(0)]


def simplify_equation(lhs: IntExpression, rhs: IntExpression) \
        -> Tuple[IntExpression, IntExpression]:
    le = reduce_in_arithmetic(lhs + [e.opposite() for e in rhs])
    re = []
    shift_to_right: Set[IntElement] = set()
    for e in le:
        if (is_var(e) and e.coefficient < 0) or (is_const(e) and e.value < 0):
            re.append(e.opposite())
            shift_to_right.add(e)
    le = [e for e in le if e not in shift_to_right]
    return (le or [IntConstant(0)]), (re or [IntConstant(0)])


class LengthConstraint:
    def __init__(self, lhs: IntExpression, rhs: IntExpression,
                 rel: Relation = Relation.equal):
        lhs, rhs = simplify_equation(lhs, rhs)
        self.lhs: IntExpression = lhs
        self.rhs: IntExpression = rhs
        self.relation: Relation = rel

    def __str__(self):
        lhs = ' '.join(map(str, self.lhs))
        rhs = ' '.join(map(str, self.rhs))
        return f'ic[{lhs} {self.relation.value} {rhs}]'

    def __eq__(self, other):
        return isinstance(other, LengthConstraint) and str(self) == str(other)

    def variables(self) -> Set[IntVariable]:
        return {e for e in self.lhs + self.rhs if isinstance(e, IntVariable)}

    def negate(self) -> 'LengthConstraint':
        return self.__class__(self.lhs, self.rhs, negation[self.relation])


def print_lenc_elements_pretty(elem: IntElement) -> str:
    sign = ''
    if elem.coefficient == -1:
        sign = '-'
    if isinstance(elem, IntVariable):
        if INTERNAL_VAR_PREFIX in elem.value and INTERNAL_LEN_VAR_POSTFIX in elem.value:
            return f'{sign}{elem.value[len(INTERNAL_VAR_PREFIX):-len(INTERNAL_LEN_VAR_POSTFIX)]}'
        else:
            return f'{sign}{elem.value}'
    if isinstance(elem, IntConstant):
        return f'{sign}{elem.value}'


def print_length_constraint_pretty(lenc: LengthConstraint) -> str:
    if len(lenc.lhs) == 0 or len(lenc.rhs) == 0:
        return ''

    left = print_lenc_elements_pretty(lenc.lhs[0])
    right = print_lenc_elements_pretty(lenc.rhs[0])
    for e in lenc.lhs[1:]:
        tmp_str = print_lenc_elements_pretty(e)
        if tmp_str[0] == '-':
            left += f' - {tmp_str[1:]}'
        else:
            left += f' + {tmp_str}'

    for e in lenc.rhs[1:]:
        tmp_str = print_lenc_elements_pretty(e)
        if tmp_str[0] == '-':
            right += f' - {tmp_str[1:]}'
        else:
            right += f' + {tmp_str}'

    if lenc.relation == Relation.equal:
        return f'{left} == {right}'
    elif lenc.relation == Relation.unequal:
        return f'{left} != {right}'
    elif lenc.relation == Relation.greater:
        return f'{left} > {right}'
    elif lenc.relation == Relation.greater_equal:
        return f'{left} >= {right}'
    elif lenc.relation == Relation.less:
        return f'{left} < {right}'
    elif lenc.relation == Relation.less_equal:
        return f'{left} >= {right}'


def print_length_constraints_as_one_condition(lencs: List[LengthConstraint]) -> [str]:
    if len(lencs) == 0:  # no length constraints
        return None
    if len(lencs) == 1:
        ret = f'{print_length_constraint_pretty(lencs[0])}'
    else:
        ret = f'({print_length_constraint_pretty(lencs[0])})'
        for lenc in lencs[1:]:
            ret += f' && ({print_length_constraint_pretty(lenc)})'
    return ret


def print_length_constraints_as_strings(lencs: List[LengthConstraint]) -> [str]:
    if len(lencs) == 0:  # no length constraints
        return None
    ret = [print_length_constraint_pretty(lencs[0])]
    if len(lencs) > 1:
        for lenc in lencs[1:]:
            ret.append(print_length_constraint_pretty(lenc))
    return ret
