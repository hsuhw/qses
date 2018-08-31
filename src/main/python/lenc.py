from enum import Enum, unique
from typing import List, Tuple, Set, Union


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


def reduce_constants(expr: IntExpression) -> IntExpression:
    """ Constant (if any) will be the last element. """
    result = []
    acc = 0
    for e in expr:
        if isinstance(e, IntConstant):
            acc += e.value
        else:
            result.append(e)
    result.append(IntConstant(acc)) if acc != 0 else None
    return result if result else [IntConstant(0)]


def simplify_equation(lhs: IntExpression, rhs: IntExpression) \
        -> Tuple[IntExpression, IntExpression]:
    le, re = reduce_constants(lhs), rhs
    if isinstance(le[-1], IntConstant) and len(le) > 1:
        re.append(le.pop().opposite())
    return le, reduce_constants(re)


class LengthConstraint:
    def __init__(self, lhs: IntExpression, rhs: IntExpression,
                 rel: Relation = Relation.equal):
        lhs, rhs = simplify_equation(lhs, rhs)
        self.lhs: IntExpression = lhs
        self.rhs: IntExpression = rhs
        self.relation: Relation = rel

    def __repr__(self):
        lhs = ' '.join([str(e) for e in self.lhs])
        rhs = ' '.join([str(e) for e in self.rhs])
        return f'ie[{lhs} {self.relation.value} {rhs}]'

    def variables(self) -> Set[IntVariable]:
        return {e for e in self.lhs + self.rhs if isinstance(e, IntVariable)}

    def negate(self) -> 'LengthConstraint':
        return self.__class__(self.lhs, self.rhs, negation[self.relation])
#
#
# functions for print length constraints
def print_lenc_elements_pretty(elem: Element) -> str:
    sign = ''
    if elem.coefficient == -1:
        sign = '-'
    if isinstance(elem,Variable):
        return f'{sign}{elem.value[3:-5]}' # remove post-fix "_len_"
    if isinstance(elem,Constant):
        return f'{sign}{elem.value}'

def print_length_constraint_pretty(lenc: LengthConstraint) -> str:
    if len(lenc.lhs)==0 or len(lenc.rhs)==0:
        return ''

    left = print_lenc_elements_pretty(lenc.lhs[0])
    right = print_lenc_elements_pretty(lenc.rhs[0])
    for e in lenc.lhs[1:]:
        tmp_str = print_lenc_elements_pretty(e)
        if temp_str[0]=='-':
            left += f' - {tmp_str[1:]}'
        else:
            left += f' + {tmp_str[1:]}'

    for e in lenc.rhs[1:]:
        tmp_str = print_lenc_elements_pretty(e)
        if temp_str[0]=='-':
            right += f' - {tmp_str[1:]}'
        else:
            right += f' + {tmp_str[1:]}'

    return f'{left} == {right}'
#
def print_length_constraints_as_condition(lencs: List[LengthConstraint]) -> str:
    if len(lencs) == 0: # no length constraints
        return ''

    if len(lencs) == 1:
        ret = f'{print_length_constraint_pretty(lencs[0])}'
    else:
        ret = f'({print_length_constraint_pretty(lencs[0])})'
        for lenc in lencs[1:]:
            ret += f' && ({print_length_constraint_pretty(lenc)})'

    return ret
#