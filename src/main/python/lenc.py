from enum import Enum, unique
from typing import List, Tuple, Set, Union


class Element:
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
        return f'{c}{self.__class__.__name__[0]}({self.value})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.value == other.value and
                    self.coefficient == other.coefficient)

    def opposite(self):
        return self.__class__('undefined')

    def multiply(self, multiplier: int):
        pass


class Variable(Element):
    def __init__(self, value: Union[str, int], coefficient: int = 1):
        assert isinstance(value, str)
        super().__init__(value, coefficient)

    def opposite(self):
        return self.__class__(self.value, -self.coefficient)

    def multiply(self, multiplier: int):
        if multiplier == 0:
            return Constant(0)
        return self.__class__(self.value, self.coefficient * multiplier)


class Constant(Element):
    def __init__(self, value: Union[str, int]):
        assert isinstance(value, int)
        super().__init__(value)

    def opposite(self):
        return self.__class__(-self.value)

    def multiply(self, multiplier: int):
        return self.__class__(self.value * multiplier)


Expression = List[Element]


def is_var(e: Element):
    return isinstance(e, Variable)


def not_var(e: Element):
    return not is_var(e)


def is_const(e: Element):
    return isinstance(e, Constant)


def not_const(e: Element):
    return not is_const(e)


def is_const_expr(e: Expression):
    return len(e) == 1 and isinstance(e[0], Constant)


def not_const_expr(e: Expression):
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


def reduce_constants(expr: Expression) -> Expression:
    """ Constant (if any) will be the last element. """
    result = []
    acc = 0
    for e in expr:
        if isinstance(e, Constant):
            acc += e.value
        else:
            result.append(e)
    result.append(Constant(acc)) if acc != 0 else None
    return result if result else [Constant(0)]


def simplify_equation(lhs: Expression, rhs: Expression) \
        -> Tuple[Expression, Expression]:
    le, re = reduce_constants(lhs), rhs
    if isinstance(le[-1], Constant) and len(le) > 1:
        re.append(le.pop().opposite())
    return le, reduce_constants(re)


class LengthConstraint:
    def __init__(self, lhs: Expression, rhs: Expression,
                 rel: Relation = Relation.equal):
        lhs, rhs = simplify_equation(lhs, rhs)
        self.lhs: Expression = lhs
        self.rhs: Expression = rhs
        self.relation: Relation = rel

    def __repr__(self):
        return f'{self.lhs} {self.relation.value} {self.rhs}'

    def variables(self) -> Set[Variable]:
        return {e for e in self.lhs + self.rhs if isinstance(e, Variable)}

    def negate(self) -> 'LengthConstraint':
        return self.__class__(self.lhs, self.rhs, negation[self.relation])
