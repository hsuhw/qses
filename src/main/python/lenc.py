from typing import List, Union


class Element:
    def __init__(self, value: Union[str, int], negative: bool = False):
        self.value: Union[str, int] = value
        self.is_negative: bool = negative

    def __repr__(self):
        sign = '-' if self.is_negative else ''
        return f'{self.__class__.__name__[0]}({sign}{self.value})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.value == other.value and
                    self.is_negative == other.is_negative)

    def opposite(self):
        return self.__class__(self.value, not self.is_negative)


class Variable(Element):
    def __init__(self, value: Union[str, int], negative: bool = False):
        assert isinstance(value, str)
        super().__init__(value, negative)


class Constant(Element):
    def __init__(self, value: Union[str, int]):
        assert isinstance(value, int)
        super().__init__(value)

    def opposite(self):
        return self.__class__(-self.value)


Expression = List[Element]


class LengthConstraint:
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs: Expression = lhs
        self.rhs: Expression = rhs

    def __repr__(self):
        return f'{self.lhs} = {self.rhs}'

    def variables(self) -> List[Variable]:
        return [e for e in self.lhs + self.rhs if isinstance(e, Variable)]
