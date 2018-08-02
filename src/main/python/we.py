from typing import List, Tuple, Optional


class Element:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'{self.__class__.__name__[0]}({self.value})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def length(self):  # TODO: do it symbolically or concretely?
        return False


class Character(Element):
    def length(self):
        return 1


class Variable(Element):
    def length(self):
        return f'{self.value}.len'


class Delimiter(Element):
    def __init__(self):
        super().__init__('')

    def length(self):
        return 1


DELIMITER = Delimiter()
WordExpression = List[Element]


class WordEquation:
    def __init__(self, lhs: WordExpression, rhs: WordExpression):
        self.lhs: WordExpression = lhs
        self.rhs: WordExpression = rhs

    def __repr__(self):
        return f'{self.lhs} = {self.rhs}'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def get_heads_or_none(lhs, rhs) \
            -> Tuple[Optional[Element], Optional[Element]]:
        [lh], [rh] = lhs[:1] or [None], rhs[:1] or [None]
        return lh, rh

    def variables(self) -> List[Variable]:
        return [x for x in self.lhs + self.rhs if isinstance(x, Variable)]

    def peek(self) -> Tuple[Optional[Element], Optional[Element]]:
        return WordEquation.get_heads_or_none(self.lhs, self.rhs)

    def copy_expressions(self) -> Tuple[WordExpression, WordExpression]:
        return self.lhs[:], self.rhs[:]  # a faster way to make list copies

    def is_quadratic(self):
        pass  # TODO

    def is_simply_unsolvable(self):
        lh, rh = self.peek()
        return ((isinstance(lh, Character) and
                 isinstance(rh, Character) and lh != rh) or
                (not lh and rh and not isinstance(rh, Variable)) or
                (not rh and lh and not isinstance(lh, Variable)) or
                (lh == DELIMITER and rh != DELIMITER) or
                (rh == DELIMITER and lh != DELIMITER))

    def is_both_var_headed(self):
        lh, rh = self.peek()
        return isinstance(lh, Variable) and isinstance(rh, Variable)

    def is_char_var_headed(self):
        lh, rh = self.peek()
        return isinstance(lh, Character) and isinstance(rh, Variable)

    def is_var_char_headed(self):
        lh, rh = self.peek()
        return isinstance(lh, Variable) and isinstance(rh, Character)

    def has_emptiness(self):
        return not (self.lhs and self.rhs)

    def merge(self, other: 'WordEquation') -> 'WordEquation':
        lhs, rhs = self.copy_expressions()
        lhs.append(DELIMITER)
        rhs.append(DELIMITER)
        return WordEquation(lhs + other.lhs, rhs + other.rhs)

    def remove_heads(self) -> 'WordEquation':
        return WordEquation(self.lhs[1:], self.rhs[1:])

    def remove_trivial_prefix(self) -> 'WordEquation':
        lhs, rhs = self.copy_expressions()
        lh, rh = WordEquation.get_heads_or_none(lhs, rhs)
        while ((isinstance(lh, Character) and
                isinstance(rh, Character) and lh == rh) or
               (lh is DELIMITER and rh is DELIMITER)):
            lhs.pop(0)
            rhs.pop(0)
            lh, rh = WordEquation.get_heads_or_none(lhs, rhs)
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
