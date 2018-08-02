from typing import Dict, List
from enum import Enum, unique
from functools import reduce
from we import WordEquation, Variable
from constr import RegularConstraint, LengthConstraint
from stree import RewriteType, Transform, SolvingTree


@unique
class VariableType(Enum):
    bool = 1
    int = 2
    string = 3


class MultiDeclarationError(Exception):
    pass


class UnknownVariableError(Exception):
    pass


class InvalidProblemError(Exception):
    pass


class Problem:
    def __init__(self):
        self.variables: Dict[str, VariableType] = {}
        self.word_equations: List[WordEquation] = []
        self.reg_constraints: List[RegularConstraint] = []
        self.len_constraints: List[LengthConstraint] = []

    def declare_variable(self, name: str, var_type: VariableType):
        if name in self.variables:
            raise MultiDeclarationError(f'variable: {name}')
        else:
            self.variables[name] = var_type

    def ensure_variable_known(self, var: str, var_type: VariableType):
        if (var not in self.variables or
                self.variables[var] is not var_type):
            raise UnknownVariableError(f'variable: {var} type: {var_type.name}')

    def add_word_equation(self, we: WordEquation):
        for var in we.variables():
            self.ensure_variable_known(var.value, VariableType.string)
        self.word_equations.append(we)

    def add_regular_constraint(self, constr: RegularConstraint):
        self.ensure_variable_known(constr.tgt_var.value, VariableType.string)
        self.reg_constraints.append(constr)

    def add_length_constraint(self, constr: LengthConstraint):
        for var in constr.variables():
            self.ensure_variable_known(var, VariableType.int)
        self.len_constraints.append(constr)

    @staticmethod
    def transform_with_emptiness(we: WordEquation,
                                 pending: List[WordEquation],
                                 result: SolvingTree):
        lh, rh = hh = we.peek()
        if not lh:
            if rh and isinstance(rh, Variable):
                new_we = we.remove_right_head_from_all()
                result.add_node(new_we,
                                Transform(we, RewriteType.rvar_be_empty, hh))
                pending.append(new_we)
        else:
            if isinstance(lh, Variable):
                new_we = we.remove_left_head_from_all()
                result.add_node(new_we,
                                Transform(we, RewriteType.lvar_be_empty, hh))
                pending.append(new_we)

    @staticmethod
    def transform_both_var_case(we: WordEquation,
                                pending: List[WordEquation],
                                result: SolvingTree):
        lh, rh = hh = we.peek()

        case1 = we.remove_left_head_from_all()
        result.add_node(case1, Transform(we, RewriteType.lvar_be_empty, hh))
        pending.append(case1)

        case2 = we.remove_right_head_from_all()
        result.add_node(case2, Transform(we, RewriteType.rvar_be_empty, hh))
        pending.append(case2)

        case3 = we.replace(lh, rh).remove_heads()
        result.add_node(case3, Transform(we, RewriteType.lvar_be_rvar, hh))
        pending.append(case3)

        case4 = we.replace_with(lh, [rh, lh]).remove_heads()
        result.add_node(case4, Transform(we, RewriteType.lvar_longer_var, hh))
        pending.append(case4)

        case5 = we.replace_with(rh, [lh, rh]).remove_heads()
        result.add_node(case5, Transform(we, RewriteType.rvar_longer_var, hh))
        pending.append(case5)

    @staticmethod
    def transform_char_var_case(we: WordEquation,
                                pending: List[WordEquation],
                                result: SolvingTree):
        lh, rh = hh = we.peek()

        case1 = we.remove_right_head_from_all()
        result.add_node(case1, Transform(we, RewriteType.rvar_be_empty, hh))
        pending.append(case1)

        case2 = we.replace(rh, lh).remove_heads()
        result.add_node(case2, Transform(we, RewriteType.rvar_be_char, hh))
        pending.append(case2)

        case3 = we.replace_with(rh, [lh, rh]).remove_heads()
        result.add_node(case3, Transform(we, RewriteType.rvar_longer_char, hh))
        pending.append(case3)

    @staticmethod
    def transform_var_char_case(we: WordEquation,
                                pending: List[WordEquation],
                                result: SolvingTree):
        lh, rh = hh = we.peek()

        case1 = we.remove_left_head_from_all()
        result.add_node(case1, Transform(we, RewriteType.lvar_be_empty, hh))
        pending.append(case1)

        case2 = we.replace(lh, rh).remove_heads()
        result.add_node(case2, Transform(we, RewriteType.lvar_be_char, hh))
        pending.append(case2)

        case3 = we.replace_with(lh, [rh, lh]).remove_heads()
        result.add_node(case3, Transform(we, RewriteType.lvar_longer_char, hh))
        pending.append(case3)

    def build_solving_tree(self) -> SolvingTree:
        if len(self.word_equations) < 1:
            raise InvalidProblemError()
        if len(self.word_equations) > 1:
            singleton = [reduce(lambda x, y: x.merge(y), self.word_equations)]
            self.word_equations = singleton

        we = self.word_equations[0]
        pending: List[WordEquation] = [we]
        result: SolvingTree = SolvingTree(we)

        while pending:
            curr_we = pending.pop(0).remove_trivial_prefix()
            if curr_we.is_simply_unsolvable():
                pass
            elif curr_we.has_emptiness():
                self.transform_with_emptiness(curr_we, pending, result)
            elif curr_we.is_both_var_headed():
                self.transform_both_var_case(curr_we, pending, result)
            elif curr_we.is_char_var_headed():
                self.transform_char_var_case(curr_we, pending, result)
            elif curr_we.is_var_char_headed():
                self.transform_var_char_case(curr_we, pending, result)

        return result
