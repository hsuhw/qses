from typing import List, Tuple, Dict, Set, Optional
from enum import Enum, unique, auto
from functools import reduce
from we import WordEquation, Element, is_var
from prob import Problem, InvalidProblemError


@unique
class Rewrite(Enum):
    lvar_be_empty = auto()
    rvar_be_empty = auto()
    lvar_be_char = auto()
    rvar_be_char = auto()
    lvar_be_rvar = auto()
    lvar_longer_char = auto()
    rvar_longer_char = auto()
    lvar_longer_var = auto()
    rvar_longer_var = auto()


class Transform:
    def __init__(self, source: WordEquation, rewrite: Rewrite,
                 record: Tuple[Optional[Element], Optional[Element]]):
        self.source: WordEquation = source
        self.rewrite: Rewrite = rewrite
        self.record = record

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.source == other.source and
                    self.rewrite == other.rewrite and
                    self.record == other.record)
        return False

    def __hash__(self):
        return hash(self.source) + hash(self.rewrite) + hash(str(self.record))


class SolveTree:
    success_end: WordEquation = WordEquation([], [])

    def __init__(self, root: WordEquation):
        self.root: WordEquation = root
        self.node_relations: Dict[WordEquation, Set[Transform]] = {}

    def has_node(self, node: WordEquation):
        return node in self.node_relations

    def has_solution(self):
        return self.has_node(SolveTree.success_end)

    def add_node(self, src: WordEquation, node: WordEquation, rewrite: Rewrite,
                 record: Tuple[Optional[Element], Optional[Element]]) -> bool:
        transform = Transform(src, rewrite, record)
        if self.has_node(node):
            self.node_relations[node].add(transform)
            return False
        else:
            self.node_relations[node] = {transform}
            return True


class BasicSolver:
    def __init__(self, prob: Problem):
        if len(prob.word_equations) < 1:
            raise InvalidProblemError()
        if len(prob.word_equations) > 1:
            we = reduce(lambda x, y: x.merge(y), prob.word_equations)
        else:
            we = prob.word_equations[0]
        self.pending_checks: List[WordEquation] = [we]
        self.resolve: SolveTree = SolveTree(we)

    def transform_with_emptiness(self, we: WordEquation):
        lh, rh = hh = we.peek()
        if not lh:
            if rh and is_var(rh):
                new_we = we.remove_right_head_from_all()
                if self.resolve.add_node(we, new_we, Rewrite.rvar_be_empty, hh):
                    self.pending_checks.append(new_we)
        else:
            if is_var(lh):
                new_we = we.remove_left_head_from_all()
                if self.resolve.add_node(we, new_we, Rewrite.lvar_be_empty, hh):
                    self.pending_checks.append(new_we)

    def transform_both_var_case(self, we: WordEquation):
        lh, rh = hh = we.peek()

        case1 = we.remove_left_head_from_all()
        if self.resolve.add_node(we, case1, Rewrite.lvar_be_empty, hh):
            self.pending_checks.append(case1)

        case2 = we.remove_right_head_from_all()
        if self.resolve.add_node(we, case2, Rewrite.rvar_be_empty, hh):
            self.pending_checks.append(case2)

        case3 = we.replace(lh, rh).remove_heads()
        if self.resolve.add_node(we, case3, Rewrite.lvar_be_rvar, hh):
            self.pending_checks.append(case3)

        case4 = we.replace_with(lh, [rh, lh]).remove_heads()
        if self.resolve.add_node(we, case4, Rewrite.lvar_longer_var, hh):
            self.pending_checks.append(case4)

        case5 = we.replace_with(rh, [lh, rh]).remove_heads()
        if self.resolve.add_node(we, case5, Rewrite.rvar_longer_var, hh):
            self.pending_checks.append(case5)

    def transform_char_var_case(self, we: WordEquation):
        lh, rh = hh = we.peek()

        case1 = we.remove_right_head_from_all()
        if self.resolve.add_node(we, case1, Rewrite.rvar_be_empty, hh):
            self.pending_checks.append(case1)

        case2 = we.replace(rh, lh).remove_heads()
        if self.resolve.add_node(we, case2, Rewrite.rvar_be_char, hh):
            self.pending_checks.append(case2)

        case3 = we.replace_with(rh, [lh, rh]).remove_heads()
        if self.resolve.add_node(we, case3, Rewrite.rvar_longer_char, hh):
            self.pending_checks.append(case3)

    def transform_var_char_case(self, we: WordEquation):
        lh, rh = hh = we.peek()

        case1 = we.remove_left_head_from_all()
        if self.resolve.add_node(we, case1, Rewrite.lvar_be_empty, hh):
            self.pending_checks.append(case1)

        case2 = we.replace(lh, rh).remove_heads()
        if self.resolve.add_node(we, case2, Rewrite.lvar_be_char, hh):
            self.pending_checks.append(case2)

        case3 = we.replace_with(lh, [rh, lh]).remove_heads()
        if self.resolve.add_node(we, case3, Rewrite.lvar_longer_char, hh):
            self.pending_checks.append(case3)

    def solve(self) -> SolveTree:
        while self.pending_checks:
            curr_we = self.pending_checks.pop(0).remove_trivial_prefix()
            if curr_we.is_simply_unsolvable():
                pass
            elif curr_we.has_emptiness():
                self.transform_with_emptiness(curr_we)
            elif curr_we.is_both_var_headed():
                self.transform_both_var_case(curr_we)
            elif curr_we.is_char_var_headed():
                self.transform_char_var_case(curr_we)
            elif curr_we.is_var_char_headed():
                self.transform_var_char_case(curr_we)
        return self.resolve
