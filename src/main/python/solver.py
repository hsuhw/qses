from enum import Enum, unique, auto
from functools import reduce
from itertools import chain, combinations
from typing import List, Tuple, Dict, Set, Optional

from graphviz import Digraph
from lenc import LengthConstraint, print_length_constraints_as_strings, internal_len_var_name
from prob import Problem, ValueType, internal_str_var_origin_name
from we import WordEquation, StrElement, StrVariable, is_var, is_del, is_char
from fsa import FSA, from_str, remove_first_char, split_by_states, FsaClassification


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


class Strategy(Enum):
    full = auto()
    first = auto()
    shortest = auto()
    shortest_side = auto()
    one_elem_first = auto()
    var_char_first = auto()
    empty_vars_first = auto()
    customized = auto()


TransformRecord = Tuple[Optional[StrElement], Optional[StrElement]]
RegConstraintClasses = Dict[str, int]
RegConstraints = Dict[str, FSA]

fsa_classification = FsaClassification()  # Object storing FSA classifications


class SolveTreeNode:
    success_we: WordEquation = WordEquation([], [])

    def __init__(self, word_equations: List[WordEquation], reg_constraints: Optional[Dict[str, FSA]] = None):
        self.word_equations = word_equations
        self.reg_constraints: [RegConstraints] = reg_constraints
        if reg_constraints:  # there are regular constraint
            self.regc_classes: [RegConstraintClasses] = dict()
            self.regc_classes = dict()
            for name in reg_constraints:
                self.regc_classes[name] = fsa_classification.get_classification(reg_constraints[name])
        else:
            self.regc_classes: [RegConstraintClasses] = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.regc_classes and other.regc_classes:
                return set(self.word_equations) == set(other.word_equations) and\
                       same_reg_constraints(self.reg_constraints, other.reg_constraints)
            elif not self.regc_classes and not other.regc_classes:
                return set(self.word_equations) == set(other.word_equations)
        return False

    def __len__(self):
        return sum(map(len, self.word_equations))

    def __bool__(self):
        return True

    def __str__(self):
        if self.reg_constraints:
            return '\n'.join(map(str, self.word_equations)) + ':\n' +\
                   '\n'.join([f'{s}:\n{str(self.reg_constraints[s])}' for s in sorted(self.reg_constraints)])
        else:
            return '\n'.join([str(e) for e in self.word_equations]) + ':{{}}'

    def __repr__(self):
        if self.regc_classes:
            return ','.join(map(str, self.word_equations)) + ':{{' +\
                   ','.join([f'{s}:{self.regc_classes[s]}' for s in sorted(self.regc_classes)]) +\
                   '}'
        else:
            return ','.join(map(str, self.word_equations)) + ':{{}}'

    def __hash__(self):
        return hash(repr(self))

    def variables(self) -> Set[StrVariable]:
        return reduce(lambda x, y: x | y, [e.variables() for e in self.word_equations])

    def var_occurrence(self, elem: StrVariable):
        return sum([we.var_occurrence(elem) for we in self.word_equations])

    def is_success_node(self):
        # return reduce(lambda x, y: x and y, map(lambda x: x == self.success_we, self.word_equations))
        return reduce(lambda x, y: x and y, [e == self.success_we for e in self.word_equations])

    def is_unsolvable_node(self):
        if self.reg_constraints:
            return reduce(lambda x, y: x or y,
                          [we.is_simply_unequal() or we.is_empty_constant() for we in self.word_equations] +
                          [self.reg_constraints[n].is_empty() for n in self.reg_constraints])
        else:
            return reduce(lambda x, y: x or y,
                          [we.is_simply_unequal() or we.is_empty_constant() for we in self.word_equations])

    def get_word_equations_to_solve(self) -> List[WordEquation]:
        return [we for we in self.word_equations if we != self.success_we and not we.is_simply_unequal()]

    def pick_first_word_equation(self) -> Optional[WordEquation]:
        candidates = [we for we in self.word_equations if we != self.success_we and not we.is_simply_unequal()]
        if len(candidates) > 0:
            return candidates[0]
        else:
            return None

    def pick_var_char_word_equation(self) -> Optional[WordEquation]:
        candidates = [we for we in self.word_equations if we != self.success_we and not we.is_simply_unequal()]
        group1 = [we for we in candidates if we.is_char_var_headed() or we.is_var_char_headed() or we.has_emptiness()]
        if len(group1) > 0:
            return group1[0]
        else:
            return None

    def pick_shortest_word_equation(self, side: bool = False) -> Optional[WordEquation]:
        candidates = [we for we in self.word_equations if we != self.success_we and not we.is_simply_unequal()]
        if len(candidates) > 0:
            if side:  # seek the lower side length of a word equation
                candidates_len = [we.min_side_len() for we in candidates]
            else:  # seek the total length of a word equation
                candidates_len = [len(we) for we in candidates]
            min_idx = candidates_len.index(min(candidates_len))
            return candidates[min_idx]
        else:
            return None

    def pick_one_elem_word_equation(self) -> Optional[WordEquation]:  # only one or zero element in one side
        candidates = [we for we in self.word_equations if we != self.success_we and not we.is_simply_unequal()]
        group1 = [we for we in candidates if len(we.lhs) <= 1 or len(we.rhs) <= 1]
        if len(group1) > 0:
            return group1[0]
        else:
            return None


def same_reg_constraints(regc1: Dict[str, int], regc2: Dict[str, int]) -> bool:
    # check for (1) the same set of keys (2) for each key, classifications of FSA are the same
    return regc1.keys() == regc2.keys() and \
           sum([1 for e in regc1 if e in regc2 and regc1[e] == regc2[e]]) == len(regc1.keys())


class Transform:
    def __init__(self, source: SolveTreeNode, rewrite: Rewrite,
                 record: TransformRecord):
        self.source: SolveTreeNode = source
        self.rewrite: Rewrite = rewrite
        self.record = record

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.source == other.source and
                    self.rewrite == other.rewrite and
                    self.record == other.record)
        return False

    def __str__(self):
        return f'{self.source}, {self.rewrite}, {self.record}'

    def __repr__(self):
        return f'{repr(self.source)}, {repr(self.rewrite)}, {repr(self.record)}'

    def __hash__(self):
        return hash(repr(self))


# this function can be removed later
def solution_word_equations(wes: List[WordEquation]) -> bool:
    success_we: WordEquation = WordEquation([], [])
    return reduce(lambda x, y: x and y, map(lambda x: x == success_we, wes))


class SolveTree:
    # success_end: WordEquation = WordEquation([], [])

    def __init__(self, root: SolveTreeNode):
        self.root: SolveTreeNode = root
        self.node_relations: Dict[SolveTreeNode, Set[Transform]] = {}

    def has_node(self, node: SolveTreeNode):
        return node in self.node_relations

    def has_solution(self) -> bool:
        # return reduce(lambda x, y: x or y,
        #               [node.is_success_node() for node in self.node_relations.keys()])
        return len([node for node in self.node_relations.keys() if node.is_success_node()]) > 0

    def get_solution_node(self) -> Set[SolveTreeNode]:
        return set(e for e in self.node_relations if e.is_success_node())

    def add_node(self, src: SolveTreeNode, node: SolveTreeNode, rewrite: Rewrite,
                 record: TransformRecord) -> bool:
        transform = Transform(src, rewrite, record)
        if self.has_node(node):
            self.node_relations[node].add(transform)
            return False
        else:
            self.node_relations[node] = {transform}
            return True  # True means a new node relation is created

    def num_transitions(self):
        return sum(map(len, self.node_relations.values()))

    def num_nodes(self):
        return len(self.node_relations.keys() |
                   {t.source for e in self.node_relations for t in self.node_relations[e]})
        # all_nodes = set()
        # for node in self.node_relations:
        #     all_nodes.add(node)
        #     for t in self.node_relations[node]:
        #         all_nodes.add(t.source)
        # return len(all_nodes)


class InvalidProblemError(Exception):
    pass


def node_with_new_reg_constraints(wes: List[WordEquation], regc_old: RegConstraints, fsa: [FSA],
                                  var_name: str, process_type: str = 'copy') -> [SolveTreeNode]:
    regc_new: RegConstraints = dict()
    if process_type == 'check':
        for r in regc_old:
            if r == var_name:  # check {var_name}'s fsa inclusion, then update
                tmp_fsa = regc_old[r].intersect(fsa)
                tmp_fsa.prune()
                if tmp_fsa.is_empty():  # inclusion is empty, transform failed
                    return None
                else:  # inclusion ok, update
                    regc_new[r] = tmp_fsa
            else:  # other variables, copy
                regc_new[r] = regc_old[r]
        if var_name not in regc_old:  # in case {var_name} has no regular constraint yet, do update
            regc_new[var_name] = fsa
    elif process_type == 'copy':
        for r in regc_old:
            regc_new[r] = regc_old[r]
    elif process_type == 'update':  # do copy first, then update {var_name}'s fsa
        for r in regc_old:
            regc_new[r] = regc_old[r]
        regc_new[var_name] = fsa  # update after for-loop, in case {var_name} has no regular constraint yet
    return SolveTreeNode(wes, regc_new)


class BasicSolver:
    def __init__(self, prob: Problem):
        if len(prob.word_equations) < 1:
            raise InvalidProblemError()
        node = SolveTreeNode([e.trim_prefix() for e in prob.word_equations],
                             prob.merge_reg_constraints())  # root node
        self.pending_checks: List[SolveTreeNode] = [node]
        self.resolve: SolveTree = SolveTree(node)
        if prob.has_reg_constraints():  # has membership(regular) constraints
            self.alphabet = list(node.reg_constraints.values())[0].alphabet
            self.empty_str_fsa = from_str('', self.alphabet)
            self.fsa_classes: FsaClassification = fsa_classification
        else:
            self.alphabet = None
            self.empty_str_fsa = None
        self.problem = prob  # reference to problem data (for on_the_fly_quadratic)
        self.strategy = Strategy.first  # word equation selection strategies
        self.disable_var_empty_transform = False  # special flag
        self.on_the_fly_quadratic = False  # incrementally turning word equations to quadratic during transform
        self.debug = False  # for printing debug info, False by default

    def transform_with_emptiness(self, node: SolveTreeNode, we: WordEquation):
        if self.disable_var_empty_transform:
            return
        lh, rh = hh = we.peek()
        if (not lh or is_del(lh)) and rh and is_var(rh):
            new_wes = [e.remove_element_from_all(rh).trim_prefix() for e in node.word_equations]
            self.process_reg_constraints(node, new_wes, Rewrite.rvar_be_empty, hh)
        elif (not rh or is_del(rh)) and lh and is_var(lh):
            new_wes = [e.remove_element_from_all(lh).trim_prefix() for e in node.word_equations]
            self.process_reg_constraints(node, new_wes, Rewrite.lvar_be_empty, hh)
        else:
            assert False

    def transform_both_var_case(self, node: SolveTreeNode, we: WordEquation):
        lh, rh = hh = we.peek()

        if not self.disable_var_empty_transform:
            case1_wes = [e.remove_element_from_all(lh).trim_prefix() for e in node.word_equations]
            self.process_reg_constraints(node, case1_wes, Rewrite.lvar_be_empty, hh)

            case2_wes = [e.remove_element_from_all(rh).trim_prefix() for e in node.word_equations]
            self.process_reg_constraints(node, case2_wes, Rewrite.rvar_be_empty, hh)

        case3_wes = [e.replace(lh, rh).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case3_wes, Rewrite.lvar_be_rvar, hh)

        if self.on_the_fly_quadratic:
            if node.var_occurrence(lh) > 2:
                print(f'perform incremential quadratic: {lh}')
                turn_one_var_to_quadratic(lh, node, self.problem)
                print(print_solve_tree_node_pretty(node))
        case4_wes = [e.replace_with(lh, [rh, lh]).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case4_wes, Rewrite.lvar_longer_var, hh)

        if self.on_the_fly_quadratic:
            if node.var_occurrence(rh) > 2:
                print(f'perform incremential quadratic: {rh}')
                turn_one_var_to_quadratic(rh, node, self.problem)
                print(print_solve_tree_node_pretty(node))
        case5_wes = [e.replace_with(rh, [lh, rh]).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case5_wes, Rewrite.rvar_longer_var, hh)

    def transform_char_var_case(self, node: SolveTreeNode, we: WordEquation):
        lh, rh = hh = we.peek()

        if not self.disable_var_empty_transform:
            case1_wes = [e.remove_element_from_all(rh).trim_prefix() for e in node.word_equations]
            self.process_reg_constraints(node, case1_wes, Rewrite.rvar_be_empty, hh)

        case2_wes = [e.replace(rh, lh).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case2_wes, Rewrite.rvar_be_char, hh)

        case3_wes = [e.replace_with(rh, [lh, rh]).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case3_wes, Rewrite.rvar_longer_char, hh)

    def transform_var_char_case(self, node: SolveTreeNode, we: WordEquation):
        lh, rh = hh = we.peek()

        if not self.disable_var_empty_transform:
            case1_we = [e.remove_element_from_all(lh).trim_prefix() for e in node.word_equations]
            self.process_reg_constraints(node, case1_we, Rewrite.lvar_be_empty, hh)

        case2_we = [e.replace(lh, rh).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case2_we, Rewrite.lvar_be_char, hh)

        case3_we = [e.replace_with(lh, [rh, lh]).trim_prefix() for e in node.word_equations]
        self.process_reg_constraints(node, case3_we, Rewrite.lvar_longer_char, hh)

    def transform_all_vars_empty(self, node: SolveTreeNode):
        # perform transform towards each element of the powerset of variables of given node
        variables = node.variables()
        vars_ps = powerset(variables)
        for vars_tuple in vars_ps:
            # perform element removal towards all variables of each tuple, then proceed to construction of solve tree
            if self.debug:
                print(f'vars_tuple: {vars_tuple}')
            wes = [e.remove_element_from_all(vars_tuple[0]).trim_prefix() for e in node.word_equations]
            for i in range(1, len(vars_tuple)):
                wes = [e.remove_element_from_all(vars_tuple[i]).trim_prefix() for e in wes]
            self.process_reg_constraints(node, wes, Rewrite.lvar_be_empty, (None, None))
        if self.debug:
            print('transform powerset of all variables empty at first.')
            print(f'    number of veriables: {len(variables)}')
            print(f'    number of pending check nodes: {len(self.pending_checks)}')
            print(f'    current number of nodes: {self.resolve.num_nodes()}')
            nodes = [node for node in self.pending_checks if not node.is_unsolvable_node()]
            for n in nodes:
                print(f'{print_solve_tree_node_pretty(n)}\n')
            input('pause... press enter to continue')

    def process_reg_constraints(self, node: SolveTreeNode, wes: List[WordEquation], rewrite: Rewrite,
                                record: TransformRecord):
        if not node.reg_constraints:  # if no regular constraints at first, don't process regular constraints
            new_node = SolveTreeNode(wes)
            # print(f'process_reg_constratins: new_node: {print_solve_tree_node_pretty(new_node)}')
            self.update_solve_tree(node, new_node, rewrite, record)
            return  # case end: no regular constraints

        # process regular constraints according to rewrite cases and construct {regc_new}
        regc: RegConstraints = node.reg_constraints
        if rewrite == Rewrite.lvar_be_empty:
            # check inclusion of empty fsa for {lvar}
            fsa_tmp = self.empty_str_fsa
            var_name = record[0].value
            new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_name, 'check')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.rvar_be_empty:
            # check inclusion of empty fsa for {rvar}
            fsa_tmp = self.empty_str_fsa
            var_name = record[1].value
            new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_name, 'check')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.lvar_be_char:
            # check inclusion of a fsa accepting only one char for {lvar}
            var_name, ch = record[0].value, record[1].value
            fsa_tmp = from_str(ch, self.alphabet)
            new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_name, 'check')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.rvar_be_char:
            # check inclusion of a fsa accepting only one char for {rvar}
            var_name, ch = record[1].value, record[0].value
            fsa_tmp = from_str(ch, self.alphabet)
            new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_name, 'check')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.lvar_be_rvar:
            # check inclusion of a fsa accepting {rvar} for {lvar}
            var_l, var_r = record[0].value, record[1].value
            if var_r in regc:  # if {rvar} has regular constraint, check inclusion
                fsa_tmp = regc[var_r]
                new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_l, 'check')
            else:  # if {rvar} has no regular constraint, just copy
                new_node = node_with_new_reg_constraints(wes, regc, None, var_l, 'copy')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.lvar_longer_char:
            # get a new fsa by removing the first char {rvar} from the fsa of {lvar}
            var_name, ch = record[0].value, record[1].value
            if var_name in regc:
                fsa_tmp = remove_first_char(regc[var_name], ch)
                if fsa_tmp:
                    new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_name, 'update')
                else:
                    return  # transform failed (constraint violation)
            else:
                new_node = node_with_new_reg_constraints(wes, regc, None, var_name, 'copy')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.rvar_longer_char:
            # get a new fsa by removing the first char {lvar} from the fsa of {rvar}
            var_name, ch = record[1].value, record[0].value
            if var_name in regc:
                fsa_tmp = remove_first_char(regc[var_name], ch)
                if fsa_tmp:
                    new_node = node_with_new_reg_constraints(wes, regc, fsa_tmp, var_name, 'update')
                else:
                    return  # transform failed (constraint violation)
            else:
                new_node = node_with_new_reg_constraints(wes, regc, None, var_name, 'copy')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.lvar_longer_var:
            # get a set of pair of fsa for ({rvar},{lvar}) by splitting the constraint of {lvar}
            var_l, var_r = record[0].value, record[1].value
            new_node = None
            if var_l in regc:  # {var_l} has regular constraint, need to do split
                fsa_paris = split_by_states(regc[var_l])
                for fsa_r, fsa_l in fsa_paris:
                    new_node = node_with_new_reg_constraints(wes, regc, fsa_l, var_l, 'check')
                    if new_node:
                        # this function call will update fsa of {var_r} if it has no regular constraint yet
                        new_node = node_with_new_reg_constraints(wes, new_node.reg_constraints, fsa_r, var_r, 'check')
            else:  # no need to update/check regular constraints
                new_node = node_with_new_reg_constraints(wes, regc, None, var_l, 'copy')
            self.update_solve_tree(node, new_node, rewrite, record)
        elif rewrite == Rewrite.rvar_longer_var:
            # get a set of pair of fsa for ({rvar},{lvar}) by splitting the constraint of {rvar}
            var_l, var_r = record[0].value, record[1].value
            new_node = None
            if var_r in regc:  # {var_r} has regular constraint, need to do split
                fsa_paris = split_by_states(regc[var_r])
                for fsa_l, fsa_r in fsa_paris:
                    new_node = node_with_new_reg_constraints(wes, regc, fsa_r, var_r, 'check')
                    if new_node:
                        # this function call will update fsa of {var_r} if it has no regular constraint yet
                        new_node = node_with_new_reg_constraints(wes, new_node.reg_constraints, fsa_l, var_l, 'check')
            else:  # no need to update/check regular constraints
                new_node = node_with_new_reg_constraints(wes, regc, None, var_r, 'copy')
            self.update_solve_tree(node, new_node, rewrite, record)

    def update_solve_tree(self, node: SolveTreeNode, new_node: Optional[SolveTreeNode], rewrite: Rewrite,
                          record: TransformRecord):
        if self.debug:
            print('----- debug info -----')
            print(f'from node:\n{print_solve_tree_node_simple(node)}')
            if new_node:
                print(f'to node:\n{print_solve_tree_node_simple(new_node)}')
            else:
                print('to node: None')
            print(f'rewrite: {rewrite}')
            print(f'head: {record}')
        if new_node:
            if len(node) < len(new_node):  # len returns the total length of all word equations of a node
                print("Warning: word equation is not quadratic")
                print(f'old we: length = {len(node)}\n{node.word_equations}')
                print(f'new we: length = {len(new_node)}\n{new_node.word_equations}')
                exit(1)
            if self.resolve.add_node(node, new_node, rewrite, record):
                self.pending_checks.append(new_node)
                if self.debug:
                    print('new node')
            else:
                if self.debug:
                    print('existing node')

    def process_node(self, curr_node: SolveTreeNode, curr_we: WordEquation) -> SolveTree:
        if curr_we.has_emptiness():
            if self.debug:
                print('transform case: has emptiness')
            self.transform_with_emptiness(curr_node, curr_we)
        elif curr_we.is_both_var_headed():
            if self.debug:
                print('transform case: variable-variable')
            self.transform_both_var_case(curr_node, curr_we)
        elif curr_we.is_char_var_headed():
            if self.debug:
                print('transform case: char-variable')
            self.transform_char_var_case(curr_node, curr_we)
        elif curr_we.is_var_char_headed():
            if self.debug:
                print('transform case: variable-char')
            self.transform_var_char_case(curr_node, curr_we)
        else:
            assert False
        return self.resolve

    def solve_full(self, node: SolveTreeNode):
        # strategy 1: proceed transform on all word equations of a node
        wes = node.get_word_equations_to_solve()
        for we in wes:
            self.process_node(node, we)
            if self.debug:
                print(f'number of pending check nodes: {len(self.pending_checks)}')
                print(f'current number of nodes: {self.resolve.num_nodes()}')
                # input('pause... press enter to continue')
        if self.debug:
            # print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')

    def solve_first(self, node: SolveTreeNode):
        we = node.pick_first_word_equation()
        if not we:  # this shall not happen (supposed to be filtered)
            assert False
        self.process_node(node, we)
        if self.debug:
            print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')
            # input('pause... press enter to continue')

    def solve_var_char_first(self, node: SolveTreeNode):
        we = node.pick_var_char_word_equation()
        if not we:
            we = node.pick_first_word_equation()
        if not we:  # this shall not happen (supposed to be filtered)
            assert False
        self.process_node(node, we)
        if self.debug:
            print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')
            # input('pause... press enter to continue')

    def solve_shortest(self, node: SolveTreeNode):
        we = node.pick_shortest_word_equation()
        if not we:
            we = node.pick_first_word_equation()
        if not we:  # this shall not happen (supposed to be filtered)
            assert False
        self.process_node(node, we)
        if self.debug:
            print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')
            # input('pause... press enter to continue')

    def solve_shortest_side(self, node: SolveTreeNode):
        we = node.pick_shortest_word_equation(side=True)
        if not we:
            we = node.pick_first_word_equation()
        if not we:  # this shall not happen (supposed to be filtered)
            assert False
        self.process_node(node, we)
        if self.debug:
            print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')
            # input('pause... press enter to continue')

    def solve_one_elem_first(self, node: SolveTreeNode):
        we = node.pick_one_elem_word_equation()
        if not we:
            we = node.pick_first_word_equation()
        if not we:  # this shall not happen (supposed to be filtered)
            assert False
        self.process_node(node, we)
        if self.debug:
            print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')
            # input('pause... press enter to continue')

    def solve_customized(self, node: SolveTreeNode):  # one element side first, then shortest side
        we = node.pick_one_elem_word_equation()
        if not we:
            we = node.pick_shortest_word_equation(side=True)
        if not we:
            we = node.pick_first_word_equation()
        if not we:  # this shall not happen (supposed to be filtered)
            assert False
        self.process_node(node, we)
        if self.debug:
            print(f'number of pending check nodes: {len(self.pending_checks)}')
            print(f'current number of nodes: {self.resolve.num_nodes()}')
            # input('pause... press enter to continue')

    def solve(self):
        while self.pending_checks:
            node = self.pending_checks.pop(0)
            if node.is_success_node():
                if self.debug:
                    print('transform case: success node')
                    print(print_solve_tree_node_pretty(node))
                continue
            elif node.is_unsolvable_node():
                if self.debug:
                    print('transform case: unsolvable node')
                    print(print_solve_tree_node_pretty(node))
                continue
            # separate different strategies
            if self.strategy == Strategy.full:
                self.solve_full(node)
            elif self.strategy == Strategy.first:
                self.solve_first(node)
            elif self.strategy == Strategy.shortest:
                self.solve_shortest(node)
            elif self.strategy == Strategy.shortest_side:
                self.solve_shortest_side(node)
            elif self.strategy == Strategy.one_elem_first:
                self.solve_one_elem_first(node)
            elif self.strategy == Strategy.var_char_first:
                self.solve_var_char_first(node)
            elif self.strategy == Strategy.empty_vars_first:
                self.transform_all_vars_empty(node)  # only do this once on root node
                self.pending_checks.append(node)  # root node should be processed with normal transform rules
                self.disable_var_empty_transform = True
                self.strategy = Strategy.first  # use strategy first for the rest nodes
            elif self.strategy == Strategy.customized:
                self.solve_customized(node)
            else:
                assert False


# powerset without empty set, returns list of tuples
def powerset(iterable):  # powerset({1,2,3}) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


# functions for turn word equations to linear/quadratic form
def turn_to_linear_wes(prob: Problem, wes: Optional[List[WordEquation]] = None):
    tgt_wes = wes or prob.word_equations
    occurred_var: Set[StrElement] = set()

    def _turn_to_linear(expr):
        nonlocal occurred_var
        for index, e in enumerate(expr):
            if e in occurred_var:
                var_copy_name = prob.new_variable(ValueType.string, e.value)
                e_copy = StrVariable(var_copy_name)
                expr[index] = e_copy
                lc = LengthConstraint([e.length()], [e_copy.length()])
                prob.add_length_constraint(lc)
                reg_cons = prob.reg_constraints
                if e.value in reg_cons:
                    reg_cons[e_copy.value] = reg_cons[e.value]
            elif is_var(e):
                occurred_var.add(e)

    for we in tgt_wes:
        _turn_to_linear(we.lhs)
        _turn_to_linear(we.rhs)


# rename a string variable if it occurs more than twice in a node
def turn_one_var_to_quadratic(var: StrVariable, node: SolveTreeNode, prob: Problem):
    if node.var_occurrence(var) <= 2:  # do nothing if no more than two occurrences
        return

    var_occurred_count: int = 0
    var_new_name: str = ''
    new_var: Optional[StrVariable] = None

    def _replace(expr):
        nonlocal var_occurred_count, var_new_name, new_var
        for index, e in enumerate(expr):
            if e == var:
                var_occurred_count += 1
                if var_occurred_count > 2:
                    # first time of replace, generate new var name and add corresponding length and regular constraints
                    if var_new_name == '' and not new_var:
                        var_original_name = internal_str_var_origin_name(var.value)
                        if var_original_name:
                            var_new_name = prob.new_variable(ValueType.string, var_original_name)
                        else:
                            var_new_name = prob.new_variable(ValueType.string, var.value)
                        new_var = StrVariable(var_new_name)
                        lc = LengthConstraint([var.length()], [new_var.length()])
                        prob.add_length_constraint(lc)
                        reg_cons = prob.reg_constraints
                        if var.value in reg_cons:
                            reg_cons[new_var.value] = reg_cons[var.value]
                    expr[index] = new_var

    for we in node.word_equations:
        _replace(we.lhs)
        _replace(we.rhs)


def turn_to_quadratic_wes(prob: Problem, wes: Optional[List[WordEquation]] = None):
    tgt_wes = wes or prob.word_equations
    occurred_var_count: Dict[StrElement, int] = dict()
    curr_var_copies: Dict[StrElement, StrElement] = dict()

    def _turn_to_quadratic(expr):
        nonlocal occurred_var_count, curr_var_copies
        for index, e in enumerate(expr):
            if e in occurred_var_count:
                occurred_var_count[e] += 1
                if occurred_var_count[e] % 2 == 1:
                    var_copy_name = prob.new_variable(ValueType.string, e.value)
                    e_copy = StrVariable(var_copy_name)
                    curr_var_copies[e] = e_copy
                    lc = LengthConstraint([e.length()], [e_copy.length()])
                    prob.add_length_constraint(lc)
                    reg_cons = prob.reg_constraints
                    if e.value in reg_cons:
                        reg_cons[e_copy.value] = reg_cons[e.value]
                expr[index] = curr_var_copies[e]
            elif is_var(e):
                occurred_var_count[e] = 1
                curr_var_copies[e] = e

    for we in tgt_wes:
        _turn_to_quadratic(we.lhs)
        _turn_to_quadratic(we.rhs)


# functions for output: pretty print, c program, graphviz, etc.
def print_word_equation_pretty(we: WordEquation) -> str:
    left_str = ''.join(
        [f'[{e.value}]' if is_var(e) else e.value if is_char(e) else '$' for e in we.lhs]) or '\"\"'
    right_str = ''.join(
        [f'[{e.value}]' if is_var(e) else e.value if is_char(e) else '$' for e in we.rhs]) or '\"\"'
    return f'{left_str} = {right_str}'


def print_word_equation_list_pretty(wes: List[WordEquation]) -> str:
    return '\n'.join([print_word_equation_pretty(e) for e in wes])


def print_reg_constraints_pretty(node: SolveTreeNode, indent: str = '') -> str:
    if node.reg_constraints:
        return '\n\n'.join([f'{indent}{s}:({str(node.regc_classes[s])}):\n' +
                            indent + str(node.reg_constraints[s]).replace('\n', '\n' + indent)
                            for s in sorted(node.reg_constraints)])
    else:
        return ''


def print_reg_constraints_simple(node: SolveTreeNode) -> str:
    if node.regc_classes:
        return '-'.join([f'{s}({str(node.regc_classes[s])})' for s in sorted(node.regc_classes)])
    else:
        return ''


def print_solve_tree_node_pretty(node: SolveTreeNode, indent: str = '') -> str:
    wes_str = indent + '\n'.join([print_word_equation_pretty(e) for e in node.word_equations])
    if node.reg_constraints:
        return f'{wes_str}:\n{print_reg_constraints_pretty(node, indent*2)}'
    else:
        return wes_str


def print_solve_tree_node_simple(node: SolveTreeNode, indent: str = '') -> str:
    wes_str = indent + '\n'.join([print_word_equation_pretty(e) for e in node.word_equations])
    if node.regc_classes:
        return f'{wes_str}:\n{indent*2}{print_reg_constraints_simple(node)}'
    else:
        return wes_str


def print_tree_plain(tree: SolveTree):
    print(f'{tree.root}: ')
    cnt_node = 1
    for k in tree.node_relations.keys():
        print(f'{cnt_node}  {k}')
        cnt_node += 1
        cnt = 1
        for t in tree.node_relations[k]:
            print(f'    {cnt}  {t})')
            cnt += 1


def print_transform_rewrite_pretty(trans: Transform) -> str:
    if trans.record[0]:
        lval = trans.record[0].value
    else:
        lval = '\"\"'
    if trans.record[1]:
        rval = trans.record[1].value
    else:
        rval = '\"\"'
    if trans.rewrite == Rewrite.lvar_be_empty:
        return f'{lval}=\"\"'
    elif trans.rewrite == Rewrite.rvar_be_empty:
        return f'{rval}=\"\"'
    elif trans.rewrite == Rewrite.lvar_be_char:
        return f'{lval}={rval}'
    elif trans.rewrite == Rewrite.rvar_be_char:
        return f'{rval}={lval}'
    elif trans.rewrite == Rewrite.lvar_be_rvar:
        return f'{lval}={rval}'
    elif trans.rewrite == Rewrite.lvar_longer_char:
        return f'{lval}={rval}{lval}'
    elif trans.rewrite == Rewrite.rvar_longer_char:
        return f'{rval}={lval}{rval}'
    elif trans.rewrite == Rewrite.lvar_longer_var:
        return f'{lval}={rval}{lval}'
    elif trans.rewrite == Rewrite.rvar_longer_var:
        return f'{rval}={lval}{rval}'


def print_transform_rewrite_length(trans: Transform) -> str:
    if trans.record[0]:
        lval = trans.record[0].value
    else:
        lval = '\"\"'
    if trans.record[1]:
        rval = trans.record[1].value
    else:
        rval = '\"\"'
    if trans.rewrite == Rewrite.lvar_be_empty:
        return f'{lval}=0'
    elif trans.rewrite == Rewrite.rvar_be_empty:
        return f'{rval}=0'
    elif trans.rewrite == Rewrite.lvar_be_char:
        return f'{lval}=1'
    elif trans.rewrite == Rewrite.rvar_be_char:
        return f'{rval}=1'
    elif trans.rewrite == Rewrite.lvar_be_rvar:
        return f'{lval}={rval}'
    elif trans.rewrite == Rewrite.lvar_longer_char:
        return f'{lval}={lval}+1'
    elif trans.rewrite == Rewrite.rvar_longer_char:
        return f'{rval}={rval}+1'
    elif trans.rewrite == Rewrite.lvar_longer_var:
        return f'{lval}={lval}+{rval}'
    elif trans.rewrite == Rewrite.rvar_longer_var:
        return f'{rval}={rval}+{lval}'


def print_tree_pretty(tree: SolveTree, max_num: int = 0):
    print(f'word equation: {print_word_equation_pretty(tree.root.word_equation)}\n')
    print(f'regular constraints:\n{print_reg_constraints_pretty(tree.root)}\n')
    cnt_node = 1
    for k in tree.node_relations:
        if max_num > 0:
            if cnt_node > max_num:
                return
        print(f'node{cnt_node}:\n')
        print(print_solve_tree_node_pretty(k))
        cnt_node += 1
        cnt = 1
        for t in tree.node_relations[k]:
            print(
                f'    child{cnt}\n'
                f'        rewrite: {print_transform_rewrite_pretty(t)}\n\n'
                f'    {print_solve_tree_node_pretty(t.source, " "*4)}\n')
            cnt += 1


def print_tree_simple(tree: SolveTree, max_num: int = 0):
    print(f'word equation: {print_word_equation_pretty(tree.root.word_equation)}\n')
    print(f'regular constraints:\n{print_reg_constraints_simple(tree.root)}\n')
    cnt_node = 1
    for k in tree.node_relations:
        if max_num > 0:
            if cnt_node > max_num:
                return
        print(f'node{cnt_node}:\n')
        print(print_solve_tree_node_simple(k))
        cnt_node += 1
        cnt = 1
        for t in tree.node_relations[k]:
            print(
                f'    child{cnt}\n'
                f'        rewrite: {print_transform_rewrite_pretty(t)}\n'
                f'    {print_solve_tree_node_simple(t.source, " "*4)}\n')
            cnt += 1


def print_tree_dot_pretty(tree: SolveTree) -> str:
    # we_str = print_word_equation_pretty(tree.root.word_equation).replace('=', '-')
    name = f'tree_obj_{id(tree)}'
    # if not tree.has_solution():
    #    print('no solution for word equation {we_str}')

    dot = Digraph(name=name, comment=name)
    for k in tree.node_relations.keys():
        node_str = print_word_equation_list_pretty(k.word_equations) + '\n' +\
                   print_reg_constraints_simple(k)
        dot.node(node_str, node_str)
        for r in tree.node_relations[k]:
            next_node_str = print_word_equation_list_pretty(r.source.word_equations) + '\n' +\
                            print_reg_constraints_simple(r.source)
            dot.edge(node_str, next_node_str, print_transform_rewrite_pretty(r))
    print(dot.source)
    dot.render()
    return name


def print_tree_c_program(tree: SolveTree, code_type: str, problem: Problem) -> str:  # returns the filename
    # check type validity
    if code_type != 'interProc' and code_type != 'UAutomizerC' and code_type != 'EldaricaC':
        print(
            'Type Error: type should be specified to \"interProc\" or \"UAutomizerC\" or \"EldaricaC\"')
        print('No c program output...')
        return

    # set some syntax keywords according to type
    if code_type == 'interProc':
        prog_start = 'begin'
        prog_end = 'end'
        while_start = 'do'
        while_end = 'done;'
        if_start = ' then'
        if_end = 'endif;'
        random_decl = '      rdn = random;'
        random_final = 'reachFinal = random;\n'
    elif code_type == 'UAutomizerC' or code_type == 'EldaricaC':
        prog_start = ''
        prog_end = '}'
        while_start = '{'
        while_end = '}'
        if_start = ' {'
        if_end = '}'
        random_decl = '      rdn =  __VERIFIER_nondet_int();\n'
        random_final = 'reachFinal = __VERIFIER_nondet_int();\n'

    # preprocessing, middle variables declaration
    trans = tree.node_relations
    visited_node = set()
    node2_count = dict()
    queued_node = set()
    variables = set()
    for var in problem.variables:
        if problem.variables[var] == ValueType.string or \
                (problem.variables[var] == ValueType.int and not internal_len_var_name.match(var)):
            variables.add(var)
    # for s in trans.keys():
    #     for t in s.word_equation.variables():
    #         variables.add(t)
    # for e in int_vars:
    #     if not length_origin_name(e):
    #         variables.add(e)  # add non-length variables in length constraints
    node_count = 0

    # open a file for writing code
    #filename = f'{print_word_equation_pretty(tree.root.word_equation).replace("=", "-")}_{type}.c'
    filename = f'tree_obj_{id(tree)}_{code_type}.c'
    fp = open(filename, "w")

    # variable declaration
    if code_type == 'interProc':
        fp.write('var \n')
        for s in variables:
            fp.write(f'{s.value}: int,\n')
        fp.write('rdn: int,\n')
        fp.write('nodeNo: int,\n')
        fp.write('reachFinal: int;\n')
    elif code_type == 'UAutomizerC':
        fp.write('extern void __VERIFIER_error() __attribute__ ((__noreturn__));\n')
        fp.write('extern int __VERIFIER_nondet_int(void);\n')
        fp.write('\n')
        fp.write('int main() {\n')
        for s in variables:
            fp.write(f'  int {s};\n')
        fp.write('  int rdn, nodeNo, reachFinal;\n')
    elif code_type == 'EldaricaC':
        fp.write('int __VERIFIER_nondet_int(void) { int n=_; return n; }\n')
        fp.write('\n')
        fp.write('int main() {\n')
        for s in variables:
            fp.write(f'  int {s};\n')
        fp.write('  int rdn, nodeNo, reachFinal;\n')

    # program begins
    fp.write(prog_start)
    fp.write(f'  nodeNo = {node_count};\n')  # set nodeNo to zero (initial node)
    fp.write('  reachFinal = 0;\n')
    fp.write(f'  while (1) {while_start}\n')
    # start traverse from init node to final node
    init = tree.get_solution_node()
    final = tree.root
    [queued_node.add(s) for s in init]
    while len(queued_node) > 0:
        tmp_node = queued_node.pop()
        # cases of node
        if tmp_node in visited_node:  # already processed: skip to next loop
            continue
        else:
            visited_node.add(tmp_node)

        if tmp_node in init:  # this is the initial node
            fp.write(f'    if (nodeNo=={node_count}) {if_start}\n')
            # node_count = 0 (the first loop)
            fp.write(f'    /* node = {print_word_equation_list_pretty(tmp_node.word_equations)} */\n')
        else:
            fp.write(f'    if (nodeNo=={node2_count[tmp_node]}) {if_start}\n')
            # node2_count must has key "tmp_node"
            fp.write(f'    /* node = {print_word_equation_list_pretty(tmp_node.word_equations)} */\n')
            if tmp_node == final:  # this is the final node
                if tmp_node in trans:  # final node has transition
                    fp.write(f'      {random_final}')
                    fp.write(f'      if (reachFinal >= 0) {if_start} /* final node */\n')
                    fp.write('        break;\n')
                    fp.write(f'      {if_end}\n')
                else:
                    fp.write('      break;\n')
                    fp.write(f'    {if_end}\n')
                    continue

        tmp_labl = trans[tmp_node]
        tmp_len = len(tmp_labl)

        if tmp_len > 1:  # two or more parent nodes # currently not completed
            fp.write(random_decl)
            # print "      assume rdn>=1 and rdn <=" + str(tmp_len) + ';'
            rdn_count = 1  # start from 1
            for s in tmp_labl:
                if rdn_count == 1:
                    fp.write(f'      if (rdn<={rdn_count}) {if_start}\n')
                elif rdn_count == tmp_len:
                    fp.write(f'      if (rdn>={rdn_count}) {if_start}\n')
                else:
                    fp.write(f'      if (rdn=={rdn_count}) {if_start}\n')
                fp.write(f'        {print_transform_rewrite_length(s)};\n')
                fp.write(f'        // {print_transform_rewrite_pretty(s)};\n')
                # information for retrieving solution
                if s.source in node2_count:
                    fp.write(f'        nodeNo={node2_count[s.source]};\n')
                else:
                    node_count += 1
                    fp.write(f'        nodeNo={node_count};\n')
                    node2_count[s.source] = node_count
                queued_node.add(s.source)
                fp.write(f'      {if_end}\n')
                rdn_count += 1
        else:
            for s in tmp_labl:
                fp.write(f'      {print_transform_rewrite_length(s)};\n')
                fp.write(f'      // {print_transform_rewrite_pretty(s)};\n')
                # information for retrieving solution
                if s.source in node2_count:
                    fp.write(f'      nodeNo={node2_count[s.source]};\n')
                else:
                    node_count += 1
                    fp.write(f'      nodeNo={node_count};\n')
                    node2_count[s.source] = node_count
                queued_node.add(s.source)

        fp.write(f'    {if_end}\n')
    fp.write(f'  {while_end}\n')
    length_cons = print_length_constraints_as_strings(problem.len_constraints)
    if length_cons:
        if len(length_cons) == 1:
            lc = length_cons[0]
        else:  # multiple length constraints, take conjunction
            lc = ' && '.join(length_cons)
    if code_type == "UAutomizerC" and length_cons:
        # length constraint (for UAutomizer)
        fp.write(f'  if ({lc}) {{ //length constraint: {length_cons}\n')
        fp.write('    ERROR: __VERIFIER_error();\n')
        fp.write('  }\n')
        fp.write('  else {\n')
        fp.write('    return 0;\n')
        fp.write('  }\n')
    if code_type == "EldaricaC" and length_cons:  # length constraint (for Eldarica)
        fp.write(f'  assert (!({lc})); //length constraint: {length_cons}\n')
    fp.write(prog_end)

    fp.close()
    return filename
