from enum import Enum, unique, auto
from functools import reduce
from typing import List, Tuple, Dict, Set, Optional

from graphviz import Digraph
from lenc import LengthConstraint
from prob import Problem, ValueType
from we import WordEquation, StrElement, StrVariable, is_var, is_del, not_del


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


TransformRecord = Tuple[Optional[StrElement], Optional[StrElement]]


class Transform:
    def __init__(self, source: WordEquation, rewrite: Rewrite,
                 record: TransformRecord):
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

    def __repr__(self):
        return f'{self.source}, {self.rewrite}, {self.record}'


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
                 record: TransformRecord) -> bool:
        transform = Transform(src, rewrite, record)
        if self.has_node(node):
            self.node_relations[node].add(transform)
            return False
        else:
            self.node_relations[node] = {transform}
            return True


class InvalidProblemError(Exception):
    pass


class BasicSolver:
    def __init__(self, prob: Problem):
        if len(prob.word_equations) < 1:
            raise InvalidProblemError()
        prob.merge_all_word_equations()
        we = prob.word_equations[0]
        self.pending_checks: List[WordEquation] = [we]
        self.resolve: SolveTree = SolveTree(we)

    def transform_with_emptiness(self, we: WordEquation):
        lh, rh = hh = we.peek()
        if (not lh or is_del(lh)) and rh and is_var(rh):
            new_we = we.remove_right_head_from_all().trim_prefix()
            if self.resolve.add_node(we, new_we, Rewrite.rvar_be_empty, hh):
                self.pending_checks.append(new_we)
        elif (not rh or is_del(rh)) and lh and is_var(lh):
            new_we = we.remove_left_head_from_all().trim_prefix()
            if self.resolve.add_node(we, new_we, Rewrite.lvar_be_empty, hh):
                self.pending_checks.append(new_we)
        else:
            assert False

    def transform_both_var_case(self, we: WordEquation):
        lh, rh = hh = we.peek()

        case1 = we.remove_left_head_from_all().trim_prefix()
        if self.resolve.add_node(we, case1, Rewrite.lvar_be_empty, hh):
            self.pending_checks.append(case1)

        case2 = we.remove_right_head_from_all().trim_prefix()
        if self.resolve.add_node(we, case2, Rewrite.rvar_be_empty, hh):
            self.pending_checks.append(case2)

        case3 = we.replace(lh, rh).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case3, Rewrite.lvar_be_rvar, hh):
            self.pending_checks.append(case3)

        case4 = we.replace_with(lh, [rh, lh]).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case4, Rewrite.lvar_longer_var, hh):
            self.pending_checks.append(case4)

        case5 = we.replace_with(rh, [lh, rh]).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case5, Rewrite.rvar_longer_var, hh):
            self.pending_checks.append(case5)

    def transform_char_var_case(self, we: WordEquation):
        lh, rh = hh = we.peek()

        case1 = we.remove_right_head_from_all().trim_prefix()
        if self.resolve.add_node(we, case1, Rewrite.rvar_be_empty, hh):
            self.pending_checks.append(case1)

        case2 = we.replace(rh, lh).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case2, Rewrite.rvar_be_char, hh):
            self.pending_checks.append(case2)

        case3 = we.replace_with(rh, [lh, rh]).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case3, Rewrite.rvar_longer_char, hh):
            self.pending_checks.append(case3)

    def transform_var_char_case(self, we: WordEquation):
        lh, rh = hh = we.peek()

        case1 = we.remove_left_head_from_all().trim_prefix()
        if self.resolve.add_node(we, case1, Rewrite.lvar_be_empty, hh):
            self.pending_checks.append(case1)

        case2 = we.replace(lh, rh).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case2, Rewrite.lvar_be_char, hh):
            self.pending_checks.append(case2)

        case3 = we.replace_with(lh, [rh, lh]).remove_heads().trim_prefix()
        if self.resolve.add_node(we, case3, Rewrite.lvar_longer_char, hh):
            self.pending_checks.append(case3)

    def solve(self) -> SolveTree:
        while self.pending_checks:
            curr_we = self.pending_checks.pop(0)
            if curr_we == self.resolve.success_end:
                pass
            elif curr_we.is_simply_unequal():
                pass
            elif curr_we.has_emptiness():
                self.transform_with_emptiness(curr_we)
            elif curr_we.is_both_var_headed():
                self.transform_both_var_case(curr_we)
            elif curr_we.is_char_var_headed():
                self.transform_char_var_case(curr_we)
            elif curr_we.is_var_char_headed():
                self.transform_var_char_case(curr_we)
            else:
                assert False
        return self.resolve


def turn_to_linear_we(prob: Problem, we: WordEquation = None):
    tgt_we = we or prob.word_equations[0]
    occurred_var: Set[StrElement] = set()

    def _turn_to_linear(expr):
        for index, e in enumerate(expr):
            if e in occurred_var:
                var_copy_name = prob.new_variable(ValueType.string, e.value)
                e_copy = StrVariable(var_copy_name)
                expr[index] = e_copy
                lc = LengthConstraint([e.length()], [e_copy.length()])
                prob.add_length_constraint(lc)
                reg_cons = prob.reg_constraints
                reg_cons_src = prob.reg_constraint_src
                if e.value in reg_cons:
                    reg_cons[e_copy.value] = reg_cons[e.value]
                    reg_cons_src[e_copy.value] = reg_cons_src[e.value]
            elif is_var(e):
                occurred_var.add(e)

    _turn_to_linear(tgt_we.lhs)
    _turn_to_linear(tgt_we.rhs)


def turn_to_quadratic_we(prob: Problem, we: WordEquation = None):
    tgt_we = we or prob.word_equations[0]
    occurred_var_count: Dict[StrElement, int] = dict()
    curr_var_copies: Dict[StrElement, StrElement] = dict()

    def _turn_to_quadratic(expr):
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
                    reg_cons_src = prob.reg_constraint_src
                    if e.value in reg_cons:
                        reg_cons[e_copy.value] = reg_cons[e.value]
                        reg_cons_src[e_copy.value] = reg_cons_src[e.value]
                expr[index] = curr_var_copies[e]
            elif is_var(e):
                occurred_var_count[e] = 1
                curr_var_copies[e] = e

    _turn_to_quadratic(tgt_we.lhs)
    _turn_to_quadratic(tgt_we.rhs)


# functions for output: pretty print, c program, graphviz, etc.
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


def print_word_equation_pretty(we: WordEquation) -> str:
    left_str = ''.join(
        [e.value if not_del(e) else '#' for e in we.lhs]) or '\"\"'
    right_str = ''.join(
        [e.value if not_del(e) else '#' for e in we.rhs]) or '\"\"'
    return f'{left_str}={right_str}'


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


def print_tree_pretty(tree: SolveTree):
    print(f'word equation: {print_word_equation_pretty(tree.root)}')
    cnt_node = 1
    for k in tree.node_relations.keys():
        print(f'node{cnt_node}  {print_word_equation_pretty(k)}')
        cnt_node += 1
        cnt = 1
        for t in tree.node_relations[k]:
            print(
                f'    child{cnt}  {print_word_equation_pretty(t.source)}, {print_transform_rewrite_pretty(t)}')
            cnt += 1


def print_tree_dot_pretty(tree: SolveTree):
    we_str = print_word_equation_pretty(tree.root).replace('=', '-')
    # if not tree.has_solution():
    #    print('no solution for word equation {we_str}')

    dot = Digraph(name=we_str, comment=we_str)
    for k in tree.node_relations.keys():
        node_str = print_word_equation_pretty(k)
        dot.node(node_str, node_str)
        for r in tree.node_relations[k]:
            next_node_str = print_word_equation_pretty(r.source)
            dot.edge(node_str, next_node_str, print_transform_rewrite_pretty(r))
    print(dot.source)
    dot.render()


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


def print_tree_c_program(tree: SolveTree, type: str, lengthCons: List[str]):
    # check type validity
    if type != 'interProc' and type != 'UAutomizerC' and type != 'EldaricaC':
        print(
            'Type Error: type should be specified to \"interProc\" or \"UAutomizerC\" or \"EldaricaC\"')
        print('No c program output...')
        return

    # set some syntax keywords according to type
    if type == 'interProc':
        prog_start = 'begin'
        prog_end = 'end'
        while_start = 'do'
        while_end = 'done;'
        if_start = ' then'
        if_end = 'endif;'
        random_decl = '      rdn = random;'
    elif type == 'UAutomizerC' or type == 'EldaricaC':
        prog_start = ''
        prog_end = '}'
        while_start = '{'
        while_end = '}'
        if_start = ' {'
        if_end = '}'
        random_decl = '      rdn =  __VERIFIER_nondet_int();\n'

    # preprocessing, middle variables declaration
    trans = tree.node_relations
    visited_node = set()
    node2_count = dict()
    queued_node = set()
    variables = set()
    for s in trans.keys():
        for t in s.variables():
            variables.add(t)
    node_count = 0

    # open a file for writing code
    fp = open(
        f'{print_word_equation_pretty(tree.root).replace("=", "-")}_{type}.c',
        "w")

    # variable declaration
    if type == 'interProc':
        fp.write('var \n')
        for s in variables:
            fp.write(f'{s.value}: int,\n')
        fp.write('rdn: int,\n')
        fp.write('nodeNo: int,\n')
        fp.write('reachFinal: int;\n')
    elif type == 'UAutomizerC':
        fp.write(
            'extern void __VERIFIER_error() __attribute__ ((__noreturn__));\n')
        fp.write('extern int __VERIFIER_nondet_int(void);\n')
        fp.write('\n')
        fp.write('int main() {\n')
        for s in variables:
            fp.write(f'  int {s.value};\n')
        fp.write('  int rdn, nodeNo, reachFinal;\n')
    elif type == 'EldaricaC':
        fp.write('int __VERIFIER_nondet_int(void) { int n=_; return n; }\n')
        fp.write()
        fp.write('int main() {\n')
        for s in variables:
            fp.write(f'  int {s.value};\n')
        fp.write('  int rdn, nodeNo, reachFinal;\n')

    # program begins
    fp.write(prog_start)
    fp.write(f'  nodeNo = {node_count};\n')  # set nodeNo to zero (initial node)
    fp.write('  reachFinal = 0;\n')
    fp.write(f'  while (reachFinal==0) {while_start}\n')
    # start traverse from init node to final node
    init = SolveTree.success_end
    final = tree.root
    queued_node.add(init)
    while len(queued_node) > 0:
        tmp_node = queued_node.pop()
        # cases of node
        if tmp_node in visited_node:  # already processed: skip to next loop
            continue
        else:
            visited_node.add(tmp_node)

        if tmp_node == init:  # this is the initial node
            fp.write(f'    if (nodeNo=={node_count}) {if_start}\n')
            # node_count = 0 (the first loop)
            fp.write(
                f'    /* node = {print_word_equation_pretty(tmp_node)} */\n')
        else:
            fp.write(f'    if (nodeNo=={node2_count[tmp_node]}) {if_start}\n')
            # node2_count must has key "tmp_node"
            fp.write(
                f'    /* node = {print_word_equation_pretty(tmp_node)} */\n')
            if tmp_node == final:  # this is the final node
                fp.write('      reachFinal=1;\n')
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
    if lengthCons:
        if len(lengthCons) == 1:
            lc = lengthCons[0]
        else:  # multiple length constraints, take conjunction
            lc = ' && '.join(lengthCons)
    if type == "UAutomizerC" and lengthCons:
        # length constraint (for UAutomizer)
        fp.write(f'  if ({lc}) {{ //length constraint: {lengthCons}\n')
        fp.write('    ERROR: __VERIFIER_error();\n')
        fp.write('  }\n')
        fp.write('  else {\n')
        fp.write('    return 0;\n')
        fp.write('  }\n')
    if type == "EldaricaC" and lengthCons:  # length constraint (for Eldarica)
        fp.write(f'  assert (!({lc})); //length constraint: {lengthCons}\n')
    fp.write(prog_end)

    fp.close()
