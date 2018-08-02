from typing import Tuple, Dict, Set, Optional
from enum import Enum, unique
from we import WordEquation, Element


@unique
class RewriteType(Enum):
    lvar_be_empty = 1
    rvar_be_empty = 2
    lvar_be_char = 3
    rvar_be_char = 4
    lvar_be_rvar = 5
    lvar_longer_char = 6
    rvar_longer_char = 7
    lvar_longer_var = 8
    rvar_longer_var = 9


class Transform:
    def __init__(self, source: WordEquation, rewrite: RewriteType,
                 head_pair: Tuple[Optional[Element], Optional[Element]]):
        self.source: WordEquation = source
        self.rewrite: RewriteType = rewrite
        self.head_pair = head_pair


class SolvingTree:
    success_end: WordEquation = WordEquation([], [])

    def __init__(self, root: WordEquation):
        self.root: WordEquation = root
        self.node_relations: Dict[WordEquation, Set[Transform]] = {}

    def has_solution(self):
        return SolvingTree.success_end in self.node_relations

    def add_node(self, node: WordEquation, trans_from: Transform):
        # TODO: not correct yet
        if node in self.node_relations:
            self.node_relations[node].add(trans_from)
        else:
            self.node_relations[node] = {trans_from}
