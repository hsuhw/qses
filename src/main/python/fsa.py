import pywrapfst as fst

from copy import deepcopy
from typing import Iterator, Tuple

from bidict import bidict
from util import nostderr

StateId = int
Symbol = str
Outgoing = Tuple[Symbol, StateId]
Arc = Tuple[StateId, Symbol, StateId]
TRUE = fst.Weight.One('tropical')
FALSE = fst.Weight.Zero('tropical')
EPSILON = '(e)'


def empty_alphabet():
    bd = bidict()
    bd[EPSILON] = 0
    return bd


class FSA:
    def __init__(self, alphabet=empty_alphabet()):
        assert alphabet[EPSILON] == 0
        self.backend_obj = fst.Fst()
        self.biggest_state = -1
        self.set_start(self.add_state())
        self.symbol_to_int = alphabet

    def __deepcopy__(self, memo):
        if self in memo:
            return memo.get(self)
        dup = FSA(self.symbol_to_int)
        memo[self] = dup
        dup.backend_obj = self.backend_obj.copy()  # supposed to be a deep copy
        dup.biggest_state = self.biggest_state
        return dup

    def __str__(self):
        states = 'states: ' + ' '.join(map(str, self.states()))
        start = f'start: {self.start()}'
        finals = 'finals: ' + ' '.join(map(str, self.finals()))
        arcs = []
        for s, sym, ns in self.arcs():
            sym = sym if sym == EPSILON else f"'{sym}'"
            arcs.append(f'{s} -> {ns} [{sym}]')
        arcs = 'arcs:' + (('\n  ' + '\n  '.join(arcs)) if arcs else '')
        return '\n'.join([states, start, finals, arcs])

    def states(self) -> Iterator[StateId]:
        return self.backend_obj.states()

    def add_state(self) -> StateId:
        self.biggest_state += 1
        return self.backend_obj.add_state()

    def start_state(self) -> StateId:
        return self.backend_obj.start()

    def is_start(self, state: StateId):
        return state == self.start_state()

    def set_start(self, state: StateId) -> 'FSA':
        self.backend_obj.set_start(state)
        return self

    def final_states(self) -> Iterator[StateId]:
        return (s for s in self.states() if self.is_final(s))

    def is_final(self, state: StateId):
        with nostderr():
            return self.backend_obj.final(state) == TRUE

    def set_final(self, state: StateId) -> 'FSA':
        self.backend_obj.set_final(state)
        return self

    def take_symbol(self, symbol: Symbol):
        if symbol not in self.symbol_to_int:
            self.symbol_to_int[symbol] = len(self.symbol_to_int)
        return self.symbol_to_int[symbol]

    def out_arcs(self, state: StateId) -> Iterator[Outgoing]:
        arcs = self.backend_obj.arcs(state)
        return ((self.symbol_to_int.inv[a.ilabel], a.nextstate) for a in arcs)

    def arcs(self) -> Iterator[Arc]:
        ss = self.states()
        return ((s, sym, ns) for s in ss for sym, ns in self.out_arcs(s))

    def add_arc(self, dept: StateId, dest: StateId, symbol: Symbol = EPSILON):
        assert len(symbol) == 1 or symbol == EPSILON
        assert dept <= self.biggest_state and dest <= self.biggest_state
        symbol_int = self.take_symbol(symbol)
        arc = fst.Arc(symbol_int, symbol_int, TRUE, dest)
        self.backend_obj.add_arc(dept, arc)
        return self

    def minimize(self, in_place=True) -> 'FSA':
        tgt = self if in_place else deepcopy(self)
        tgt.backend_obj.rmepsilon()
        tgt.backend_obj.minimize(allow_nondet=True)
        # TODO: ensure `tgt.biggest_state` is correct
        return tgt

    def closure(self, in_place=True) -> 'FSA':
        tgt = self if in_place else deepcopy(self)
        tgt.backend_obj.closure()
        # TODO: ensure `tgt.biggest_state` is correct
        return tgt

    def concat(self, fsa: 'FSA') -> 'FSA':
        assert fsa.symbol_to_int == self.symbol_to_int
        result = FSA(self.symbol_to_int)
        result.backend_obj = self.backend_obj.concat(fsa.backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return result

    def union(self, fsa: 'FSA') -> 'FSA':
        assert fsa.symbol_to_int == self.symbol_to_int
        result = FSA(self.symbol_to_int)
        result.backend_obj = self.backend_obj.union(fsa.backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return result

    def intersect(self, fsa: 'FSA') -> 'FSA':
        assert fsa.symbol_to_int == self.symbol_to_int
        result = FSA(self.symbol_to_int)
        result.backend_obj = fst.intersect(self.backend_obj, fsa.backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return result


def from_str(s: str, alphabet=empty_alphabet()) -> FSA:
    result = FSA(alphabet)
    curr_state = result.start_state()
    for ch in s:
        next_state = result.add_state()
        result.add_arc(curr_state, next_state, ch)
        curr_state = next_state
    result.set_final(curr_state)
    return result
