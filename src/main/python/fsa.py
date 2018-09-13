import pywrapfst as fst

from copy import deepcopy
from typing import Iterator, Tuple, List

from bidict import bidict
from util import nostderr

StateId = int
Symbol = str
Outgoing = Tuple[Symbol, StateId]
Arc = Tuple[StateId, Symbol, StateId]
TRUE = fst.Weight.One('tropical')
FALSE = fst.Weight.Zero('tropical')
EPSILON = '(e)'


class Alphabet:
    def __init__(self):
        self._symbol_to_int = bidict()
        self._symbol_to_int[EPSILON] = 0

    def symbols(self):
        return self._symbol_to_int

    def take(self, symbol: Symbol):
        assert len(symbol) == 1 or symbol == EPSILON
        if symbol not in self._symbol_to_int:
            self._symbol_to_int[symbol] = len(self._symbol_to_int)
        return self._symbol_to_int[symbol]

    def symbol_index(self, symbol: Symbol):
        return self._symbol_to_int[symbol]

    def symbol(self, index: int):
        return self._symbol_to_int.inv[index]


def all_fsa(alphabet: Alphabet):
    result = FSA(alphabet)
    ss = result.start_state()
    result.set_final(ss)
    for symbol in alphabet.symbols():
        result.add_arc(ss, ss, symbol)
    return result


class FSA:
    def __init__(self, alphabet=Alphabet()):
        self._backend_obj = fst.Fst()
        self._biggest_state = -1
        self.set_start(self.add_state())
        self.alphabet = alphabet

    def __deepcopy__(self, memo):
        if self in memo:
            return memo.get(self)
        dup = FSA(self.alphabet)
        memo[self] = dup
        dup._backend_obj = self._backend_obj.copy()  # should be a deep copy
        dup._biggest_state = self._biggest_state
        return dup

    def __str__(self):
        states = 'states: ' + ' '.join(map(str, self.states()))
        start = f'start: {self.start_state()}'
        finals = 'finals: ' + ' '.join(map(str, self.final_states()))
        arcs = []
        for s, sym, ns in self.arcs():
            sym = sym if sym == EPSILON else f"'{sym}'"
            arcs.append(f'{s} -> {ns} [{sym}]')
        arcs = 'arcs:' + (('\n  ' + '\n  '.join(arcs)) if arcs else '')
        return '\n'.join([states, start, finals, arcs])

    def __repr__(self):
        start = f'S({self.start_state()})'
        finals = 'F(' + ' '.join(map(str, self.final_states())) + ')'
        arcs = []
        for s, sym, ns in self.arcs():
            sym = sym if sym == EPSILON else f"'{sym}'"
            arcs.append(f'({s}-{sym}-{ns})')
        arcs = 'ARCS(' + ''.join(arcs) + ')'
        return ''.join([start, finals, arcs])

    def __eq__(self, other: 'FSA'):
        return fst.equivalent(self._backend_obj, other._backend_obj)

    def __hash__(self):
        return hash(repr(self))

    def states(self) -> Iterator[StateId]:
        return self._backend_obj.states()

    def add_state(self) -> StateId:
        self._biggest_state += 1
        return self._backend_obj.add_state()

    def start_state(self) -> StateId:
        return self._backend_obj.start()

    def is_start(self, state: StateId):
        return state == self.start_state()

    def set_start(self, state: StateId) -> 'FSA':
        self._backend_obj.set_start(state)
        return self

    def final_states(self) -> Iterator[StateId]:
        return (s for s in self.states() if self.is_final(s))

    def is_final(self, state: StateId):
        with nostderr():
            return self._backend_obj.final(state) == TRUE

    def set_final(self, state: StateId) -> 'FSA':
        self._backend_obj.set_final(state)
        return self

    def unset_final(self, state: StateId) -> 'FSA':
        self._backend_obj.set_final(state, FALSE)
        return self

    def out_arcs(self, state: StateId) -> Iterator[Outgoing]:
        arcs = self._backend_obj.arcs(state)
        return ((self.alphabet.symbol(a.ilabel), a.nextstate) for a in arcs)

    def arcs(self) -> Iterator[Arc]:
        ss = self.states()
        return ((s, sym, ns) for s in ss for sym, ns in self.out_arcs(s))

    def add_arc(self, dept: StateId, dest: StateId, symbol: Symbol = EPSILON):
        assert dept <= self._biggest_state and dest <= self._biggest_state
        symbol_int = self.alphabet.take(symbol)
        arc = fst.Arc(symbol_int, symbol_int, TRUE, dest)
        self._backend_obj.add_arc(dept, arc)
        return self

    def determinize(self, destructive=True) -> 'FSA':
        tgt = self if destructive else deepcopy(self)
        tgt._backend_obj.rmepsilon()
        # result = FSA(tgt.alphabet)
        # result._backend_obj = fst.determinize(tgt._backend_obj)
        tgt._backend_obj = fst.determinize(tgt._backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return tgt

    def minimize(self, destructive=True) -> 'FSA':
        tgt = self if destructive else deepcopy(self)
        tgt._backend_obj.rmepsilon()
        tgt._backend_obj.minimize(allow_nondet=True)
        # TODO: ensure `tgt.biggest_state` is correct
        return tgt

    def complement(self, destructive=True):
        u = all_fsa(self.alphabet)._backend_obj
        tgt = self.determinize(destructive)._backend_obj
        result = self if destructive else FSA(self.alphabet)
        result._backend_obj = fst.difference(u, tgt)
        result._backend_obj.rmepsilon()
        # TODO: ensure `tgt.biggest_state` is correct
        return result

    def closure(self, destructive=True) -> 'FSA':
        tgt = self if destructive else deepcopy(self)
        tgt._backend_obj.closure()
        # TODO: ensure `tgt.biggest_state` is correct
        return tgt

    def concat(self, fsa: 'FSA') -> 'FSA':
        assert fsa.alphabet == self.alphabet
        result = FSA(self.alphabet)
        result._backend_obj = self._backend_obj.concat(fsa._backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return result

    def union(self, fsa: 'FSA') -> 'FSA':
        assert fsa.alphabet == self.alphabet
        result = FSA(self.alphabet)
        result._backend_obj = self._backend_obj.union(fsa._backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return result

    def intersect(self, fsa: 'FSA') -> 'FSA':
        assert fsa.alphabet == self.alphabet
        result = FSA(self.alphabet)
        result._backend_obj = fst.intersect(self._backend_obj, fsa._backend_obj)
        # TODO: ensure `tgt.biggest_state` is correct
        return result


def from_str(s: str, alphabet=Alphabet()) -> FSA:
    result = FSA(alphabet)
    curr_state = result.start_state()
    for ch in s:
        next_state = result.add_state()
        result.add_arc(curr_state, next_state, ch)
        curr_state = next_state
    result.set_final(curr_state)
    return result


def remove_first_char(fsa: FSA, ch: Symbol) -> [FSA]:
    new_init_states = {arc[1] for arc in fsa.out_arcs(fsa.start_state()) if arc[0] == ch}
    num = len(new_init_states)
    assert(num <= 1)
    if num == 0:  # no such char to remove, return an empty FSA
        return None
    ret_fsa = deepcopy(fsa)
    new_start = new_init_states.pop()
    ret_fsa.set_start(new_start)
    #if num == 1:  # just one arc found for the char to remove
    #    assert(len(new_init_states) == 0)
    #elif num > 1:  # add epsilon arc from the new start (PS: this case won't happen)
    #    for s in new_init_states:
    #        ret_fsa.add_arc(new_start, s)
    #    ret_fsa.determinize()  # determinize before return
    return ret_fsa


def split_by_states(fsa: FSA) -> List[Tuple[FSA, FSA]]:
    # assume fsa is deterministic
    ret = list()
    for s in fsa.states():
        if fsa.is_start(s):  # in this case fsa1 shall be an empty FSA
            fsa1, fsa2 = FSA(fsa.alphabet), deepcopy(fsa)
            fsa1.set_final(fsa1.start_state())
        else:
            fsa1, fsa2 = deepcopy(fsa), deepcopy(fsa)
            for f in fsa1.final_states():
                fsa1.unset_final(f)
            fsa1.set_final(s)
            fsa2.set_start(s)
        ret.append((fsa1, fsa2))
    return ret
