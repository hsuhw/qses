import sys

import parsing.basic
import solver

from codegen.smtlib import SMTLIBLayout, Z3STR3_SYNTAX

sys.setrecursionlimit(4000)


def main(argv):
    problem = parsing.basic.parse_file(argv[1])
    # problem.len_constraints = []
    problem.merge_all_word_equations()
    solver.turn_to_quadratic_wes(problem)
    SMTLIBLayout(problem, Z3STR3_SYNTAX).print(sys.stdout)


if __name__ == '__main__':
    main(sys.argv)
