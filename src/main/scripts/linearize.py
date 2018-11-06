import sys

import parsing.basic
import codegen.smtlib
import solver

sys.setrecursionlimit(4000)


def main(argv):
    file_name = argv[1].split('/')[-1]
    problem = parsing.basic.parse_file(argv[1])
    # problem.len_constraints = []
    problem.merge_all_word_equations()
    solver.turn_to_linear_we(problem)
    codegen.smtlib.to_file(problem, f'benchmarks/linear/{file_name}')


if __name__ == '__main__':
    main(sys.argv)
