import sys

import codegen.smtlib
import parsing.basic

sys.setrecursionlimit(4000)


def main(argv):
    file = argv[1].split('/')[-1]
    problem = parsing.basic.parse_file(argv[1])
    problem.len_constraints = []
    codegen.smtlib.to_file(problem, f'benchmarks/modified/{file}')


if __name__ == '__main__':
    main(sys.argv)
