import sys

import parsing.basic

from codegen.smtlib import SMTLIBLayout, Z3STR3_SYNTAX

sys.setrecursionlimit(4000)


def main(argv):
    problem = parsing.basic.parse_file(argv[1])
    problem.len_constraints = []
    SMTLIBLayout(problem, Z3STR3_SYNTAX).print(sys.stdout)


if __name__ == '__main__':
    main(sys.argv)
