import sys

import parsing.basic


def main(argv):
    parsing.basic.parse_file(argv[1])


if __name__ == '__main__':
    main(sys.argv)
