import re

STR_THEORY_AND = 'and'
STR_THEORY_OR = 'or'
STR_THEORY_NOT = 'not'
STR_THEORY_EQ = '='
STR_THEORY_GT = '>'
STR_THEORY_LT = '<'
STR_THEORY_GEQ = '>='
STR_THEORY_LEQ = '<='
STR_THEORY_PLUS = '+'
STR_THEORY_MINUS = '-'
STR_THEORY_TIMES = '*'
STR_THEORY_STR_CONCAT_V1 = 'Concat'
STR_THEORY_STR_CONCAT_V2 = 'str.++'
STR_THEORY_STR_LENGTH_V1 = 'Length'
STR_THEORY_STR_LENGTH_V2 = 'str.len'
STR_THEORY_STR_CONTAINS_V1 = 'Contains'
STR_THEORY_STR_CONTAINS_V2 = 'str.contains'
STR_THEORY_STR_IN_RE_V1 = 'RegexIn'
STR_THEORY_STR_IN_RE_V2 = 'str.in.re'
STR_THEORY_RE_FROM_STR_V1 = 'RegexIn'
STR_THEORY_RE_FROM_STR_V2 = 'str.to.re'
STR_THEORY_RE_CONCAT_V1 = 'RegexConcat'
STR_THEORY_RE_CONCAT_V2 = 're.++'
STR_THEORY_RE_UNION_V1 = 'RegexUnion'
STR_THEORY_RE_UNION_V2 = 're.union'
STR_THEORY_RE_CLOSURE_V1 = 'RegexStar'
STR_THEORY_RE_CLOSURE_V2 = 're.*'

INTERNAL_VAR_PREFIX = 'xx_'
INTERNAL_LEN_VAR_POSTFIX = '_len'

internal_var_name = re.compile(f'{INTERNAL_VAR_PREFIX}.*')
