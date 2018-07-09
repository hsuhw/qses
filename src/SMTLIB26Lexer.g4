lexer grammar SMTLIB26Lexer;

// Predefined Symbols
SYMBOL_BOOL                : 'Bool';
SYMBOL_CONTINUED_EXECUTION : 'continued-execution';
SYMBOL_ERROR               : 'error';
SYMBOL_FALSE               : 'false';
SYMBOL_IMMEDIATE_EXIT      : 'immediate-exit';
SYMBOL_INCOMPLETE          : 'incomplete';
SYMBOL_LOGIC               : 'logic';
SYMBOL_MEMOUT              : 'memout';
SYMBOL_NOT                 : 'not';
SYMBOL_SAT                 : 'sat';
SYMBOL_SUCCESS             : 'success';
SYMBOL_THEORY              : 'theory';
SYMBOL_TRUE                : 'true';
SYMBOL_UNKNOWN             : 'unknown';
SYMBOL_UNSAT               : 'unsat';
SYMBOL_UNSUPPORTED         : 'unsupported';

// Predefined Keywords
KEYWORD_ALL_STATISTICS              : ':all-statistics';
KEYWORD_ASSERTION_STACK_LEVELS      : ':assertion-stack-levels';
KEYWORD_AUTHORS                     : ':authors';
KEYWORD_CATEGORY                    : ':category';
KEYWORD_CHAINABLE                   : ':chainable';
KEYWORD_DEFINITION                  : ':definition';
KEYWORD_DIAGNOSTIC_OUTPUT_CHANNEL   : ':diagnostic-output-channel';
KEYWORD_ERROR_BEHAVIOR              : ':error-behavior';
KEYWORD_EXTENSIONS                  : ':extensions';
KEYWORD_FUNS                        : ':funs';
KEYWORD_FUNS_DESCRIPTION            : ':funs-description';
KEYWORD_GLOBAL_DECLARATIONS         : ':global-declarations';
KEYWORD_INTERACTIVE_MODE            : ':interactive-mode';
KEYWORD_LANGUAGE                    : ':language';
KEYWORD_LEFT_ASSOC                  : ':left-assoc';
KEYWORD_LICENSE                     : ':license';
KEYWORD_NAME                        : ':name';
KEYWORD_NAMED                       : ':named';
KEYWORD_NOTES                       : ':notes';
KEYWORD_PATTERN                     : ':pattern';
KEYWORD_PRINT_SUCCESS               : ':print-success';
KEYWORD_PRODUCE_ASSIGNMENTS         : ':produce-assignments';
KEYWORD_PRODUCE_ASSERTIONS          : ':produce-assertions';
KEYWORD_PRODUCE_MODELS              : ':produce-models';
KEYWORD_PRODUCE_PROOFS              : ':produce-proofs';
KEYWORD_PRODUCE_UNSAT_ASSUMPTIONS   : ':produce-unsat-assumptions';
KEYWORD_PRODUCE_UNSAT_CORES         : ':produce-unsat-cores';
KEYWORD_RANDOM_SEED                 : ':random-seed';
KEYWORD_REASON_UNKNOWN              : ':reason-unknown';
KEYWORD_REGULAR_OUTPUT_CHANNEL      : ':regular-output-channel';
KEYWORD_REPRODUCIBLE_RESOURCE_LIMIT : ':reproducible-resource-limit';
KEYWORD_RIGHT_ASSOC                 : ':right-assoc';
KEYWORD_SMT_LIB_VERSION             : ':smt-lib-version';
KEYWORD_SORTS                       : ':sorts';
KEYWORD_SORTS_DESCRIPTION           : ':sorts-description';
KEYWORD_SOURCE                      : ':source';
KEYWORD_STATUS                      : ':status';
KEYWORD_THEORIES                    : ':theories';
KEYWORD_VALUES                      : ':values';
KEYWORD_VERBOSITY                   : ':verbosity';
KEYWORD_VERSION                     : ':version';

// Auxiliary Parts
WS                 : [\t\r\n\f ]+ { skip(); };
fragment DIGIT     : [0-9];
fragment HEXDIGIT  : DIGIT | [a-fA-F];
fragment LETTER    : [a-zA-Z];
fragment NONLETTER : [-+~!@$%^&*_=<>.?\/];
fragment ESCAPE    : '""';

// General Tokens
TOKEN_BINARY      : 'BINARY';
TOKEN_DECIMAL     : 'DECIMAL';
TOKEN_HEXADECIMAL : 'HEXADECIMAL';
TOKEN_NUMERAL     : 'NUMERAL';
TOKEN_STRING      : 'STRING';
TOKEN_BANG        : '!';
TOKEN_UNDERSCORE  : '_';
TOKEN_AS          : 'as';
TOKEN_LET         : 'let';
TOKEN_EXISTS      : 'exists';
TOKEN_FORALL      : 'forall';
TOKEN_MATCH       : 'match';
TOKEN_PAR         : 'par';

// Command Name Tokens
TOKEN_CMD_ASSERT                : 'assert';
TOKEN_CMD_CHECK_SAT             : 'check-sat';
TOKEN_CMD_CHECK_SAT_ASSUMING    : 'check-sat-assuming';
TOKEN_CMD_DECLARE_CONST         : 'declare-const';
TOKEN_CMD_DECLARE_DATATYPE      : 'declare-datatype';
TOKEN_CMD_DECLARE_DATATYPES     : 'declare-datatypes';
TOKEN_CMD_DECLARE_FUN           : 'declare-fun';
TOKEN_CMD_DECLARE_SORT          : 'declare-sort';
TOKEN_CMD_DEFINE_FUN            : 'define-fun';
TOKEN_CMD_DEFINE_FUN_REC        : 'define-fun';
TOKEN_CMD_DEFINE_FUNS_REC       : 'define-funs-rec';
TOKEN_CMD_DEFINE_SORT           : 'define-sort';
TOKEN_CMD_ECHO                  : 'echo';
TOKEN_CMD_EXIT                  : 'exit';
TOKEN_CMD_GET_ASSERTIONS        : 'get-assertions';
TOKEN_CMD_GET_ASSIGNMENT        : 'get-assignment';
TOKEN_CMD_GET_INFO              : 'get-info';
TOKEN_CMD_GET_MODEL             : 'get-model';
TOKEN_CMD_GET_OPTION            : 'get-option';
TOKEN_CMD_GET_PROOF             : 'get-proof';
TOKEN_CMD_GET_UNSET_ASSUMPTIONS : 'get-unsat-assumptions';
TOKEN_CMD_GET_UNSAT_CORE        : 'get-unsat-core';
TOKEN_CMD_GET_VALUE             : 'get-value';
TOKEN_CMD_POP                   : 'pop';
TOKEN_CMD_PUSH                  : 'push';
TOKEN_CMD_RESET                 : 'reset';
TOKEN_CMD_RESET_ASSERTIONS      : 'reset-assertions';
TOKEN_CMD_SET_INFO              : 'set-info';
TOKEN_CMD_SET_LOGIC             : 'set-logic';
TOKEN_CMD_SET_OPTION            : 'set-option';

// Other Tokens
OPEN_PAR      : '(';
CLOSE_PAR     : ')';
NUMERAL       : '0' | [1-9] DIGIT*;
DECIMAL       : NUMERAL '.' [0]* NUMERAL;
HEXADECIMAL   : '#x' HEXDIGIT+;
BINARY        : '#b' [01]+;
STRING        : '"' ((ESCAPE | ~('"'))*) '"';
SIMPLE_SYMBOL : (LETTER | NONLETTER) (DIGIT | LETTER | NONLETTER)*;
QUOTED_SYMBOL : '|' ~('|' | '\\')* '|';
COMMENT       : ';' ~('\n' | '\r')* { skip(); };
KEYWORD_TOKEN : ':' SIMPLE_SYMBOL;

SYMBOL
  : SIMPLE_SYMBOL
  | QUOTED_SYMBOL
  | SYMBOL_BOOL
  | SYMBOL_CONTINUED_EXECUTION
  | SYMBOL_ERROR
  | SYMBOL_FALSE
  | SYMBOL_IMMEDIATE_EXIT
  | SYMBOL_INCOMPLETE
  | SYMBOL_LOGIC
  | SYMBOL_MEMOUT
  | SYMBOL_NOT
  | SYMBOL_SAT
  | SYMBOL_SUCCESS
  | SYMBOL_THEORY
  | SYMBOL_TRUE
  | SYMBOL_UNKNOWN
  | SYMBOL_UNSAT
  | SYMBOL_UNSUPPORTED
  ;

KEYWORD
  : KEYWORD_TOKEN
  | KEYWORD_ALL_STATISTICS
  | KEYWORD_ASSERTION_STACK_LEVELS
  | KEYWORD_AUTHORS
  | KEYWORD_CATEGORY
  | KEYWORD_CHAINABLE
  | KEYWORD_DEFINITION
  | KEYWORD_DIAGNOSTIC_OUTPUT_CHANNEL
  | KEYWORD_ERROR_BEHAVIOR
  | KEYWORD_EXTENSIONS
  | KEYWORD_FUNS
  | KEYWORD_FUNS_DESCRIPTION
  | KEYWORD_GLOBAL_DECLARATIONS
  | KEYWORD_INTERACTIVE_MODE
  | KEYWORD_LANGUAGE
  | KEYWORD_LEFT_ASSOC
  | KEYWORD_LICENSE
  | KEYWORD_NAME
  | KEYWORD_NAMED
  | KEYWORD_NOTES
  | KEYWORD_PATTERN
  | KEYWORD_PRINT_SUCCESS
  | KEYWORD_PRODUCE_ASSIGNMENTS
  | KEYWORD_PRODUCE_ASSERTIONS
  | KEYWORD_PRODUCE_MODELS
  | KEYWORD_PRODUCE_PROOFS
  | KEYWORD_PRODUCE_UNSAT_ASSUMPTIONS
  | KEYWORD_PRODUCE_UNSAT_CORES
  | KEYWORD_RANDOM_SEED
  | KEYWORD_REASON_UNKNOWN
  | KEYWORD_REGULAR_OUTPUT_CHANNEL
  | KEYWORD_REPRODUCIBLE_RESOURCE_LIMIT
  | KEYWORD_RIGHT_ASSOC
  | KEYWORD_SMT_LIB_VERSION
  | KEYWORD_SORTS
  | KEYWORD_SORTS_DESCRIPTION
  | KEYWORD_SOURCE
  | KEYWORD_STATUS
  | KEYWORD_THEORIES
  | KEYWORD_VALUES
  | KEYWORD_VERBOSITY
  | KEYWORD_VERSION
  ;
