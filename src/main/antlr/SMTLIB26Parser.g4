parser grammar SMTLIB26Parser;

options {
  tokenVocab = SMTLIB26Lexer;
}

// S-expressions
spec_constant
  : NUMERAL | DECIMAL | HEXADECIMAL | BINARY | STRING;
s_expr
  : spec_constant | SYMBOL | KEYWORD | OPEN_PAR s_expr* CLOSE_PAR;

// Identifiers
index
  : NUMERAL | SYMBOL;
identifier
  : SYMBOL | OPEN_PAR TOKEN_UNDERSCORE SYMBOL index+ CLOSE_PAR;

// Sorts
sort
  : identifier | OPEN_PAR identifier sort+ CLOSE_PAR;

// Attributes
attribute_value
  : spec_constant | SYMBOL | OPEN_PAR s_expr* CLOSE_PAR;
attribute
  : KEYWORD | KEYWORD attribute_value;

// Terms
qual_identifier
  : identifier | OPEN_PAR TOKEN_AS identifier sort CLOSE_PAR;
var_binding
  : OPEN_PAR SYMBOL term CLOSE_PAR;
sorted_var
  : OPEN_PAR SYMBOL sort CLOSE_PAR;
pattern
  : SYMBOL | OPEN_PAR SYMBOL SYMBOL+ CLOSE_PAR;
match_case
  : OPEN_PAR pattern term CLOSE_PAR;
term
  : spec_constant
  | qual_identifier
  | OPEN_PAR qual_identifier term+ CLOSE_PAR
  | OPEN_PAR TOKEN_LET OPEN_PAR var_binding+ CLOSE_PAR term CLOSE_PAR
  | OPEN_PAR TOKEN_FORALL OPEN_PAR sorted_var+ CLOSE_PAR term CLOSE_PAR
  | OPEN_PAR TOKEN_EXISTS OPEN_PAR sorted_var+ CLOSE_PAR term CLOSE_PAR
  | OPEN_PAR TOKEN_MATCH term OPEN_PAR match_case+ CLOSE_PAR CLOSE_PAR
  | OPEN_PAR TOKEN_BANG term attribute+ CLOSE_PAR
  ;

// Theories
sort_symbol_decl
  : OPEN_PAR identifier NUMERAL attribute* CLOSE_PAR;
meta_spec_constant
  : TOKEN_NUMERAL | TOKEN_DECIMAL | TOKEN_STRING;
fun_symbol_decl
  : OPEN_PAR spec_constant sort attribute* CLOSE_PAR
  | OPEN_PAR meta_spec_constant sort attribute* CLOSE_PAR
  | OPEN_PAR identifier sort+ attribute* CLOSE_PAR
  ;
par_fun_symbol_decl
  : fun_symbol_decl
  | OPEN_PAR TOKEN_PAR OPEN_PAR SYMBOL+ CLOSE_PAR OPEN_PAR identifier sort+ attribute* CLOSE_PAR CLOSE_PAR
  ;
theory_attribute
  : KEYWORD_SORTS OPEN_PAR sort_symbol_decl+ CLOSE_PAR
  | KEYWORD_FUNS OPEN_PAR par_fun_symbol_decl+ CLOSE_PAR
  | KEYWORD_SORTS_DESCRIPTION STRING
  | KEYWORD_FUNS_DESCRIPTION STRING
  | KEYWORD_DEFINITION STRING
  | KEYWORD_VALUES STRING
  | KEYWORD_NOTES STRING
  | attribute
  ;
theory_decl
  : OPEN_PAR SYMBOL_THEORY SYMBOL? theory_attribute+ CLOSE_PAR;

// Logics
logic_attribute
  : KEYWORD_THEORIES OPEN_PAR SYMBOL+ CLOSE_PAR
  | KEYWORD_LANGUAGE STRING
  | KEYWORD_EXTENSIONS STRING
  | KEYWORD_VALUES STRING
  | KEYWORD_NOTES STRING
  | attribute
  ;
logic
  : OPEN_PAR SYMBOL_LOGIC SYMBOL logic_attribute+ CLOSE_PAR;


// Info flags
info_flag
  : KEYWORD_ALL_STATISTICS
  | KEYWORD_ASSERTION_STACK_LEVELS
  | KEYWORD_AUTHORS
  | KEYWORD_ERROR_BEHAVIOR
  | KEYWORD_NAME
  | KEYWORD_REASON_UNKNOWN
  | KEYWORD_VERSION
  | KEYWORD
  ;

// Command options
b_value
  : SYMBOL_TRUE | SYMBOL_FALSE;
option
  : KEYWORD_DIAGNOSTIC_OUTPUT_CHANNEL STRING
  | KEYWORD_GLOBAL_DECLARATIONS b_value
  | KEYWORD_INTERACTIVE_MODE b_value
  | KEYWORD_PRINT_SUCCESS b_value
  | KEYWORD_PRODUCE_ASSERTIONS b_value
  | KEYWORD_PRODUCE_ASSIGNMENTS b_value
  | KEYWORD_PRODUCE_MODELS b_value
  | KEYWORD_PRODUCE_PROOFS b_value
  | KEYWORD_PRODUCE_UNSAT_ASSUMPTIONS b_value
  | KEYWORD_PRODUCE_UNSAT_CORES b_value
  | KEYWORD_RANDOM_SEED NUMERAL
  | KEYWORD_REGULAR_OUTPUT_CHANNEL STRING
  | KEYWORD_REPRODUCIBLE_RESOURCE_LIMIT NUMERAL
  | KEYWORD_VERBOSITY NUMERAL
  | attribute
  ;

// Commands
sort_dec
  : OPEN_PAR SYMBOL NUMERAL CLOSE_PAR;
selector_dec
  : OPEN_PAR SYMBOL sort CLOSE_PAR;
constructor_dec
  : OPEN_PAR SYMBOL selector_dec* CLOSE_PAR;
datatype_dec
  : OPEN_PAR constructor_dec+ CLOSE_PAR
  | OPEN_PAR TOKEN_PAR OPEN_PAR SYMBOL+ CLOSE_PAR OPEN_PAR constructor_dec+ CLOSE_PAR CLOSE_PAR
  ;
function_dec
  : OPEN_PAR SYMBOL OPEN_PAR sorted_var* CLOSE_PAR sort CLOSE_PAR;
function_def
  : SYMBOL OPEN_PAR sorted_var* CLOSE_PAR sort term;
prop_literal
  : SYMBOL | OPEN_PAR SYMBOL_NOT SYMBOL CLOSE_PAR;
command
  : OPEN_PAR TOKEN_CMD_ASSERT term CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_CHECK_SAT CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_CHECK_SAT_ASSUMING OPEN_PAR prop_literal* CLOSE_PAR CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DECLARE_CONST SYMBOL sort CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DECLARE_DATATYPE SYMBOL datatype_dec CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DECLARE_DATATYPES OPEN_PAR sort_dec+ CLOSE_PAR OPEN_PAR datatype_dec+ CLOSE_PAR CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DECLARE_FUN SYMBOL OPEN_PAR sort* CLOSE_PAR sort CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DECLARE_SORT SYMBOL NUMERAL CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DEFINE_FUN function_def CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DEFINE_FUN_REC function_def CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DEFINE_FUNS_REC OPEN_PAR function_dec+ CLOSE_PAR OPEN_PAR term+ CLOSE_PAR CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DEFINE_SORT SYMBOL OPEN_PAR SYMBOL* CLOSE_PAR sort CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_ECHO STRING CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_EXIT CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_ASSERTIONS CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_ASSIGNMENT CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_INFO info_flag CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_MODEL CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_OPTION KEYWORD CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_PROOF CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_UNSET_ASSUMPTIONS CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_UNSAT_CORE CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_GET_VALUE OPEN_PAR term+ CLOSE_PAR CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_POP NUMERAL CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_PUSH NUMERAL CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_RESET CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_RESET_ASSERTIONS CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_SET_INFO attribute CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_SET_LOGIC SYMBOL CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_SET_OPTION option CLOSE_PAR
  ;
script : command+;

// Command responses
error_behavior
  : SYMBOL_IMMEDIATE_EXIT | SYMBOL_CONTINUED_EXECUTION;
reason_unknown
  : SYMBOL_MEMOUT | SYMBOL_INCOMPLETE | s_expr;
model_response
  : OPEN_PAR TOKEN_CMD_DEFINE_FUN function_def CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DEFINE_FUN_REC function_def CLOSE_PAR
  | OPEN_PAR TOKEN_CMD_DEFINE_FUNS_REC OPEN_PAR function_dec+ CLOSE_PAR OPEN_PAR term+ CLOSE_PAR CLOSE_PAR
  ;
info_response
  : KEYWORD_ASSERTION_STACK_LEVELS NUMERAL
  | KEYWORD_AUTHORS STRING
  | KEYWORD_ERROR_BEHAVIOR error_behavior
  | KEYWORD_NAME STRING
  | KEYWORD_REASON_UNKNOWN reason_unknown
  | KEYWORD_VERSION STRING
  | attribute
  ;
valuation_pair
  : OPEN_PAR term term CLOSE_PAR;
t_valuation_pair
  : OPEN_PAR SYMBOL b_value CLOSE_PAR;
check_sat_response
  : SYMBOL_SAT | SYMBOL_UNSAT | SYMBOL_UNKNOWN;
echo_response
  : STRING;
get_assertions_response
  : OPEN_PAR term* CLOSE_PAR;
get_assignment_response
  : OPEN_PAR t_valuation_pair* CLOSE_PAR;
get_info_response
  : OPEN_PAR info_response+ CLOSE_PAR;
get_model_response
  : OPEN_PAR model_response* CLOSE_PAR;
get_option_response
  : attribute_value;
get_proof_response
  : s_expr;
get_unsat_assump_response
  : OPEN_PAR SYMBOL* CLOSE_PAR;
get_unsat_core_response
  : OPEN_PAR SYMBOL* CLOSE_PAR;
get_value_response
  : OPEN_PAR valuation_pair+ CLOSE_PAR;
specific_success_response
  : check_sat_response
  | echo_response
  | get_assertions_response
  | get_assignment_response
  | get_info_response
  | get_model_response
  | get_option_response
  | get_proof_response
  | get_unsat_assump_response
  | get_unsat_core_response
  | get_value_response
  ;
general_response
  : SYMBOL_SUCCESS
  | specific_success_response
  | SYMBOL_UNSUPPORTED
  | OPEN_PAR SYMBOL_ERROR STRING CLOSE_PAR;
