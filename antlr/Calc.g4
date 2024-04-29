grammar Calc;
import CalcLexerRules;  // 引入CalcLexerRules的词法规则

calc: (expr)* EOF;

expr:
    BRACKET_L expr BRACKET_R
    | (ADD | SUB)? (NUMBER | PERCENT_NUMBER)
    | expr (MUL | DIV) expr
    | expr (ADD | SUB) expr;