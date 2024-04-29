lexer grammar CalcLexerRules;

PERCENT_NUMBER: NUMBER PERCENT;
NUMBER: DIGIT (POINT DIGIT)?;

DIGIT: [0-9]+;
BRACKET_L: '(';
BRACKET_R: ')';
ADD: '+';
SUB: '-';
MUL: '*';
DIV: '/';
PERCENT: '%';
POINT: '.';

WS: [ \t\r\n]+ -> skip;