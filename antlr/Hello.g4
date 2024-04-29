grammar Hello;              // 语法名称，必须要和文件名称一样

r  : 'hello' ID ;           // 表示匹配字符串hello和ID这个token，语法名称用小写字母定义
ID : [a-z]+ ;               // ID这个token的定义只允许小写字母，词法名称用大写字母定义
WS : [ \t\r\n]+ -> skip ;   // 忽略一些字符