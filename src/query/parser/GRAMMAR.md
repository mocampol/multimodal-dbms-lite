```text
<Statement>    ::= ( <SelectStmt> | <InsertStmt> | <DeleteStmt> ) SEMICOL
<SelectStmt>   ::= SELECT <SelectList> FROM ID [ <WhereClause> ] [ <GroupOrOrder> ]
<SelectList>   ::= MUL | ID { COMA ID }
<WhereClause>  ::= WHERE <Condition>
<Condition>    ::= ID <Operator> <Value>
<Operator>     ::= EQ | LE | LEQ | GT | GEQ
<Value>        ::= NUM | STRING | ID
<GroupOrOrder> ::= ORDER_BY ID { COMA ID } | GROUP_BY ID { COMA ID }
<InsertStmt>   ::= INSERT_INTO ID VALUES LPAREN <ValueList> RPAREN
<ValueList>    ::= <Value> { COMA <Value> }
<DeleteStmt>   ::= DELETE FROM ID [ <WhereClause> ]
```