```text
<Statement>    ::= ( <SelectStmt> | <InsertStmt> | <DeleteStmt> | <DropTableStmt> ) SEMICOL
<SelectStmt>   ::= SELECT <SelectList> FROM ID [ <WhereClause> ] [ <GroupOrOrder> ]
<SelectList>   ::= MUL | ID { COMA ID }
<WhereClause>  ::= WHERE <Condition>
<Condition>    ::= <QualifiedName> <Operator> <Value>
<Operator>     ::= EQ | LE | LEQ | GT | GEQ
<Value>        ::= NUM | STRING | <QualifiedName>
<QualifiedName> ::= ID [ DOT ID ]
<GroupOrOrder> ::= { GROUP_BY <QualifiedName> { COMA <QualifiedName> } }
				   { ORDER_BY <QualifiedName> { COMA <QualifiedName> } }
<InsertStmt>   ::= INSERT_INTO ID VALUES <ValueRow> { COMA <ValueRow> }
<ValueRow>     ::= LPAREN <ValueList> RPAREN
<ValueList>    ::= <Value> { COMA <Value> }
<DeleteStmt>   ::= DELETE FROM ID [ <WhereClause> ]
<DropTableStmt> ::= DROP_TABLE ID
<CreateTableStmt> ::= CREATE_TABLE ID LPAREN <ColumnDefList> RPAREN [ USING <StorageType> ]
<StorageType>     ::= HEAP | SEQUENTIAL
```
