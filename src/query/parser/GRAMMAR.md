```text
<Statement>          ::= <SelectStmt> SEMICOL
					   | <InsertStmt> SEMICOL
					   | <DeleteStmt> SEMICOL
					   | <UpdateStmt> SEMICOL
					   | <CreateTableStmt> SEMICOL
					   | <CreateIndexStmt> SEMICOL
					   | BEGIN_TRANSACTION SEMICOL
					   | END_TRANSACTION SEMICOL
					   | EXPLAIN [ ANALYZE ] <SelectStmt> SEMICOL

<SelectStmt>         ::= SELECT <SelectList> FROM ID [ <JoinClause> ]
						 [ <WhereClause> ] [ <GroupOrOrder> ] [ <LimitClause> ]
<SelectList>         ::= MUL | <SelectItem> { COMA <SelectItem> }
<SelectItem>         ::= <QualifiedName> | <Aggregate>
<Aggregate>          ::= COUNT ( MUL | <QualifiedName> )
					   | SUM ( <QualifiedName> )
					   | AVG ( <QualifiedName> )
					   | MIN ( <QualifiedName> )
					   | MAX ( <QualifiedName> )
<JoinClause>         ::= JOIN ID ON <QualifiedName> EQ <QualifiedName>
<WhereClause>        ::= WHERE <Condition>
<Condition>          ::= <QualifiedName> <Operator> <Value>
					   | <DistanceExp> <Operator> <Value>
					   | <WithinExp>
<Operator>           ::= EQ | NEQ | LE | LEQ | GT | GEQ
<Value>              ::= NUM | STRING | <QualifiedName> | NULL | <Point> | <Polygon>
<QualifiedName>      ::= ID [ DOT ID ]
<GroupOrOrder>       ::= { <GroupByClause> | <OrderByClause> }
<GroupByClause>      ::= GROUP_BY <QualifiedName> { COMA <QualifiedName> }
<OrderByClause>      ::= ORDER_BY <OrderByItem> { COMA <OrderByItem> }
<OrderByItem>        ::= <QualifiedName> | <DistanceExp>
<LimitClause>        ::= LIMIT NUM

<InsertStmt>         ::= INSERT_INTO ID [ LPAREN <ColumnNameList> RPAREN ]
						 VALUES <ValueRow> { COMA <ValueRow> }
<ColumnNameList>     ::= ID { COMA ID }
<ValueRow>           ::= LPAREN <ValueList> RPAREN
<ValueList>          ::= <Value> { COMA <Value> }
<DeleteStmt>         ::= DELETE FROM ID [ <WhereClause> ]
<UpdateStmt>         ::= UPDATE ID SET ID EQ <Value> WHERE <Condition>

<CreateTableStmt>    ::= CREATE_TABLE ID LPAREN <ColumnDefList> RPAREN
						 [ USING <StorageType> ]
<ColumnDefList>      ::= <ColumnDef> { COMA <ColumnDef> }
<ColumnDef>          ::= ID <TypeName> [ LPAREN NUM RPAREN ] { <ColumnConstraint> }
<ColumnConstraint>   ::= PRIMARY_KEY | NOT_NULL | UNIQUE
<TypeName>           ::= SMALLINT | INTEGER | BIGINT | NUMERIC | REAL
					   | DOUBLE_PRECISION | CHAR | VARCHAR | TEXT | BOOLEAN
					   | DATE | TIME | TIMESTAMP | BYTEA | POINT
<StorageType>        ::= HEAP | SEQUENTIAL
<CreateIndexStmt>    ::= CREATE_INDEX ID ON ID LPAREN ID RPAREN USING <IndexType>
<IndexType>          ::= BTREE | HASH | RTREE

<Point>              ::= POINT LPAREN NUM COMA NUM RPAREN
<PointList>          ::= <Point> { COMA <Point> }
<Polygon>            ::= POLYGON LPAREN <PointList> RPAREN
<DistanceExp>        ::= DISTANCIA LPAREN <QualifiedName> COMA <SpatialPointArg>
						 [ COMA <Metric> ] RPAREN
<SpatialPointArg>    ::= <Point> | <QualifiedName>
<WithinExp>          ::= DENTRO_DE LPAREN <QualifiedName> COMA <Polygon> RPAREN
<Metric>             ::= EUCLIDEAN | HAVERSINE
```
