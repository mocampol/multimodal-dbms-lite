````text
<Statement>    ::= [ EXPLAIN [ ANALYZE ] ] ( <SelectStmt> | <InsertStmt> | <DeleteStmt> ) SEMICOL
<SelectStmt>   ::= SELECT <SelectList> FROM ID [ <WhereClause> ] [ <GroupOrOrder> ]
<SelectList>   ::= MUL | ID { COMA ID }
<WhereClause>  ::= WHERE <Condition>
<Condition>    ::= <QualifiedName> <Operator> <Value>
<Operator>     ::= EQ | LE | LEQ | GT | GEQ
<Value>        ::= NUM | STRING | <QualifiedName> | NULL | <Point>
<QualifiedName> ::= ID [ DOT ID ]
<GroupOrOrder> ::= { GROUP_BY <QualifiedName> { COMA <QualifiedName> } }
				   { ORDER_BY <QualifiedName> { COMA <QualifiedName> } }
<InsertStmt>      ::= INSERT_INTO ID [ LPAREN <ColumnNameList> RPAREN ] VALUES <ValueRow> { COMA <ValueRow> }
<ColumnNameList>  ::= ID { COMA ID }
<ValueRow>     ::= LPAREN <ValueList> RPAREN
<ValueList>    ::= <Value> { COMA <Value> }
<DeleteStmt>   ::= DELETE FROM ID [ <WhereClause> ]
<CreateTableStmt> ::= CREATE_TABLE ID LPAREN <ColumnDefList> RPAREN [ USING <StorageType> ]
<StorageType>     ::= HEAP | SEQUENTIAL

## Valores espaciales

El parser acepta `POINT` como valor literal en `INSERT` y `UPDATE`. Las demás
producciones espaciales de esta sección describen extensiones futuras.

```text
<SpatialType> ::= POINT | RECTANGLE | POLYGON
<Point>       ::= POINT LPAREN NUM COMA NUM RPAREN
<Rectangle>   ::= RECTANGLE LPAREN <Point> COMA <Point> RPAREN
<PointList>   ::= <Point> { COMA <Point> }
<Polygon>     ::= POLYGON LPAREN <PointList> RPAREN
<SpatialValue> ::= <Point> | <Rectangle> | <Polygon>
<GeometryArg> ::= <QualifiedName> | <SpatialValue>
<Metric>      ::= HAVERSINE | EUCLIDEAN
<WithinDist>  ::= WITHIN_DISTANCE LPAREN <GeometryArg> COMA <Point> COMA NUM COMA <Metric> RPAREN
<InRange>     ::= IN_RANGE LPAREN <GeometryArg> COMA <Rectangle> RPAREN
<Intersects>  ::= INTERSECTS LPAREN <GeometryArg> COMA <GeometryArg> RPAREN
````

En `<Point>`, el primer número es longitud y el segundo latitud; ambos se
almacenan en grados. Longitud está en `[-180, 180]`, latitud en `[-90, 90]`.
Valores no numéricos, booleanos, NaN, infinitos y coordenadas fuera de rango
son errores; no se recortan ni normalizan.

`RECTANGLE` recibe las esquinas `(oeste, sur)` y `(este, norte)`. Se requiere
`sur <= norte`; `oeste > este` representa un rectángulo que cruza el
antimeridiano. `POLYGON` recibe al menos tres vértices distintos y debe repetir
el primer punto al final. Sus lados son segmentos directos en el plano de
longitud/latitud, por lo que un anillo que cruce el antimeridiano es inválido.

Los rangos incluyen sus bordes y una intersección que solo comparte borde
también cuenta como intersección. Haversine elige el arco longitudinal más
corto, usa radio terrestre `6 371 008.8 m` y devuelve metros; su radio de
consulta también va en metros. Euclidiana opera sobre las coordenadas sin
proyección y devuelve grados; su radio va en unidades de coordenada. Los polos
son válidos: Haversine considera equivalentes sus longitudes, pero los rangos y
Euclidiana conservan los valores numéricos de longitud.

En Python, `NULL` es `None`; un operando nulo produce `NULL` en operaciones
espaciales y no pasa un filtro `WHERE`. Un radio debe ser finito y no negativo.

Ejemplos SQL normativos (la sintaxis está reservada para el soporte futuro):

```sql
CREATE TABLE sitios (ubicacion POINT);
INSERT INTO sitios VALUES (POINT(-122.3, 47.6));
SELECT ubicacion FROM sitios
WHERE WITHIN_DISTANCE(ubicacion, POINT(-122.4, 47.6), 5000, HAVERSINE);
```

```sql
SELECT ubicacion FROM sitios
WHERE WITHIN_DISTANCE(ubicacion, POINT(-122.4, 47.6), 0.1, EUCLIDEAN);

SELECT ubicacion FROM sitios
WHERE IN_RANGE(ubicacion, RECTANGLE(POINT(-123, 47), POINT(-122, 48)));

SELECT ubicacion FROM sitios
WHERE INTERSECTS(
	POLYGON(POINT(-123, 47), POINT(-122, 47), POINT(-122, 48), POINT(-123, 47)),
	RECTANGLE(POINT(-123, 47), POINT(-122, 48))
);
```

```

```
