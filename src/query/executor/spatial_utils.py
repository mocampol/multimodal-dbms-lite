from common.value import Point
from query.parser.ast_nodes import IdExp, PointExp
from spatial.geometry import Point2D, Polygon


def column_index(schema, name: str) -> int:
    column = schema.get_column(name)
    if column is not None:
        return schema.column_index(column.name)

    suffix = name.rsplit(".", 1)[-1]
    matches = [
        column.name
        for column in schema.columns
        if column.name.rsplit(".", 1)[-1] == suffix
    ]
    if len(matches) != 1:
        raise ValueError(f"La columna '{name}' no se puede resolver en '{schema.table_name}'")
    return schema.column_index(matches[0])


def point2d_from_value(value) -> Point2D | None:
    if isinstance(value, Point2D):
        return value
    if isinstance(value, Point):
        return Point2D(value.longitude, value.latitude)
    return None


def point2d_from_expression(expression, record, schema) -> Point2D | None:
    if isinstance(expression, PointExp):
        return point2d_from_value(expression.to_point())
    if isinstance(expression, IdExp) and record is not None:
        return point2d_from_value(record[column_index(schema, expression.value)].data)
    return None


def polygon_from_literal(expression) -> Polygon:
    points = [
        Point2D(point.longitude, point.latitude)
        for point in expression.points
    ]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return Polygon(tuple(points))