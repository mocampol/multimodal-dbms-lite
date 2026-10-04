from common.value import Point, Polygon as ValuePolygon, Rectangle
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
    if isinstance(value, ValuePolygon):
        unique_points = value.points[:-1]
        avg_lon = sum(p.longitude for p in unique_points) / len(unique_points)
        avg_lat = sum(p.latitude for p in unique_points) / len(unique_points)
        return Point2D(avg_lon, avg_lat)
    if isinstance(value, Rectangle):
        avg_lon = (value.west + value.east) / 2.0
        avg_lat = (value.south + value.north) / 2.0
        return Point2D(avg_lon, avg_lat)
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