from types import SimpleNamespace

from common import Column, DataType, Schema
from common.value import Distance, DistanceUnit
from explain import serialize_explain_result
from query.executor.access.spatial_scan import KNNScan, SpatialIndexScan
from query.executor.processing.limit import Limit
from query.executor.processing.projection import Projection
from query.executor.processing.spatial_filter import SpatialFilter
from query.parser.ast_nodes import BinaryOp, DistanceExp, IdExp, NumExp, PointLiteral, SpatialPredicate
from query.query_engine import ExplainResult
from serialize import serialize_value
from spatial.geometry import BoundingBox, Point2D


def test_serialize_point2d_and_calculated_distance():
    point = serialize_value(SimpleNamespace(data=Point2D(-77.04, -12.05)))
    distance = serialize_value(
        SimpleNamespace(data=Distance(250.0, DistanceUnit.METERS))
    )

    assert point == {"longitude": -77.04, "latitude": -12.05}
    assert distance == {"value": 250.0, "unit": "meters"}


def test_explain_response_includes_spatial_index_knn_filter_and_limit():
    schema = Schema("places", [Column("location", DataType.POINT)])
    distance = DistanceExp(
        IdExp("location"), PointLiteral(-77.04, -12.05), "HAVERSINE"
    )
    predicate = SpatialPredicate(distance, BinaryOp.LE_OP, NumExp(500))
    scan = SpatialIndexScan(
        index=object(),
        storage=object(),
        table_name="places",
        boxes=[BoundingBox(-78, -13, -76, -11)],
        index_id=9,
        column_name="location",
    )
    root = Projection(
        Limit(
            KNNScan(SpatialFilter(scan, predicate, schema), [distance], schema),
            5,
        ),
        ["location"],
        schema,
    )

    response = serialize_explain_result(ExplainResult(root, analyze=False))
    nodes = []

    def collect(node):
        nodes.append(node)
        for child in node.get("children", []):
            collect(child)

    collect(response["plan"])
    by_name = {node["node"]: node for node in nodes}

    assert response["type"] == "explain"
    assert response["columns"] == ["location"]
    assert by_name["SpatialIndexScan"]["index_type"] == "rtree"
    assert by_name["SpatialIndexScan"]["index_id"] == 9
    assert by_name["SpatialIndexScan"]["column_name"] == "location"
    assert by_name["KNNScan"]["metric"] == "HAVERSINE"
    assert by_name["SpatialFilter"]["predicate"]
    assert by_name["Limit"]["limit"] == 5