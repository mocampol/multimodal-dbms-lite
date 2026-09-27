from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.parser.parser import Parser
from query.parser.scanner import Scanner
from query.planner.plan_builder import build_select_plan
from query.query_engine import execute


def make_catalog(base_dir, with_rtree):
    heap_factory = make_heap_factory(str(base_dir))
    catalog = Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )
    execute(
        "CREATE TABLE places (id INTEGER PRIMARY KEY, location POINT);",
        catalog,
    )
    execute(
        "INSERT INTO places VALUES "
        "(1, POINT(-77.0428, -12.0464)), "
        "(2, POINT(-77.02, -12.0464)), "
        "(3, POINT(-77.01, -12.01)), "
        "(4, POINT(-76.8, -12.0)), "
        "(5, NULL), "
        "(6, POINT(-77.0656, -12.0464));",
        catalog,
    )
    if with_rtree:
        execute(
            "CREATE INDEX location_idx ON places (location) USING RTREE;",
            catalog,
        )
    return catalog


def row_ids(rows):
    return [row.values[0].data for row in rows]


def plan_nodes(root):
    yield root
    for child in root.children():
        yield from plan_nodes(child)


def parse(sql):
    return Parser(Scanner(sql)).parse_sql_statements()[0]


def test_radius_query_matches_with_and_without_rtree(tmp_path):
    sql = (
        "SELECT id FROM places WHERE distancia(location, "
        "POINT(-77.0428, -12.0464), HAVERSINE) < 5000;"
    )
    indexed = make_catalog(tmp_path / "indexed", with_rtree=True)
    sequential = make_catalog(tmp_path / "sequential", with_rtree=False)

    indexed_ids = row_ids(execute(sql, indexed))
    sequential_ids = row_ids(execute(sql, sequential))

    assert indexed_ids == sequential_ids == [1, 2, 6]


def test_euclidean_radius_query_matches_with_and_without_rtree(tmp_path):
    sql = (
        "SELECT id FROM places WHERE distancia(location, "
        "POINT(-77.0428, -12.0464), EUCLIDEAN) < 0.03;"
    )
    indexed = make_catalog(tmp_path / "indexed", with_rtree=True)
    sequential = make_catalog(tmp_path / "sequential", with_rtree=False)

    indexed_ids = row_ids(execute(sql, indexed))
    sequential_ids = row_ids(execute(sql, sequential))

    assert indexed_ids == sequential_ids == [1, 2, 6]


def test_polygon_query_exactly_filters_rtree_bbox_candidates(tmp_path):
    sql = (
        "SELECT id FROM places WHERE dentro_de(location, POLYGON("
        "POINT(-77.25, -12.2), POINT(-76.85, -12.2), "
        "POINT(-77.25, -11.8), POINT(-77.25, -12.2)));"
    )
    indexed = make_catalog(tmp_path / "indexed", with_rtree=True)
    sequential = make_catalog(tmp_path / "sequential", with_rtree=False)

    indexed_ids = row_ids(execute(sql, indexed))
    sequential_ids = row_ids(execute(sql, sequential))

    assert indexed_ids == sequential_ids == [1, 2, 6]


def test_knn_orders_by_exact_haversine_distance_and_limits_rows(tmp_path):
    sql = (
        "SELECT id FROM places ORDER BY distancia(location, "
        "POINT(-77.0428, -12.0464), HAVERSINE) LIMIT 3;"
    )
    indexed = make_catalog(tmp_path / "indexed", with_rtree=True)
    sequential = make_catalog(tmp_path / "sequential", with_rtree=False)

    indexed_ids = row_ids(execute(sql, indexed))
    sequential_ids = row_ids(execute(sql, sequential))

    assert indexed_ids == sequential_ids == [1, 2, 6]
    assert len(indexed_ids) == 3


def test_planner_selects_spatial_index_or_sequential_fallback(tmp_path):
    sql = (
        "SELECT id FROM places WHERE distancia(location, "
        "POINT(-77.0428, -12.0464)) < 5000;"
    )
    indexed = make_catalog(tmp_path / "indexed", with_rtree=True)
    sequential = make_catalog(tmp_path / "sequential", with_rtree=False)

    indexed_plan = build_select_plan(parse(sql), indexed)
    sequential_plan = build_select_plan(parse(sql), sequential)

    assert "SpatialIndexScan" in {type(node).__name__ for node in plan_nodes(indexed_plan)}
    assert "SpatialFilter" in {type(node).__name__ for node in plan_nodes(indexed_plan)}
    assert "SeqScan" in {type(node).__name__ for node in plan_nodes(sequential_plan)}
    assert "SpatialFilter" in {type(node).__name__ for node in plan_nodes(sequential_plan)}


def test_knn_plan_uses_index_and_places_limit_after_sort(tmp_path):
    sql = (
        "SELECT id FROM places ORDER BY distancia(location, "
        "POINT(-77.0428, -12.0464)) LIMIT 2;"
    )
    catalog = make_catalog(tmp_path / "indexed", with_rtree=True)

    plan = build_select_plan(parse(sql), catalog)
    names = [type(node).__name__ for node in plan_nodes(plan)]

    assert "SpatialIndexScan" in names
    assert "KNNScan" in names
    assert "Limit" in names
    assert names.index("Projection") < names.index("Limit") < names.index("KNNScan")