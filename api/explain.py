from query.executor.access.seq_scan import SeqScan
from query.executor.access.index_scan import IndexScan
from query.executor.access.clustered_scan import ClusteredScan
from query.executor.processing.filter import Filter
from query.executor.processing.projection import Projection
from query.executor.processing.sort import Sort
from query.executor.aggregate.group_aggregate import GroupAggregate
from query.executor.aggregate.hash_aggregate import HashAggregate
from query.executor.join.hash_join import HashJoin
from query.parser.ast_nodes import AggregateSpec


def describe_plan(node) -> dict:
    info = {"node": type(node).__name__}
    children = []

    if isinstance(node, SeqScan):
        info["access"] = "sequential_scan"
        info["table"] = node.table_name
    elif isinstance(node, IndexScan):
        info["access"] = "index_scan"
        info["index_type"] = type(node.index).__name__
        info["key"] = node.key.data
    elif isinstance(node, ClusteredScan):
        info["access"] = "clustered_index_scan"
        info["key"] = node.key.data
    elif isinstance(node, Filter):
        info["condition"] = repr(node.condition)
        children = [node.child]
    elif isinstance(node, Projection):
        info["columns"] = node.columns
        children = [node.child]
    elif isinstance(node, Sort):
        info["order_by"] = node.columns
        info["strategy"] = "external_sort_k_way_merge"
        children = [node.child]
    elif isinstance(node, HashAggregate):
        info["group_by"] = node.group_columns
        info["strategy"] = "external_hash_partitioning"
        children = [node.child]
    elif isinstance(node, GroupAggregate):
        info["strategy"] = "external_sort_then_stream"
        children = [node.child]
    elif isinstance(node, HashJoin):
        info["join_type"] = "hash"
        info["strategy"] = "grace_hash_external"
        children = [node.left, node.right]

    if children:
        info["children"] = [describe_plan(child) for child in children]
    return info


def output_columns(root) -> list[str]:
    if isinstance(root, Projection):
        if root.columns == ["*"]:
            return [c.name for c in root.schema.columns]
        return list(root.columns)
    if isinstance(root, HashAggregate):
        return [
            f"{item.function}({item.column})" if isinstance(item, AggregateSpec) else item
            for item in root.output_items
        ]
    return []
