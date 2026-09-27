from query.executor.processing.projection import Projection
from query.executor.aggregate.hash_aggregate import HashAggregate
from query.parser.ast_nodes import AggregateSpec
from query.query_engine import describe_plan as _describe_plan, unwrap_instrumented


def describe_plan(node) -> dict:
    return _describe_plan(node)


def output_columns(root) -> list[str]:
    root = unwrap_instrumented(root)
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


def serialize_explain_result(result) -> dict:
    return {
        "type": "explain_analyze" if result.analyze else "explain",
        "plan": result.description,
        "columns": output_columns(result.plan_root),
        "row_count": result.row_count,
        "elapsed_ms": result.elapsed_ms,
    }
