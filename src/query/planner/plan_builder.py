"""
Builds a physical plan tree directly from a (rewritten, semantically
validated) AST statement — no cost-based Optimizer, per spec. Access
method selection follows a simple heuristic: use an index if the
catalog reports one over the WHERE column, otherwise fall back to a
Sequential Scan.
"""

from query.parser.ast_nodes import SelectStm, InsertStm, DeleteStm, UpdateStm, IdExp, BinaryExp, AggregateSpec
from common.value import Value
from common.record import Record
from common.schema import Schema, Column
from transaction.lock_manager import LockMode

from query.executor.access.seq_scan import SeqScan
from query.executor.access.index_scan import IndexScan
from query.executor.access.clustered_scan import ClusteredScan
from query.executor.processing.filter import Filter
from query.executor.processing.projection import Projection
from query.executor.processing.sort import Sort
from query.executor.aggregate.group_aggregate import GroupAggregate
from query.executor.aggregate.hash_aggregate import HashAggregate
from query.executor.join.hash_join import HashJoin


def build_select_plan(stm: SelectStm, catalog, lock_rid=None):
    """
    Builds the physical plan for a SELECT statement:
    (Sequential/Index Scan) -> [Filter] -> [Sort] -> Projection
    """
    schema = catalog.get_schema(stm.table)

    if stm.where_cond is not None and catalog.get_clustered_index(stm.table) is not None:
        if isinstance(stm.where_cond, BinaryExp) and stm.where_cond.op.name == "EQ_OP":
            clustered = catalog.get_clustered_index(stm.table)
            key_column = schema.primary_key()
            if stm.where_cond.left.value == key_column.name and not isinstance(stm.where_cond.right, IdExp):
                key = Value(key_column.data_type, stm.where_cond.right.value)
                return Projection(ClusteredScan(clustered, key), stm.columns, schema)

    aggregate_items = [item for item in stm.columns if isinstance(item, AggregateSpec)]
    if aggregate_items:
        if stm.join is not None:
            raise ValueError("Agregaciones sobre JOIN aún no están soportadas")
        child = SeqScan(stm.table, catalog, lock_rid=lock_rid)
        if stm.where_cond is not None:
            child = Filter(child, stm.where_cond, schema)
        aggregate = HashAggregate(child, stm.group_by.columns if stm.group_by else [], schema, stm.columns)
        return aggregate

    if stm.join is not None:
        right_schema = catalog.get_schema(stm.join.table)
        left_join_table, left_join_column = stm.join.left.split(".", 1)
        right_join_table, right_join_column = stm.join.right.split(".", 1)
        left_schema = schema if left_join_table == stm.table else right_schema
        right_join_schema = schema if right_join_table == stm.table else right_schema
        left_index = left_schema.column_index(left_join_column)
        right_index = right_join_schema.column_index(right_join_column)
        node = HashJoin(
            SeqScan(stm.table, catalog, lock_rid=lock_rid),
            SeqScan(stm.join.table, catalog, lock_rid=lock_rid),
            left_index if left_join_table == stm.table else right_index,
            right_index if right_join_table == stm.join.table else left_index,
        )
        combined_columns = [
            Column(f"{stm.table}.{column.name}", column.data_type, column.size, nullable=column.nullable)
            for column in schema.columns
        ] + [
            Column(f"{stm.join.table}.{column.name}", column.data_type, column.size, nullable=column.nullable)
            for column in right_schema.columns
        ]
        combined_schema = Schema(f"{stm.table}_join_{stm.join.table}", combined_columns)
        selected = stm.columns
        if selected != ["*"]:
            selected = [
                name if "." in name else f"{stm.table}.{name}"
                for name in selected
            ]
        node = Projection(node, selected, combined_schema)
        return node

    # Equality predicates use a physical index when one is available.
    node = SeqScan(stm.table, catalog, lock_rid=lock_rid)
    indexed = _equality_index(catalog, stm.table, stm.where_cond)
    if indexed is not None:
        index, key = indexed
        node = IndexScan(index, catalog.get_storage(stm.table), key, lock_rid=lock_rid)

    if stm.where_cond is not None and indexed is None:
        node = Filter(node, stm.where_cond, schema)

    if stm.group_by is not None:
        node = Sort(node, stm.group_by.columns, schema)
        node = GroupAggregate(node, stm.group_by.columns, schema)

    if stm.order_by is not None:
        node = Sort(node, stm.order_by.columns, schema)

    node = Projection(node, stm.columns, schema)
    return node


def execute_insert(stm: InsertStm, catalog, lock_rid=None, before_insert=None) -> None:
    """
    INSERT has no plan tree: it writes each literal row directly to storage.
    """
    schema = catalog.get_schema(stm.table)
    storage = catalog.get_storage(stm.table)

    rids = []
    for values in stm.values:
        record = Record([Value(col.data_type, exp.value) for col, exp in zip(schema.columns, values)])
        violated = catalog.check_insert_uniques(stm.table, record)
        if violated:
            raise ValueError(f"Valor duplicado en columna UNIQUE '{violated}' de '{stm.table}'")

        if before_insert is not None:
            before_insert(record)
        rid = storage.insert(record)
        if lock_rid is not None:
            lock_rid(rid, LockMode.EXCLUSIVE)
        catalog.register_insert(stm.table, record, rid)
        catalog.register_insert_uniques(stm.table, record)
        rids.append(rid)
    return rids


def execute_delete(stm: DeleteStm, catalog, lock_rid=None, before_delete=None) -> int:
    """
    DELETE also has no meaningful plan tree to stream to a consumer —
    it's a scan-and-mutate operation. Reuses SeqScan (+ Filter, if there's
    a WHERE) purely as a convenient way to iterate matching records, but
    calls storage.delete() as a side effect rather than yielding rows.

    HeapFile exposes stable RIDs for this operation. Sequential storage
    supports the same operation only for its current equality-by-key API.
    """
    schema = catalog.get_schema(stm.table)
    storage = catalog.get_storage(stm.table)

    if hasattr(storage, "scan_with_rid"):
        matches = []
        for rid, record in storage.scan_with_rid():
            if stm.where_cond is None or _condition_matches(stm.where_cond, record, schema):
                if lock_rid is not None:
                    lock_rid(rid, LockMode.EXCLUSIVE)
                matches.append((rid, record))
        for rid, record in matches:
            if before_delete is not None:
                before_delete(rid, record)
            catalog.unregister_delete(stm.table, record, rid)
            storage.delete(rid)
        return len(matches)

    if stm.where_cond is not None and isinstance(stm.where_cond, BinaryExp):
        if isinstance(stm.where_cond.right, IdExp):
            raise ValueError("DELETE sobre SequentialFile requiere un literal como clave")
        if stm.where_cond.op.name != "EQ_OP":
            raise ValueError("DELETE sobre SequentialFile solo soporta igualdad")
        deleted = storage.delete(stm.where_cond.right.value)
        return deleted

    raise ValueError("DELETE sin filtro sobre SequentialFile no está soportado")


def execute_update(stm: UpdateStm, catalog, lock_rid=None, on_update=None) -> int:
    schema = catalog.get_schema(stm.table)
    storage = catalog.get_storage(stm.table)
    if not hasattr(storage, "scan_with_rid"):
        raise ValueError("UPDATE requiere HeapFile con RIDs")
    column_index = schema.column_index(stm.column)
    new_value = Value(schema.columns[column_index].data_type, stm.value.value)
    matches = []
    for rid, record in storage.scan_with_rid():
        if _condition_matches(stm.where_cond, record, schema):
            if lock_rid is not None:
                lock_rid(rid, LockMode.EXCLUSIVE)
            values = list(record.values)
            values[column_index] = new_value
            matches.append((rid, record, Record(values)))
    for rid, old_record, new_record in matches:
        new_rid = storage.update(rid, new_record)
        if on_update is not None:
            on_update(rid, new_rid, old_record, new_record)
        catalog.register_update(stm.table, rid, new_rid, old_record, new_record)
    return len(matches)


def _equality_index(catalog, table_name, condition):
    if not isinstance(condition, BinaryExp) or condition.op.name != "EQ_OP":
        return None
    if not isinstance(condition.right, (IdExp,)) and not hasattr(condition.right, "value"):
        return None
    column_name = condition.left.value
    index = catalog.get_physical_index(table_name, column_name)
    if index is None or isinstance(condition.right, IdExp):
        return None
    schema = catalog.get_schema(table_name)
    return index, Value(schema.get_column(column_name).data_type, condition.right.value)


def _condition_matches(condition, record: Record, schema) -> bool:
    left_value = record[schema.column_index(condition.left.value)].data
    if isinstance(condition.right, IdExp):
        right_value = record[schema.column_index(condition.right.value)].data
    else:
        right_value = condition.right.value

    operations = {
        "EQ_OP": lambda: left_value == right_value,
        "NEQ_OP": lambda: left_value != right_value,
        "LE_OP": lambda: left_value < right_value,
        "LEQ_OP": lambda: left_value <= right_value,
        "GT_OP": lambda: left_value > right_value,
        "GEQ_OP": lambda: left_value >= right_value,
    }
    return operations[condition.op.name]()
