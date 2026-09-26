"""
Builds a physical plan tree directly from a (rewritten, semantically
validated) AST statement — no cost-based Optimizer, per spec. Access
method selection follows a simple heuristic: use an index if the
catalog reports one over the WHERE column, otherwise fall back to a
Sequential Scan.
"""

from query.parser.ast_nodes import SelectStm, InsertStm, DeleteStm, UpdateStm, IdExp, BinaryExp, AggregateSpec
from common.value import DataType, Value
from common.record import Record
from common.schema import Schema, Column
from transaction.lock_manager import LockMode

from query.executor.access.seq_scan import SeqScan
from query.executor.access.index_scan import IndexScan
from query.executor.processing.filter import Filter
from query.executor.processing.projection import Projection
from query.executor.processing.sort import Sort
from query.executor.aggregate.group_aggregate import GroupAggregate
from query.executor.aggregate.hash_aggregate import HashAggregate
from query.executor.join.hash_join import HashJoin


def build_select_plan(stm: SelectStm, catalog, lock_rid=None):
    """
    Builds the physical plan for a SELECT statement:
    source -> Filter -> Group/Aggregate -> Sort -> Projection.
    """
    input_schema = catalog.get_schema(stm.table)

    if stm.join is not None:
        right_schema = catalog.get_schema(stm.join.table)
        left_join_table, left_join_column = stm.join.left.split(".", 1)
        right_join_table, right_join_column = stm.join.right.split(".", 1)
        left_schema = input_schema if left_join_table == stm.table else right_schema
        left_index = (
            input_schema.column_index(left_join_column)
            if left_join_table == stm.table
            else right_schema.column_index(left_join_column)
        )
        right_index = (
            right_schema.column_index(right_join_column)
            if right_join_table == stm.join.table
            else input_schema.column_index(right_join_column)
        )
        node = HashJoin(
            SeqScan(stm.table, catalog, lock_rid=lock_rid),
            SeqScan(stm.join.table, catalog, lock_rid=lock_rid),
            left_index,
            right_index,
        )
        schema = _join_schema(stm.table, input_schema, stm.join.table, right_schema)
    else:
        schema = input_schema
        node = SeqScan(stm.table, catalog, lock_rid=lock_rid)
        indexed = _equality_index(catalog, stm.table, stm.where_cond)
        if indexed is not None:
            index, key = indexed
            node = IndexScan(index, catalog.get_storage(stm.table), key, lock_rid=lock_rid)

    physical_condition = _qualify_condition(stm.where_cond, schema)
    if physical_condition is not None and (
        stm.join is not None or _equality_index(catalog, stm.table, stm.where_cond) is None
    ):
        node = Filter(node, physical_condition, schema)

    aggregate_items = [item for item in stm.columns if isinstance(item, AggregateSpec)]
    if aggregate_items:
        group_columns = [
            _resolve_column_name(schema, name) for name in stm.group_by.columns
        ] if stm.group_by is not None else []
        physical_items = []
        for item in stm.columns:
            if isinstance(item, AggregateSpec):
                physical_item = AggregateSpec(item.function, item.column)
                physical_item.resolved_column = (
                    None if item.column == "*" else _resolve_column_name(schema, item.column)
                )
                physical_items.append(physical_item)
            else:
                physical_items.append(_resolve_column_name(schema, item))
        node = HashAggregate(node, group_columns, schema, physical_items)
        schema = _aggregate_schema(schema, physical_items)
        projection_columns = [
            item.name if isinstance(item, AggregateSpec) else item
            for item in physical_items
        ]
    else:
        projection_columns = [
            _resolve_column_name(schema, name) for name in stm.columns
        ] if stm.columns != ["*"] else ["*"]
        if stm.group_by is not None:
            group_columns = [_resolve_column_name(schema, name) for name in stm.group_by.columns]
            node = Sort(node, group_columns, schema)
            node = GroupAggregate(node, group_columns, schema)

    if stm.order_by is not None:
        order_columns = [_resolve_column_name(schema, name) for name in stm.order_by.columns]
        node = Sort(node, order_columns, schema)

    return Projection(node, projection_columns, schema)


def _join_schema(left_name, left_schema, right_name, right_schema):
    columns = [
        Column(f"{left_name}.{column.name}", column.data_type, column.size, nullable=column.nullable)
        for column in left_schema.columns
    ] + [
        Column(f"{right_name}.{column.name}", column.data_type, column.size, nullable=column.nullable)
        for column in right_schema.columns
    ]
    return Schema(f"{left_name}_join_{right_name}", columns)


def _resolve_column_name(schema, name):
    if name == "*":
        return name
    if schema.get_column(name) is not None:
        return name
    if "." in name:
        short_name = name.rsplit(".", 1)[1]
        if schema.get_column(short_name) is not None:
            return short_name
    matches = [column.name for column in schema.columns if column.name.rsplit(".", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"La columna '{name}' no se puede resolver en '{schema.table_name}'")
    return matches[0]


def _qualify_condition(condition, schema):
    if condition is None:
        return None
    left = IdExp(_resolve_column_name(schema, condition.left.value))
    right = condition.right
    if isinstance(right, IdExp):
        right = IdExp(_resolve_column_name(schema, right.value))
    return BinaryExp(left, right, condition.op)


def _aggregate_schema(input_schema, output_items):
    columns = []
    for item in output_items:
        if not isinstance(item, AggregateSpec):
            source = input_schema.get_column(item)
            columns.append(Column(source.name, source.data_type, source.size, nullable=source.nullable))
            continue
        if item.function == "COUNT":
            data_type = DataType.INTEGER
            size = None
        else:
            source = input_schema.get_column(item.resolved_column)
            data_type = DataType.DOUBLE_PRECISION if item.function == "AVG" else source.data_type
            size = source.size
        columns.append(Column(item.name, data_type, size, nullable=True))
    return Schema(f"aggregate_{input_schema.table_name}", columns)


def execute_insert(stm: InsertStm, catalog, lock_rid=None, before_insert=None) -> None:
    """
    INSERT has no plan tree: it writes each literal row directly to storage.
    """
    schema = catalog.get_schema(stm.table)
    storage = catalog.get_storage(stm.table)

    column_positions = (
        {name: schema.column_index(name) for name in stm.columns}
        if stm.columns is not None else None
    )

    rids = []
    for values in stm.values:
        if column_positions is not None:
            row_values = [None] * len(schema.columns)
            for name, exp in zip(stm.columns, values):
                idx = column_positions[name]
                col = schema.columns[idx]
                row_values[idx] = Value(col.data_type, exp.value)
            for idx, col in enumerate(schema.columns):
                if row_values[idx] is None:
                    row_values[idx] = Value(col.data_type, None)
            record = Record(row_values)
        else:
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
