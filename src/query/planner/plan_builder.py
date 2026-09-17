"""
Builds a physical plan tree directly from a (rewritten, semantically
validated) AST statement — no cost-based Optimizer, per spec. Access
method selection follows a simple heuristic: use an index if the
catalog reports one over the WHERE column, otherwise fall back to a
Sequential Scan.
"""

from query.parser.ast_nodes import SelectStm, InsertStm, DeleteStm, IdExp, BinaryExp
from common.value import Value
from common.record import Record

from query.executor.access.seq_scan import SeqScan
from query.executor.access.index_scan import IndexScan
from query.executor.processing.filter import Filter
from query.executor.processing.projection import Projection
from query.executor.processing.sort import Sort
from query.executor.aggregate.group_aggregate import GroupAggregate


def build_select_plan(stm: SelectStm, catalog):
    """
    Builds the physical plan for a SELECT statement:
    (Sequential/Index Scan) -> [Filter] -> [Sort] -> Projection
    """
    schema = catalog.get_schema(stm.table)

    # Access method: SeqScan is the only one implemented so far.
    # TODO once index_scan.py is wired up: if stm.where_cond is a
    # BinaryExp comparing an indexed column with EQ_OP, and
    # catalog.get_indexes(stm.table) has a matching entry, use
    # IndexScan(stm.table, stm.where_cond, catalog) instead, and skip
    # wrapping in Filter below (the index scan already applies the
    # condition). This is exactly the "regla heurística de selección
    # "access selection heuristic" the assignment asks for.
    node = SeqScan(stm.table, catalog)
    indexed = _equality_index(catalog, stm.table, stm.where_cond)
    if indexed is not None:
        index, key = indexed
        node = IndexScan(index, catalog.get_storage(stm.table), key)

    if stm.where_cond is not None and indexed is None:
        node = Filter(node, stm.where_cond, schema)

    if stm.group_by is not None:
        node = Sort(node, stm.group_by.columns, schema)
        node = GroupAggregate(node, stm.group_by.columns, schema)

    if stm.order_by is not None:
        node = Sort(node, stm.order_by.columns, schema)

    node = Projection(node, stm.columns, schema)
    return node


def execute_insert(stm: InsertStm, catalog) -> None:
    """
    INSERT has no plan tree: it's a single direct write, not a pull-based
    stream of tuples. Builds a Record from the AST's literal values (in
    schema column order, already validated by SemanticVisitor) and
    writes it straight to storage.
    """
    schema = catalog.get_schema(stm.table)
    storage = catalog.get_storage(stm.table)

    violated = catalog.check_insert_uniques(
        stm.table,
        Record([Value(col.data_type, exp.value) for col, exp in zip(schema.columns, stm.values)]),
    )
    if violated:
        raise ValueError(f"Valor duplicado en columna UNIQUE '{violated}' de '{stm.table}'")

    record = Record([Value(col.data_type, exp.value) for col, exp in zip(schema.columns, stm.values)])
    rid = storage.insert(record)
    catalog.register_insert(stm.table, record, rid)
    catalog.register_insert_uniques(stm.table, record)


def execute_delete(stm: DeleteStm, catalog) -> int:
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
                matches.append(rid)
        for rid in matches:
            record = storage.get(rid)
            if record is not None:
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
