"""
Builds a physical plan tree directly from a (rewritten, semantically
validated) AST statement — no cost-based Optimizer, per spec. Access
method selection follows a simple heuristic: use an index if the
catalog reports one over the WHERE column, otherwise fall back to a
Sequential Scan.
"""

from ast_nodes import SelectStm, InsertStm, DeleteStm, IdExp
from common.value import Value
from common.record import Record

from query.executor.access.seq_scan import SeqScan
from query.executor.processing.filter import Filter
from query.executor.processing.projection import Projection
from query.executor.processing.sort import Sort


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
    # de acceso" the assignment asks for.
    node = SeqScan(stm.table, catalog)

    if stm.where_cond is not None:
        node = Filter(node, stm.where_cond, schema)

    if stm.order_by is not None:
        node = Sort(node, stm.order_by.columns, schema)

    # GROUP BY without aggregate functions in the grammar has no
    # meaningful physical operator yet — left unimplemented on purpose,
    # see aggregate/ placeholders.

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
    storage.insert(record)
    catalog.register_insert_uniques(stm.table, record)


def execute_delete(stm: DeleteStm, catalog) -> int:
    """
    DELETE also has no meaningful plan tree to stream to a consumer —
    it's a scan-and-mutate operation. Reuses SeqScan (+ Filter, if there's
    a WHERE) purely as a convenient way to iterate matching records, but
    calls storage.delete() as a side effect rather than yielding rows.

    NOTE: HeapFile.scan() doesn't expose the RID of each record it
    yields, only the Record itself — this means delete-by-scan can't
    call storage.delete(rid) directly yet. Flagging this as a real gap:
    either HeapFile.scan() needs to optionally yield (RID, Record) pairs,
    or DeleteStm execution needs its own storage-level method that scans
    and deletes matching records internally, page by page. Left as an
    explicit TODO rather than a bad workaround.
    """
    raise NotImplementedError(
        "DELETE requiere que HeapFile.scan() exponga el RID de cada "
        "registro (actualmente solo yield-ea el Record). Ver el "
        "docstring de execute_delete para las dos alternativas de diseño."
    )
