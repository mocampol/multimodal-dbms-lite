"""
Orchestrates the full DML pipeline: SQL text -> tokens -> AST ->
semantic check -> rewrite -> physical plan -> execution.
"""

import time

from query.parser.token_ import TokenType
from query.parser.scanner import Scanner
from query.parser.parser import Parser
from query.parser.visitor import SemanticVisitor
from query.parser.exceptions import SemanticError
from query.parser.ast_nodes import (
    SelectStm, ExplainStm, InsertStm, DeleteStm, UpdateStm, CreateTableStm, CreateIndexStm,
    BeginTransactionStm, EndTransactionStm,
)
from transaction import TransactionManager, rid_resource
from transaction.row_undo import undo_delete, undo_insert, undo_update

from query.rewriter.rewriter import rewrite
from query.planner.plan_builder import (
    build_select_plan, execute_insert, execute_delete, execute_update,
)
from query.executor.executor import run_plan
from query.executor.plan_node import PlanNode


class QueryError(Exception):
    """Wraps lexical, syntactic, or semantic errors into a single type
    the frontend can catch and display, without needing to know about
    the parser's internal RuntimeError/SemanticError distinction."""


class _InstrumentedNode(PlanNode):
    """Decorator around a real PlanNode that measures, for EXPLAIN
    ANALYZE, exactly this node's own next()-time and how many rows it
    produced — without touching the wrapped node's logic at all.

    Built bottom-up by _instrument_tree(): a node's children are wrapped
    first, then spliced back into the node itself via replace_children(),
    so when the node calls self.child.next() (or self.left/self.right),
    it's transparently calling the instrumented child. That's what makes
    the per-node timing exact instead of an estimate: open()/close() are
    also timed but not shown by default, since the numbers that usually
    matter for EXPLAIN ANALYZE are next()-time and row count.
    """

    def __init__(self, wrapped: PlanNode):
        self.wrapped = wrapped
        self.rows_produced = 0
        self.next_time_ms = 0.0
        self.open_time_ms = 0.0

    def open(self) -> None:
        start = time.perf_counter()
        self.wrapped.open()
        self.open_time_ms = (time.perf_counter() - start) * 1000

    def next(self):
        start = time.perf_counter()
        record = self.wrapped.next()
        self.next_time_ms += (time.perf_counter() - start) * 1000
        if record is not None:
            self.rows_produced += 1
        return record

    def close(self) -> None:
        self.wrapped.close()

    def children(self) -> list:
        # wrapped.children() now returns the already-instrumented
        # children, since _instrument_tree() spliced them in via
        # wrapped.replace_children() before wrapping `wrapped` itself.
        return self.wrapped.children()

    def describe_self(self) -> dict:
        return self.wrapped.describe_self()


def _instrument_tree(node: PlanNode) -> _InstrumentedNode:
    """Wraps every node in the plan with _InstrumentedNode, bottom-up."""
    instrumented_children = [_instrument_tree(child) for child in node.children()]
    if instrumented_children:
        node.replace_children(instrumented_children)
    return _InstrumentedNode(node)


def unwrap_instrumented(node: PlanNode) -> PlanNode:
    """Given a node that may or may not be _instrument_tree()-wrapped,
    returns the real underlying PlanNode. Public on purpose: consumers
    outside this module (api/explain.py's output_columns(), for
    instance) need to isinstance()-check the real node type — e.g. "is
    this a Projection?" — and shouldn't have to reach into a private
    class to do it."""
    while isinstance(node, _InstrumentedNode):
        node = node.wrapped
    return node


def describe_plan(node: PlanNode) -> dict:
    """Builds the nested plan description EXPLAIN / EXPLAIN ANALYZE show.

    Works on either a plain plan tree (EXPLAIN: just describe_self() +
    children(), recursively) or an _instrument_tree()-wrapped one
    (EXPLAIN ANALYZE: same recursion, plus each node's own rows/time).
    """
    info = node.describe_self()
    if isinstance(node, _InstrumentedNode):
        info["rows"] = node.rows_produced
        info["time_ms"] = round(node.next_time_ms, 3)
    children = node.children()
    if children:
        info["children"] = [describe_plan(child) for child in children]
    return info


class ExplainResult:
    """Return type for EXPLAIN / EXPLAIN ANALYZE.

    `plan_root` is the actual physical PlanNode built by build_select_plan
    for stm.inner (or its _instrument_tree()-wrapped version for
    ANALYZE) — the same object build_select_plan() would return for a
    plain SELECT, so `description` is guaranteed to describe the query
    that would really run, not an approximation.

    `description` is the nested dict from describe_plan(plan_root),
    ready for api/explain.py to serialize as JSON for the frontend.
    """

    def __init__(self, plan_root, analyze: bool, row_count: int = None, elapsed_ms: float = None):
        self.plan_root = plan_root
        self.analyze = analyze
        self.row_count = row_count
        self.elapsed_ms = elapsed_ms
        self.description = describe_plan(plan_root)

    def __repr__(self):
        base = f"ExplainResult(plan={self.description.get('node')}"
        if self.analyze:
            base += f", rows={self.row_count}, elapsed_ms={self.elapsed_ms:.3f}"
        return base + ")"


def _transaction_manager(catalog):
    manager = getattr(catalog, "_transaction_manager", None)
    if manager is None:
        manager = TransactionManager()
        catalog._transaction_manager = manager
    return manager


def execute(sql: str, catalog):
    """
    Runs one SQL statement end to end.

    Returns a list[Record] for SELECT, an ExplainResult for EXPLAIN /
    EXPLAIN ANALYZE, or None for INSERT/DELETE/UPDATE/CREATE/DROP
    (which mutate storage directly rather than producing a result set).

    Raises QueryError on any lexical, syntactic, or semantic failure.
    """
    statements = _parse_statements(sql)
    if len(statements) != 1:
        raise QueryError("Se recibieron varias sentencias; usa execute_many para ejecutar un bloque")
    return _execute_statement(statements[0], catalog)


def execute_many(sql: str, catalog):
    """Run every semicolon-terminated statement in a SQL block in order."""
    return [
        (stm, _execute_statement(stm, catalog))
        for stm in _parse_statements(sql)
    ]


def _parse_statements(sql: str):
    try:
        return Parser(Scanner(sql)).parse_sql_statements()
    except RuntimeError as e:
        raise QueryError(str(e)) from e


def _execute_statement(stm, catalog):

    try:
        SemanticVisitor(catalog).check(stm)
    except SemanticError as e:
        raise QueryError(str(e)) from e

    stm = rewrite(stm)
    manager = _transaction_manager(catalog)
    def lock_rid(table, rid, mode, timeout=None):
        if timeout is None:
            return manager.lock(rid_resource(table, rid), mode)
        return manager.lock(rid_resource(table, rid), mode, timeout=timeout)

    if isinstance(stm, BeginTransactionStm):
        return manager.begin()
    if isinstance(stm, EndTransactionStm):
        manager.commit()
        return None

    if isinstance(stm, ExplainStm):
        inner = stm.inner

        if not stm.analyze:
            # EXPLAIN puro: solo arma el plan, no lo ejecuta. No toca
            # storage ni locks, así que no necesita transacción.
            plan = build_select_plan(inner, catalog)
            return ExplainResult(plan, analyze=False)

        # EXPLAIN ANALYZE: corre el plan de verdad, con la misma
        # semántica de transacción implícita y locking que un SELECT
        # normal (ver el bloque de SelectStm de abajo), envuelto con
        # _instrument_tree() para medir tiempo y filas por nodo.
        def explain_analyze():
            plan = build_select_plan(inner, catalog, lock_rid=lock_rid)
            instrumented_plan = _instrument_tree(plan)
            start = time.perf_counter()
            rows = list(run_plan(instrumented_plan))
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExplainResult(instrumented_plan, analyze=True, row_count=len(rows), elapsed_ms=elapsed_ms)
        return _run_in_transaction(manager, explain_analyze)

    if isinstance(stm, SelectStm):
        return _run_in_transaction(
            manager,
            lambda: list(run_plan(build_select_plan(stm, catalog, lock_rid=lock_rid))),
        )

    # the hooks below run inside the table latch, so each WAL record is
    # durable before its page can be flushed
    if isinstance(stm, InsertStm):
        storage = catalog.get_storage(stm.table)
        def after_insert(rid, record):
            manager.log_data_change("INSERT", stm.table, rid, None, record)
            # registered per row so a failure mid-batch still undoes earlier rows
            manager.add_undo(lambda: undo_insert(manager, catalog, stm.table, storage, rid))
        return _run_in_transaction(
            manager,
            lambda: execute_insert(stm, catalog, lock_rid=lock_rid, after_insert=after_insert),
            catalog, stm.table,
        )

    if isinstance(stm, DeleteStm):
        storage = catalog.get_storage(stm.table)
        def before_delete(rid, record):
            manager.log_data_change("DELETE", stm.table, rid, record, None)
            manager.add_undo(lambda: undo_delete(manager, catalog, stm.table, storage, record))
        return _run_in_transaction(
            manager,
            lambda: execute_delete(stm, catalog, lock_rid=lock_rid, before_delete=before_delete),
            catalog, stm.table,
        )

    if isinstance(stm, UpdateStm):
        storage = catalog.get_storage(stm.table)
        def on_update(old_rid, new_rid, old_record, new_record):
            manager.log_data_change("UPDATE", stm.table, old_rid, old_record, new_record, new_rid=new_rid)
            manager.add_undo(
                lambda: undo_update(manager, catalog, stm.table, storage, new_rid, new_record, old_record)
            )
        return _run_in_transaction(
            manager,
            lambda: execute_update(stm, catalog, lock_rid=lock_rid, on_update=on_update),
            catalog, stm.table,
        )

    if isinstance(stm, (CreateTableStm, CreateIndexStm)):
        return None

    raise QueryError(f"Tipo de sentencia no soportado: {type(stm).__name__}")


def _run_in_transaction(manager, operation, catalog=None, written_table=None):
    """
    Runs `operation` inside the thread's transaction, opening an implicit
    one if none is active. Any failure (including a DeadlockError) aborts
    the transaction so its locks are released and its changes undone.
    `written_table`'s pages are flushed before the transaction ends (FORCE).
    """
    implicit = manager.current() is None
    if implicit:
        manager.begin()
    if written_table is not None:
        manager.add_flush(
            ("table", written_table), lambda: catalog.flush_table(written_table)
        )
    try:
        result = operation()
        if implicit:
            manager.commit()
        return result
    except Exception:
        if manager.current() is not None:
            manager.abort()
        raise

