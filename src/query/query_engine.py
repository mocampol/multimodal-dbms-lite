"""
Orchestrates the full DML pipeline: SQL text -> tokens -> AST ->
semantic check -> rewrite -> physical plan -> execution.
"""

from query.parser.token_ import TokenType
from query.parser.scanner import Scanner
from query.parser.parser import Parser
from query.parser.visitor import SemanticVisitor, SemanticError
from query.parser.ast_nodes import (
    SelectStm, InsertStm, DeleteStm, UpdateStm, CreateTableStm, CreateIndexStm,
    BeginTransactionStm, EndTransactionStm,
)
from transaction import TransactionManager, LockMode

from query.rewriter.rewriter import rewrite
from query.planner.plan_builder import (
    build_select_plan, execute_insert, execute_delete, execute_update,
)
from query.executor.executor import run_plan


class QueryError(Exception):
    """Wraps lexical, syntactic, or semantic errors into a single type
    the frontend can catch and display, without needing to know about
    the parser's internal RuntimeError/SemanticError distinction."""


def _transaction_manager(catalog):
    manager = getattr(catalog, "_transaction_manager", None)
    if manager is None:
        manager = TransactionManager()
        catalog._transaction_manager = manager
    return manager


def execute(sql: str, catalog):
    """
    Runs one SQL statement end to end.

    Returns a list[Record] for SELECT, or None for INSERT/DELETE
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

    if isinstance(stm, BeginTransactionStm):
        return manager.begin()
    if isinstance(stm, EndTransactionStm):
        manager.commit()
        return None

    if isinstance(stm, SelectStm):
        implicit = manager.current() is None
        if implicit:
            manager.begin()
        try:
            lock_rid = lambda rid, mode: manager.lock(f"rid:{stm.table}:{rid}", mode)
            plan = build_select_plan(stm, catalog, lock_rid=lock_rid)
            result = list(run_plan(plan))
            if implicit:
                manager.commit()
            return result
        except Exception:
            if implicit and manager.current() is not None:
                manager.abort()
            raise

    if isinstance(stm, InsertStm):
        storage = catalog.get_storage(stm.table)
        lock_rid = lambda rid, mode: manager.lock(f"rid:{stm.table}:{rid}", mode)
        def insert_operation():
            return execute_insert(
                stm, catalog,
                lock_rid=lock_rid,
                before_insert=lambda record: manager.log_data_change(
                    "INSERT", stm.table, None, None, _record_data(record)
                ),
            )
        return _execute_write(
            manager,
            stm.table,
            insert_operation,
            lambda rid: manager.add_undo(lambda: _undo_insert(catalog, stm.table, storage, rid)),
        )

    if isinstance(stm, DeleteStm):
        storage = catalog.get_storage(stm.table)
        lock_rid = lambda rid, mode: manager.lock(f"rid:{stm.table}:{rid}", mode)
        def before_delete(rid, record):
            manager.log_data_change("DELETE", stm.table, _rid_data(rid), _record_data(record), None)
            manager.add_undo(
                lambda rid=rid, record=record: _undo_delete(
                    catalog, stm.table, storage, record
                )
            )
        return _execute_write(manager, stm.table, lambda: execute_delete(stm, catalog, lock_rid=lock_rid, before_delete=before_delete), None)

    if isinstance(stm, UpdateStm):
        storage = catalog.get_storage(stm.table)
        lock_rid = lambda rid, mode: manager.lock(f"rid:{stm.table}:{rid}", mode)
        def on_update(old_rid, new_rid, old_record, new_record):
            if new_rid != old_rid:
                manager.lock(f"rid:{stm.table}:{new_rid}", LockMode.EXCLUSIVE)
            manager.log_data_change("UPDATE", stm.table, _rid_data(old_rid), _record_data(old_record), _record_data(new_record))
            manager.add_undo(
                lambda old_rid=old_rid, new_rid=new_rid, old_record=old_record, new_record=new_record:
                _undo_update(catalog, stm.table, storage, old_rid, new_rid, new_record, old_record)
            )
        return _execute_write(manager, stm.table, lambda: execute_update(stm, catalog, lock_rid=lock_rid, on_update=on_update), None)

    if isinstance(stm, (CreateTableStm, CreateIndexStm)):
        return None

    raise QueryError(f"Tipo de sentencia no soportado: {type(stm).__name__}")


def _execute_write(manager, table_name, operation, after):
    implicit = manager.current() is None
    if implicit:
        manager.begin()
    try:
        result = operation()
        if after is not None:
            after(result)
        if implicit:
            manager.commit()
        return result
    except Exception:
        if manager.current() is not None:
            manager.abort()
        raise


def _record_data(record):
    return [{"type": value.data_type.value, "data": value.data} for value in record.values]


def _rid_data(rid):
    return {"page_id": rid.page_id, "slot": rid.slot}


def _undo_insert(catalog, table_name, storage, rid):
    record = storage.get(rid)
    if record is not None:
        catalog.unregister_delete(table_name, record, rid)
        storage.delete(rid)


def _undo_delete(catalog, table_name, storage, record):
    rid = storage.insert(record)
    catalog.register_insert(table_name, record, rid)
    catalog.register_insert_uniques(table_name, record)


def _undo_update(catalog, table_name, storage, old_rid, new_rid, current, previous):
    restored_rid = storage.update(new_rid, previous)
    catalog.register_update(table_name, new_rid, restored_rid, current, previous)
