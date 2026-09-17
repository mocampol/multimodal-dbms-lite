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
    try:
        scanner = Scanner(sql)
        parser = Parser(scanner)
        stm = parser.parse_sql_statement()
    except RuntimeError as e:
        raise QueryError(str(e)) from e

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
            manager.lock(f"table:{stm.table}", LockMode.SHARED)
            plan = build_select_plan(stm, catalog)
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
        def insert_operation():
            return execute_insert(
                stm, catalog,
                before_insert=lambda record: manager.log_data_change(
                    "INSERT", stm.table, None, None, _record_data(record)
                ),
            )
        return _execute_write(manager, stm.table, insert_operation, lambda rid: manager.add_undo(lambda: storage.delete(rid)))

    if isinstance(stm, DeleteStm):
        storage = catalog.get_storage(stm.table)
        def before_delete(rid, record):
            manager.log_data_change("DELETE", stm.table, _rid_data(rid), _record_data(record), None)
            manager.add_undo(lambda rid=rid, record=record: storage.insert(record))
        return _execute_write(manager, stm.table, lambda: execute_delete(stm, catalog, before_delete), None)

    if isinstance(stm, UpdateStm):
        storage = catalog.get_storage(stm.table)
        def before_update(rid, old_record, new_record):
            manager.log_data_change("UPDATE", stm.table, _rid_data(rid), _record_data(old_record), _record_data(new_record))
            manager.add_undo(lambda rid=rid, old_record=old_record: storage.update(rid, old_record))
        return _execute_write(manager, stm.table, lambda: execute_update(stm, catalog, before_update), None)

    if isinstance(stm, (CreateTableStm, CreateIndexStm)):
        return None

    raise QueryError(f"Tipo de sentencia no soportado: {type(stm).__name__}")


def _execute_write(manager, table_name, operation, after):
    implicit = manager.current() is None
    if implicit:
        manager.begin()
    try:
        manager.lock(f"table:{table_name}", LockMode.EXCLUSIVE)
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
