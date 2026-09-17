"""
Orchestrates the full DML pipeline: SQL text -> tokens -> AST ->
semantic check -> rewrite -> physical plan -> execution.
"""

from query.parser.token_ import TokenType
from query.parser.scanner import Scanner
from query.parser.parser import Parser
from query.parser.visitor import SemanticVisitor, SemanticError
from query.parser.ast_nodes import (
    SelectStm, InsertStm, DeleteStm, CreateTableStm, CreateIndexStm,
)

from query.rewriter.rewriter import rewrite
from query.planner.plan_builder import (
    build_select_plan, execute_insert, execute_delete,
)
from query.executor.executor import run_plan


class QueryError(Exception):
    """Wraps lexical, syntactic, or semantic errors into a single type
    the frontend can catch and display, without needing to know about
    the parser's internal RuntimeError/SemanticError distinction."""


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

    if isinstance(stm, SelectStm):
        plan = build_select_plan(stm, catalog)
        return list(run_plan(plan))

    if isinstance(stm, InsertStm):
        execute_insert(stm, catalog)
        return None

    if isinstance(stm, DeleteStm):
        return execute_delete(stm, catalog)

    if isinstance(stm, (CreateTableStm, CreateIndexStm)):
        return None

    raise QueryError(f"Tipo de sentencia no soportado: {type(stm).__name__}")
