import time

from fastapi import APIRouter

from engine import catalog
from explain import describe_plan, output_columns
from schemas import QueryRequest
from serialize import serialize_record

from query.parser.scanner import Scanner
from query.parser.parser import Parser
from query.parser.ast_nodes import SelectStm, InsertStm, UpdateStm, DeleteStm, BeginTransactionStm
from query.rewriter.rewriter import rewrite
from query.planner.plan_builder import build_select_plan
from query.query_engine import execute

router = APIRouter()


def _classify(sql: str):
    try:
        return Parser(Scanner(sql)).parse_sql_statement()
    except Exception:
        return None


@router.post("/query")
def run_query(payload: QueryRequest):
    stm = _classify(payload.sql)

    t0 = time.perf_counter()
    result = execute(payload.sql, catalog)
    execution_ms = (time.perf_counter() - t0) * 1000

    if isinstance(stm, SelectStm):
        root = build_select_plan(rewrite(stm), catalog)
        return {
            "type": "SELECT",
            "columns": output_columns(root),
            "rows": [serialize_record(r) for r in result],
            "row_count": len(result),
            "execution_ms": execution_ms,
            "plan": describe_plan(root),
        }

    if isinstance(stm, (InsertStm, UpdateStm, DeleteStm)):
        return {
            "type": type(stm).__name__.replace("Stm", "").upper(),
            "rows_affected": result if isinstance(result, int) else 1,
            "execution_ms": execution_ms,
            "plan": None,
        }

    response = {
        "type": type(stm).__name__.replace("Stm", "").upper() if stm else "UNKNOWN",
        "ok": True,
        "execution_ms": execution_ms,
        "plan": None,
    }
    if isinstance(stm, BeginTransactionStm):
        response["transaction_id"] = result
    return response
