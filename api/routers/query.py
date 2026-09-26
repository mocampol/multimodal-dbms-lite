import time

from fastapi import APIRouter

from engine import catalog
from explain import describe_plan, output_columns, serialize_explain_result
from schemas import QueryRequest
from serialize import serialize_record

from query.parser.ast_nodes import SelectStm, InsertStm, UpdateStm, DeleteStm, BeginTransactionStm, ExplainStm
from query.rewriter.rewriter import rewrite
from query.planner.plan_builder import build_select_plan
from query.query_engine import execute_many

router = APIRouter()


@router.post("/query")
def run_query(payload: QueryRequest):
    t0 = time.perf_counter()
    executed = execute_many(payload.sql, catalog)
    execution_ms = (time.perf_counter() - t0) * 1000

    responses = [_response_for_statement(stm, result, execution_ms / len(executed)) for stm, result in executed]
    if len(responses) == 1:
        return responses[0]
    return {
        "type": "BATCH",
        "statements": responses,
        "execution_ms": execution_ms,
    }


def _response_for_statement(stm, result, execution_ms):
    # Verificado contra el query_engine.py real: execute_many() llama a
    # _execute_statement() por cada sentencia, y ese es el mismo punto
    # donde despacha ExplainStm (construye/instrumenta el plan) — así
    # que acá `result` sí es un ExplainResult cuando stm es ExplainStm.
    if isinstance(stm, ExplainStm):
        return {
            **serialize_explain_result(result),
            "execution_ms": execution_ms,
        }

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