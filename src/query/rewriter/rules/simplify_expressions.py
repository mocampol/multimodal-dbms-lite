from query.parser.ast_nodes import Stm, SelectStm, DeleteStm, BinaryExp, IdExp, BinaryOp


def simplify(stm: Stm) -> Stm:
    """
    Currently handles one real case: a WHERE comparing a column against
    itself is always true for EQ_OP/LEQ_OP/GEQ_OP and always false for
    LE_OP/GT_OP — so the condition can be dropped (always-true) or the
    statement short-circuited (always-false).

    Only SelectStm and DeleteStm carry a where_cond in this grammar.
    """
    if isinstance(stm, (SelectStm, DeleteStm)) and stm.where_cond is not None:
        cond = stm.where_cond
        if (
            isinstance(cond, BinaryExp)
            and isinstance(cond.left, IdExp)
            and isinstance(cond.right, IdExp)
            and cond.left.value == cond.right.value
        ):
            if cond.op in (BinaryOp.EQ_OP, BinaryOp.LEQ_OP, BinaryOp.GEQ_OP):
                stm.where_cond = None  # always true: drop the filter entirely
            # LE_OP / GT_OP (always false) intentionally left as-is:
            # short-circuiting to "no rows" needs a dedicated AST node
            # or planner-level flag this grammar doesn't have yet.

    return stm
