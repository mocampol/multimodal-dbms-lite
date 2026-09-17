"""
Applies rewrite rules to a validated AST before it reaches the planner.

HONEST SCOPE NOTE: with the current grammar (no AND/OR, no subqueries,
exactly one optional WHERE per SELECT), there is very little a rewriter
can meaningfully do yet — there's nothing to push a predicate THROUGH
(no JOIN to push it past), and no compound conditions to simplify. The
rules below are intentionally minimal placeholders that do real, if
small, work today, and are structured so JOIN support can slot into
predicate_pushdown.py later without restructuring this file.
"""

from query.parser.ast_nodes import Stm

from query.rewriter.rules.simplify_expressions import simplify
from query.rewriter.rules.predicate_pushdown import push_down_predicates


def rewrite(stm: Stm) -> Stm:
    stm = simplify(stm)
    stm = push_down_predicates(stm)
    return stm
