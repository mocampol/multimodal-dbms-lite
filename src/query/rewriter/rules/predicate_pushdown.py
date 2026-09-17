"""
NO-OP for now: predicate pushdown only has meaning when a condition can
be moved past another operator — typically pushing a filter below a
JOIN so it runs on smaller inputs. The current grammar has no JOIN
(a single SELECT reads from exactly one table), so there is nothing to
push a predicate through yet.

This file exists as the designated home for that logic once JOIN
support is added to the grammar/parser — at that point, this function
would inspect a join tree and relocate applicable conditions onto the
appropriate child scan before the planner builds physical nodes.
"""

from ast_nodes import Stm


def push_down_predicates(stm: Stm) -> Stm:
    return stm
