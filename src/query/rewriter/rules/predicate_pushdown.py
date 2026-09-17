"""Predicate rewrite hook for the supported SQL grammar."""

from query.parser.ast_nodes import Stm


def push_down_predicates(stm: Stm) -> Stm:
    """Preserve the validated AST until compound predicates are available.

    Equality access selection is already handled by the physical planner.
    The current grammar has one binary predicate, so moving predicates
    between join children would not change the represented query.
    """
    return stm
