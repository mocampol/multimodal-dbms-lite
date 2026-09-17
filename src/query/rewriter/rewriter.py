"""Applies validated-AST rewrites before physical planning."""

from query.parser.ast_nodes import Stm
from query.rewriter.rules.simplify_expressions import simplify
from query.rewriter.rules.predicate_pushdown import push_down_predicates


def rewrite(stm: Stm) -> Stm:
    """Apply all enabled local rewrite rules in a stable order."""
    return push_down_predicates(simplify(stm))
