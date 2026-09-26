"""
Volcano/iterator interface implemented by every node in a physical
execution plan. A plan is a tree of PlanNodes; the root's next() calls
cascade down to leaf access nodes (SeqScan, IndexScan) and results flow
back up through processing/join/aggregate nodes.
"""

from abc import ABC, abstractmethod
from typing import Optional

from common.record import Record


class PlanNode(ABC):
    """
    Base class for every physical plan node.

    Nodes receive their children in __init__ and never construct them
    on their own — building the tree is exclusively the planner's job.
    """

    @abstractmethod
    def open(self) -> None:
        """Initializes this node's internal state."""
        ...

    @abstractmethod
    def next(self) -> Optional[Record]:
        """Produces the next output Record, or None when exhausted."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Releases any resources held by this node and its children."""
        ...

    def children(self) -> list["PlanNode"]:
        """Direct child plan nodes, in a stable order (e.g. left/right for
        joins). Default: no children — this is a leaf/access node."""
        return []

    def replace_children(self, new_children: list["PlanNode"]) -> None:
        """Replaces this node's children in place, in the same order
        children() returns them. Used by EXPLAIN ANALYZE to splice in
        instrumented wrappers around each child without touching this
        node's own execution logic. Default: no-op for a leaf node —
        raises if called with a non-empty list, since that would mean
        children() and replace_children() are out of sync."""
        if new_children:
            raise NotImplementedError(
                f"{type(self).__name__} no declara children() pero se "
                "le pidió replace_children() con hijos"
            )

    def describe_self(self) -> dict:
        """One-line, node-specific description for EXPLAIN — this node's
        own parameters (table, condition, columns...), without recursing
        into children. Default: just the class name."""
        return {"node": type(self).__name__}