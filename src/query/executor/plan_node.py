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
