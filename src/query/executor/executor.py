"""
Driver for a physical plan tree: opens the root, pulls records via
next() until exhausted, then closes it. Doesn't need to know what kinds
of nodes exist below the root. Only that they all implement PlanNode.
"""

from typing import Iterator

from common.record import Record
from query.executor.plan_node import PlanNode


def run_plan(root: PlanNode) -> Iterator[Record]:
    """
    Executes root end to end, yielding each output Record in order.
    """
    root.open()
    try:
        while True:
            record = root.next()
            if record is None:
                return
            yield record
    finally:
        root.close()
