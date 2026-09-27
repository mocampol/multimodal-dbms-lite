"""
Physical latches: short-term mutual exclusion over a table's pages.

Row locks (transaction.LockManager) isolate transactions from each other
and are held until COMMIT/ABORT. Latches protect the physical structures
themselves (pages, buffer pool, indexes) while a single storage operation
runs, and are released as soon as it finishes.

Rule: never wait for a row lock while holding a latch. The lock holder may
need that same latch to finish or undo its work, which would deadlock
outside the lock manager's wait-for graph where it can't be detected.
"""

from contextlib import nullcontext

NO_LATCH = nullcontext()


def table_latch(catalog, table_name):
    """The catalog's latch for table_name, or a no-op for catalogs without latches."""
    get_latch = getattr(catalog, "table_latch", None)
    return get_latch(table_name) if get_latch is not None else NO_LATCH


def latched_scan(rows, latch):
    """
    Steps the `rows` iterator one item at a time under `latch`, releasing
    it between items so the consumer can block on row locks.
    """
    try:
        while True:
            with latch:
                item = next(rows, None)
            if item is None:
                return
            yield item
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            with latch:
                close()
