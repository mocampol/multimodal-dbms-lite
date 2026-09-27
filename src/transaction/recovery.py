"""
WAL crash recovery for undo logging with FORCE.

The protocol (see transaction_manager.py) guarantees that a transaction
with a COMMIT or ABORT record has all its pages on disk, so there is
nothing to REDO. Recovery only undoes "losers": transactions with BEGIN
but neither COMMIT nor ABORT. Each undo step checks the current row first,
so running recovery again after a crash mid-recovery is harmless.
"""

from storage.heap.rid import RID


class RecoveryManager:
    def __init__(self, log_manager):
        self.log_manager = log_manager

    def recover(self, undo):
        records = self.log_manager.records()
        committed = {r["txn_id"] for r in records if r["kind"] == "COMMIT"}
        finished = committed | {r["txn_id"] for r in records if r["kind"] == "ABORT"}
        losers = {r["txn_id"] for r in records if r["kind"] == "BEGIN"} - finished
        loser_changes = [
            r for r in records if r["kind"] == "DATA" and r["txn_id"] in losers
        ]

        for record in reversed(loser_changes):
            undo(record)

        return {
            "committed": committed,
            "undone": losers,
            "tables": {r["table"] for r in loser_changes},
        }

    def run_startup_recovery(self, catalog):
        result = self.recover(make_undo(catalog))
        for table_name in sorted(result["tables"]):
            if table_name not in catalog.tables:
                continue
            rebuild = getattr(catalog, "rebuild_derived", None)
            if rebuild is not None:
                rebuild(table_name)
            _flush_table(catalog, table_name)
        # the undo is durable now: mark losers as aborted so they are not undone again
        for txn_id in sorted(result["undone"]):
            self.log_manager.abort(txn_id)
        return result


def make_undo(catalog):
    # a row recovery re-inserts gets a new RID; earlier log records still
    # name the old one, so they are redirected through this map
    moved = {}

    def undo(record):
        table_name = record["table"]
        logged_rid = record["rid"]
        # records written before the WAL stored real RIDs cannot be undone
        if table_name not in catalog.tables or not isinstance(logged_rid, RID):
            return
        storage = catalog.get_storage(table_name)
        rid = moved.get(logged_rid, logged_rid)
        operation = record["operation"]

        if operation == "INSERT":
            if _same(_read(storage, rid), record["after"]):
                storage.delete(rid)
        elif operation == "DELETE":
            if not _same(_read(storage, rid), record["before"]):
                moved[logged_rid] = storage.insert(record["before"])
        elif operation == "UPDATE":
            new_rid = record.get("new_rid")
            new_rid = moved.get(new_rid, new_rid) if isinstance(new_rid, RID) else rid
            reinserted = _undo_update(storage, rid, new_rid, record["before"])
            if reinserted is not None:
                moved[logged_rid] = reinserted
                if isinstance(record.get("new_rid"), RID):
                    moved[record["new_rid"]] = reinserted
    return undo


def _undo_update(storage, rid, new_rid, before):
    """Restores `before`; returns the new RID if the row had to be re-inserted."""
    target, current = new_rid, _read(storage, new_rid)
    if current is None and new_rid != rid:
        target, current = rid, _read(storage, rid)
    if current is None:
        return storage.insert(before)
    if not _same(current, before):
        storage.update(target, before)
    return None


def _read(storage, rid):
    """storage.get(rid), treating a page that never reached disk as a missing row."""
    try:
        return storage.get(rid)
    except ValueError:
        return None


def _same(record, other):
    if record is None or other is None:
        return record is other
    return [v.data for v in record.values] == [v.data for v in other.values]


def _flush_table(catalog, table_name):
    flush_table = getattr(catalog, "flush_table", None)
    if flush_table is not None:
        flush_table(table_name)
        return
    storage = catalog.get_storage(table_name)
    storage.bm.flush_all()
    if hasattr(storage, "overflow"):
        storage.overflow.bm.flush_all()
