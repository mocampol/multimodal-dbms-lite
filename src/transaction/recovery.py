"""WAL recovery: REDO committed work, UNDO incomplete work, on startup."""


class RecoveryManager:
    def __init__(self, log_manager):
        self.log_manager = log_manager

    def recover(self, redo, undo):
        records = self.log_manager.records()
        committed = {record["txn_id"] for record in records if record["kind"] == "COMMIT"}
        aborted = {record["txn_id"] for record in records if record["kind"] == "ABORT"}
        updates = [record for record in records if record["kind"] in {"UPDATE", "DATA"}]

        for record in updates:
            if record["txn_id"] in committed:
                redo(record)
        for record in reversed(updates):
            if record["txn_id"] not in committed and record["txn_id"] not in aborted:
                undo(record)

        return {
            "committed": committed,
            "undone": {
                record["txn_id"] for record in updates
                if record["txn_id"] not in committed and record["txn_id"] not in aborted
            },
        }

    def run_startup_recovery(self, catalog):
        redo = make_redo(catalog)
        undo = make_undo(catalog)
        result = self.recover(redo, undo)
        self._flush_all(catalog)
        return result

    def _flush_all(self, catalog):
        for table_name in list(catalog.tables):
            storage = catalog.get_storage(table_name)
            storage.bm.flush_all()
            if hasattr(storage, "overflow"):
                storage.overflow.bm.flush_all()


def make_redo(catalog):
    def redo(record):
        if record["kind"] != "DATA":
            return
        storage = catalog.get_storage(record["table"])
        operation = record["operation"]
        rid = record["rid"]

        if operation in ("INSERT", "UPDATE"):
            storage.update(rid, record["after"])
        elif operation == "DELETE":
            storage.delete(rid)
    return redo


def make_undo(catalog):
    def undo(record):
        if record["kind"] != "DATA":
            return
        storage = catalog.get_storage(record["table"])
        operation = record["operation"]
        rid = record["rid"]

        if operation == "INSERT":
            storage.delete(rid)
        elif operation in ("UPDATE", "DELETE"):
            storage.update(rid, record["before"])
    return undo