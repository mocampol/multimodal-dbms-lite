"""WAL recovery helpers for REDO committed and UNDO incomplete work."""


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
        return {"committed": committed, "undone": {
            record["txn_id"] for record in updates
            if record["txn_id"] not in committed and record["txn_id"] not in aborted
        }}
