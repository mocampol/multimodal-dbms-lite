import json

from transaction.transaction_manager import TransactionManager


class InspectingBufferManager:
    def __init__(self, wal_path):
        self.wal_path = wal_path
        self.saw_flush = False

    def flush_all(self):
        with open(self.wal_path, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        self.saw_flush = any(record["kind"] == "FLUSH" for record in records)


def test_flush_persists_wal_before_dirty_pages(tmp_path):
    wal_path = tmp_path / "transactions.wal"
    manager = TransactionManager(str(wal_path))
    buffer_manager = InspectingBufferManager(wal_path)

    manager.begin()
    manager.flush([buffer_manager])

    assert buffer_manager.saw_flush
    assert manager.log_manager.records()[-1]["kind"] == "FLUSH"


def test_commit_persists_commit_record(tmp_path):
    wal_path = tmp_path / "transactions.wal"
    manager = TransactionManager(str(wal_path))

    manager.begin()
    manager.commit()

    assert manager.log_manager.records()[-1]["kind"] == "COMMIT"