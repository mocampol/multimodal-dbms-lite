from datetime import date
from decimal import Decimal

from common.record import Record
from common.value import DataType, Value
from query.query_engine import execute
from storage.heap.rid import RID
from storage.sequential.sequential_file import SeqRID
from transaction.log_manager import LogManager


def _rows(catalog, table="accounts"):
    return sorted(tuple(v.data for v in r.values) for r in execute(f"SELECT * FROM {table};", catalog))


def _restart(open_catalog, base_dir):
    catalog = open_catalog(base_dir)
    result = catalog._transaction_manager.recovery_manager.run_startup_recovery(catalog)
    return catalog, result


def _crash_with_pages_on_disk(catalog, table="accounts"):
    """Simulates a crash after the buffer pool stole the uncommitted pages."""
    catalog.flush_table(table)


def test_wal_round_trips_rids_and_records(tmp_path):
    log = LogManager(str(tmp_path / "wal"))
    record = Record([
        Value(DataType.INTEGER, 1),
        Value(DataType.DATE, date(2024, 5, 1)),
        Value(DataType.NUMERIC, Decimal("10.50")),
        Value(DataType.BYTEA, b"\x00\xff"),
    ])
    seq_rid = SeqRID(in_overflow=True, overflow_rid=RID(2, 3))

    log.data_change(1, "UPDATE", "t", RID(4, 5), record, record, new_rid=seq_rid)
    stored = log.records()[-1]

    assert stored["rid"] == RID(4, 5)
    assert stored["new_rid"] == seq_rid
    assert [v.data for v in stored["after"].values] == [1, date(2024, 5, 1), Decimal("10.50"), b"\x00\xff"]


def test_lsn_keeps_increasing_after_reopening_the_wal(tmp_path):
    path = str(tmp_path / "wal")
    LogManager(path).begin(1)
    LogManager(path).begin(2)

    assert [r["lsn"] for r in LogManager(path).records()] == [1, 2]


def test_committed_rows_survive_a_crash_without_shutdown_flush(tmp_path, reopen):
    catalog = reopen(tmp_path)
    execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);", catalog)
    execute("INSERT INTO accounts VALUES (1, 100);", catalog)

    restarted, result = _restart(reopen, tmp_path)

    assert result["undone"] == set()
    assert _rows(restarted) == [(1, 100)]


def test_recovery_undoes_uncommitted_insert_update_and_delete(tmp_path, reopen):
    catalog = reopen(tmp_path)
    execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER, email VARCHAR(20) UNIQUE);", catalog)
    execute("CREATE INDEX idx_balance ON accounts (balance) USING BTREE;", catalog)
    execute("INSERT INTO accounts VALUES (1, 100, 'a@x'), (2, 200, 'b@x');", catalog)

    manager = catalog._transaction_manager
    loser = manager.begin()
    execute("INSERT INTO accounts VALUES (3, 300, 'c@x');", catalog)
    execute("UPDATE accounts SET balance = 999 WHERE id = 1;", catalog)
    execute("DELETE FROM accounts WHERE id = 2;", catalog)
    _crash_with_pages_on_disk(catalog)

    restarted, result = _restart(reopen, tmp_path)

    assert result["undone"] == {loser}
    assert _rows(restarted) == [(1, 100, "a@x"), (2, 200, "b@x")]
    # derived structures were rebuilt: UNIQUE sets and the balance index agree with the data
    execute("INSERT INTO accounts VALUES (3, 300, 'c@x');", restarted)
    assert [r[0].data for r in execute("SELECT * FROM accounts WHERE balance = 200;", restarted)] == [2]
    assert execute("SELECT * FROM accounts WHERE balance = 999;", restarted) == []


def test_recovery_is_not_repeated_on_the_next_restart(reopen, tmp_path):
    catalog = reopen(tmp_path)
    execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);", catalog)
    execute("INSERT INTO accounts VALUES (1, 100);", catalog)
    catalog._transaction_manager.begin()
    execute("DELETE FROM accounts WHERE id = 1;", catalog)
    _crash_with_pages_on_disk(catalog)

    _restart(reopen, tmp_path)
    restarted, result = _restart(reopen, tmp_path)

    assert result["undone"] == set()
    assert _rows(restarted) == [(1, 100)]


def test_crash_in_the_middle_of_an_abort_restores_each_row_once(tmp_path, reopen):
    catalog = reopen(tmp_path)
    execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);", catalog)
    execute("INSERT INTO accounts VALUES (1, 100);", catalog)

    manager = catalog._transaction_manager
    manager.begin()
    execute("DELETE FROM accounts WHERE id = 1;", catalog)
    # run the abort's undo (re-inserts the row, logging a compensation) but crash before ABORT
    for callback in reversed(manager._local.undo):
        callback()
    _crash_with_pages_on_disk(catalog)

    restarted, _ = _restart(reopen, tmp_path)

    assert _rows(restarted) == [(1, 100)]
