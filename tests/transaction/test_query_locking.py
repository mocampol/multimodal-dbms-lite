import threading

import pytest

from query.query_engine import execute
from storage.heap.rid import RID
from storage.sequential.sequential_file import SeqRID
from transaction.lock_manager import DeadlockError, rid_resource


def _in_thread(target):
    outcome = {}

    def run():
        try:
            outcome["result"] = target()
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def test_select_waits_for_writer_and_never_sees_aborted_value(catalog):
    manager = catalog._transaction_manager
    execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);", catalog)
    execute("INSERT INTO accounts VALUES (1, 100);", catalog)
    updated = threading.Event()
    release = threading.Event()

    def writer():
        manager.begin()
        execute("UPDATE accounts SET balance = 50 WHERE id = 1;", catalog)
        updated.set()
        release.wait(timeout=5)
        manager.abort()

    writer_thread, _ = _in_thread(writer)
    updated.wait(timeout=5)
    reader_thread, read = _in_thread(lambda: execute("SELECT * FROM accounts;", catalog))

    reader_thread.join(timeout=0.3)
    assert reader_thread.is_alive(), "SELECT must block on the writer's X lock"

    release.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)
    assert "error" not in read
    assert [record[1].data for record in read["result"]] == [100]


def test_join_locks_rows_of_each_table_under_its_own_name(catalog):
    manager = catalog._transaction_manager
    execute("CREATE TABLE a (id INTEGER PRIMARY KEY, b_id INTEGER);", catalog)
    execute("CREATE TABLE b (id INTEGER PRIMARY KEY, name VARCHAR(10));", catalog)
    execute("INSERT INTO a VALUES (1, 1);", catalog)
    execute("INSERT INTO b VALUES (1, 'x');", catalog)

    txn_id = manager.begin()
    execute("SELECT * FROM a JOIN b ON a.b_id = b.id;", catalog)
    held = manager.lock_manager.held_resources(txn_id)
    manager.commit()

    assert any(resource.startswith("rid:a:") for resource in held)
    assert any(resource.startswith("rid:b:") for resource in held)


def test_seq_rid_and_plain_rid_map_to_same_lock():
    assert rid_resource("t", SeqRID(in_overflow=False, page_id=3, slot=7)) == rid_resource("t", RID(3, 7))


def test_deadlock_during_explicit_select_aborts_and_releases_locks(catalog):
    manager = catalog._transaction_manager
    execute("CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER);", catalog)
    execute("CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER);", catalog)
    execute("INSERT INTO a VALUES (1, 0);", catalog)
    execute("INSERT INTO b VALUES (1, 0);", catalog)
    barrier = threading.Barrier(2)

    def worker(mine, other):
        def run():
            txn_id = manager.begin()
            execute(f"UPDATE {mine} SET v = 1 WHERE id = 1;", catalog)
            barrier.wait(timeout=5)
            try:
                execute(f"SELECT * FROM {other};", catalog)
            except DeadlockError:
                return txn_id, "deadlock", manager.current()
            manager.commit()
            return txn_id, "committed", manager.current()
        return run

    first, first_outcome = _in_thread(worker("a", "b"))
    second, second_outcome = _in_thread(worker("b", "a"))
    first.join(timeout=5)
    second.join(timeout=5)

    outcomes = [first_outcome["result"], second_outcome["result"]]
    assert sorted(status for _, status, _ in outcomes) == ["committed", "deadlock"]
    for txn_id, _, current in outcomes:
        assert current is None
        assert manager.lock_manager.held_resources(txn_id) == frozenset()


def test_failed_multi_row_insert_undoes_earlier_rows(catalog):
    execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(20) UNIQUE);", catalog)

    with pytest.raises(ValueError):
        execute("INSERT INTO users VALUES (1, 'a@x'), (2, 'a@x');", catalog)

    assert execute("SELECT * FROM users;", catalog) == []
