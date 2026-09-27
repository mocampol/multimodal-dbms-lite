import threading

import pytest

from transaction.lock_manager import DeadlockError, LockManager, LockMode, LockTimeoutError
from transaction.transaction_manager import TransactionManager


def test_shared_locks_are_compatible():
    locks = LockManager()

    locks.acquire(1, "r", LockMode.SHARED)
    locks.acquire(2, "r", LockMode.SHARED, timeout=0.1)

    assert locks.held_mode(1, "r") == LockMode.SHARED
    assert locks.held_mode(2, "r") == LockMode.SHARED


def test_exclusive_lock_blocks_other_transactions():
    locks = LockManager()
    locks.acquire(1, "r", LockMode.EXCLUSIVE)

    with pytest.raises(LockTimeoutError):
        locks.acquire(2, "r", LockMode.SHARED, timeout=0.05)


def test_exclusive_lock_is_not_downgraded_by_later_shared_request():
    locks = LockManager()
    locks.acquire(1, "r", LockMode.EXCLUSIVE)

    locks.acquire(1, "r", LockMode.SHARED)

    assert locks.held_mode(1, "r") == LockMode.EXCLUSIVE
    with pytest.raises(LockTimeoutError):
        locks.acquire(2, "r", LockMode.SHARED, timeout=0.05)


def test_shared_lock_upgrades_to_exclusive_when_alone():
    locks = LockManager()
    locks.acquire(1, "r", LockMode.SHARED)

    locks.acquire(1, "r", LockMode.EXCLUSIVE, timeout=0.1)

    assert locks.held_mode(1, "r") == LockMode.EXCLUSIVE


def test_timeout_does_not_leave_stale_wait_edge():
    locks = LockManager()
    locks.acquire(1, "a", LockMode.EXCLUSIVE)
    with pytest.raises(LockTimeoutError):
        locks.acquire(2, "a", LockMode.EXCLUSIVE, timeout=0.05)

    # txn 2 is no longer waiting on "a", so txn 1 waiting on txn 2 is not a cycle
    locks.acquire(2, "b", LockMode.EXCLUSIVE)
    with pytest.raises(LockTimeoutError):
        locks.acquire(1, "b", LockMode.EXCLUSIVE, timeout=0.05)


def test_release_all_wakes_waiters():
    locks = LockManager()
    locks.acquire(1, "r", LockMode.EXCLUSIVE)
    acquired = threading.Event()

    def waiter():
        locks.acquire(2, "r", LockMode.EXCLUSIVE, timeout=2)
        acquired.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    locks.release_all(1)
    thread.join(timeout=2)

    assert acquired.is_set()
    assert locks.held_resources(1) == frozenset()


def test_deadlock_is_detected_and_one_side_proceeds():
    locks = LockManager()
    barrier = threading.Barrier(2)
    outcomes = {}

    def worker(txn_id, first, second):
        locks.acquire(txn_id, first, LockMode.EXCLUSIVE)
        barrier.wait()
        try:
            locks.acquire(txn_id, second, LockMode.EXCLUSIVE, timeout=2)
            outcomes[txn_id] = "acquired"
        except DeadlockError:
            outcomes[txn_id] = "deadlock"
        finally:
            locks.release_all(txn_id)

    threads = [
        threading.Thread(target=worker, args=(1, "a", "b")),
        threading.Thread(target=worker, args=(2, "b", "a")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(outcomes.values()) == ["acquired", "deadlock"]


def test_abort_releases_locks_even_if_undo_fails(tmp_path):
    manager = TransactionManager(str(tmp_path / "transactions.wal"))
    txn_id = manager.begin()
    manager.lock("r", LockMode.EXCLUSIVE)

    def failing_undo():
        raise RuntimeError("undo failed")

    manager.add_undo(failing_undo)
    with pytest.raises(RuntimeError):
        manager.abort()

    assert manager.current() is None
    assert manager.lock_manager.held_resources(txn_id) == frozenset()
    # no ABORT record: startup recovery is responsible for undoing it
    assert manager.log_manager.records()[-1]["kind"] != "ABORT"


def test_commit_failure_aborts_and_releases_locks(tmp_path, monkeypatch):
    manager = TransactionManager(str(tmp_path / "transactions.wal"))
    txn_id = manager.begin()
    manager.lock("r", LockMode.EXCLUSIVE)

    def failing_commit(_txn_id):
        raise OSError("disk full")

    monkeypatch.setattr(manager.log_manager, "commit", failing_commit)
    with pytest.raises(OSError):
        manager.commit()

    assert manager.current() is None
    assert manager.lock_manager.held_resources(txn_id) == frozenset()
    assert manager.log_manager.records()[-1]["kind"] == "ABORT"
