"""Threaded demonstration of race handling and deadlock detection."""

import threading
import time

from .lock_manager import DeadlockError, LockManager, LockMode
from .transaction_manager import TransactionManager


def run_demo(log_path="/tmp/multimodal-dbms-demo.wal"):
    manager = TransactionManager(log_path, LockManager())
    barrier = threading.Barrier(2)
    events = []
    events_lock = threading.Lock()

    def note(message):
        with events_lock:
            line = f"thread={threading.get_ident()} {message}"
            events.append(line)
            print(line)

    def worker(name):
        txn_id = manager.begin()
        note(f"BEGIN {name} txn={txn_id}")
        barrier.wait()
        manager.lock("account:shared", LockMode.EXCLUSIVE)
        note(f"LOCK X account:shared txn={txn_id}")
        time.sleep(0.02)
        manager.commit()
        note(f"COMMIT {name} txn={txn_id}")

    threads = [threading.Thread(target=worker, args=(f"race-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    deadlock_barrier = threading.Barrier(2)

    def deadlock_worker(first, second):
        txn_id = manager.begin()
        note(f"BEGIN deadlock txn={txn_id}")
        manager.lock(first, LockMode.EXCLUSIVE)
        deadlock_barrier.wait()
        try:
            manager.lock(second, LockMode.EXCLUSIVE)
        except DeadlockError:
            note(f"ABORT deadlock txn={txn_id}")
            manager.abort()
            return
        manager.commit()

    first = threading.Thread(target=deadlock_worker, args=("row:a", "row:b"))
    second = threading.Thread(target=deadlock_worker, args=("row:b", "row:a"))
    first.start()
    second.start()
    first.join()
    second.join()
    return events


if __name__ == "__main__":
    run_demo()
