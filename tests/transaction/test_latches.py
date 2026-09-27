import threading

from query.query_engine import execute
from storage.latch import latched_scan


class RecordingLatch:
    def __init__(self):
        self.held = False
        self.entries = 0

    def __enter__(self):
        self.held = True
        self.entries += 1

    def __exit__(self, *exc):
        self.held = False


def test_latched_scan_releases_latch_between_items_and_closes_source():
    latch = RecordingLatch()
    closed = []

    def rows():
        try:
            yield 1
            yield 2
        finally:
            closed.append(latch.held)

    scan = latched_scan(rows(), latch)
    assert next(scan) == 1
    assert latch.held is False
    scan.close()

    assert closed == [True], "the source must be closed under the latch"


def test_waiting_for_row_lock_does_not_block_the_table(catalog):
    manager = catalog._transaction_manager
    execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);", catalog)
    execute("INSERT INTO accounts VALUES (1, 100);", catalog)
    updated = threading.Event()
    release = threading.Event()

    def holder():
        manager.begin()
        execute("UPDATE accounts SET balance = 50 WHERE id = 1;", catalog)
        updated.set()
        release.wait(timeout=5)
        manager.commit()

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    updated.wait(timeout=5)

    blocked = threading.Thread(target=lambda: execute("UPDATE accounts SET balance = 0 WHERE id = 1;", catalog))
    blocked.start()
    blocked.join(timeout=0.2)
    assert blocked.is_alive()

    inserter = threading.Thread(target=lambda: execute("INSERT INTO accounts VALUES (2, 10);", catalog))
    inserter.start()
    inserter.join(timeout=2)
    assert not inserter.is_alive(), "a thread waiting on a row lock must not hold the table latch"

    release.set()
    holder_thread.join(timeout=5)
    blocked.join(timeout=5)
    rows = sorted((r[0].data, r[1].data) for r in execute("SELECT * FROM accounts;", catalog))
    assert rows == [(1, 0), (2, 10)]


def test_concurrent_inserts_keep_every_row(catalog):
    execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name VARCHAR(20));", catalog)
    threads_count, per_thread = 8, 25
    barrier = threading.Barrier(threads_count)
    errors = []

    def worker(offset):
        barrier.wait(timeout=5)
        try:
            for i in range(per_thread):
                item_id = offset * per_thread + i
                execute(f"INSERT INTO items VALUES ({item_id}, 'item-{item_id}');", catalog)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    ids = sorted(r[0].data for r in execute("SELECT * FROM items;", catalog))
    assert ids == list(range(threads_count * per_thread))


def test_concurrent_unique_inserts_admit_only_one(catalog):
    execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(20) UNIQUE);", catalog)
    threads_count = 8
    barrier = threading.Barrier(threads_count)
    results = []

    def worker(user_id):
        barrier.wait(timeout=5)
        try:
            execute(f"INSERT INTO users VALUES ({user_id}, 'same@x');", catalog)
            results.append("ok")
        except ValueError:
            results.append("duplicate")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert results.count("ok") == 1
    assert len(execute("SELECT * FROM users;", catalog)) == 1
