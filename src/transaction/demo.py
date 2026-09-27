"""
Threaded demonstration of race conditions and how the engine resolves them.

Every scenario runs real SQL through query_engine.execute() against a
throwaway database, with one thread per transaction:

  1. Lost update: two deposits do read-modify-write on the same account.
     In autocommit mode each statement releases its locks, both threads read
     the same balance and one deposit is lost. Inside BEGIN/END TRANSACTION,
     Strict 2PL keeps the S lock from the read; both writers need to upgrade
     to X, the deadlock detector aborts one, it retries, and no deposit is lost.
  2. Dirty read: a reader blocks on a writer's X lock and, when the writer
     aborts, sees the original value instead of the uncommitted one.
  3. Deadlock: two transfers lock two accounts in opposite order; one is
     chosen as victim and rolled back, the other commits.

Run from src/:  python -m transaction.demo
"""

import tempfile
import threading
import time

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from query.query_engine import execute
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.heap_file import HeapFile

from .lock_manager import DeadlockError
from .transaction_manager import TransactionManager


class Timeline:
    """Thread-safe event log printed as `+elapsed  actor  message`."""

    def __init__(self, echo=True):
        self.events = []
        self.echo = echo
        self._start = time.perf_counter()
        self._lock = threading.Lock()

    def __call__(self, actor, message):
        with self._lock:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self.events.append((actor, message))
            if self.echo:
                print(f"  +{elapsed_ms:6.1f} ms  {actor:<3} {message}")


def _open_database(base_dir):
    def heap_factory(schema):
        buffer_manager = BufferManager(FileManager(f"{base_dir}/{schema.table_name}.tbl"), pool_size=16)
        return HeapFile(schema, buffer_manager)

    catalog = Catalog(heap_factory=heap_factory, storage_factories={StorageType.HEAP: heap_factory})
    catalog._transaction_manager = TransactionManager(f"{base_dir}/transactions.wal")
    return catalog


def _balance(catalog, account_id):
    rows = execute(f"SELECT balance FROM accounts WHERE id = {account_id};", catalog)
    return rows[0][0].data


def _run_threads(*targets):
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def lost_update(catalog, log, use_transactions):
    """Two concurrent deposits of 50 and 30 on an account holding 100."""
    execute("UPDATE accounts SET balance = 100 WHERE id = 1;", catalog)
    both_read = threading.Barrier(2)

    def deposit(actor, amount):
        attempt = 0
        while True:
            attempt += 1
            try:
                if use_transactions:
                    execute("BEGIN TRANSACTION;", catalog)
                balance = _balance(catalog, 1)
                log(actor, f"lee saldo = {balance}")
                if attempt == 1:
                    # force the worst interleaving: both read before anyone writes
                    both_read.wait(timeout=5)
                execute(f"UPDATE accounts SET balance = {balance + amount} WHERE id = 1;", catalog)
                log(actor, f"escribe saldo = {balance + amount}")
                if use_transactions:
                    execute("END TRANSACTION;", catalog)
                    log(actor, "COMMIT")
                return
            except DeadlockError:
                log(actor, "DEADLOCK: elegido como víctima, ABORT y reintento")

    _run_threads(lambda: deposit("T1", 50), lambda: deposit("T2", 30))
    return _balance(catalog, 1)


def dirty_read(catalog, log):
    """A reader must never see a value whose writer later aborts."""
    execute("UPDATE accounts SET balance = 100 WHERE id = 1;", catalog)
    manager = catalog._transaction_manager
    written = threading.Event()
    seen = {}

    def writer():
        execute("BEGIN TRANSACTION;", catalog)
        execute("UPDATE accounts SET balance = 0 WHERE id = 1;", catalog)
        log("T1", "UPDATE saldo = 0 (sin commit, tiene lock X)")
        written.set()
        time.sleep(0.2)
        log("T1", "ABORT")
        manager.abort()

    def reader():
        written.wait(timeout=5)
        log("T2", "SELECT saldo ... (espera el lock S)")
        seen["balance"] = _balance(catalog, 1)
        log("T2", f"lee saldo = {seen['balance']}")

    _run_threads(writer, reader)
    return seen["balance"]


def deadlock(catalog, log):
    """Two transfers locking accounts 1 and 2 in opposite order."""
    execute("UPDATE accounts SET balance = 100 WHERE id = 1;", catalog)
    execute("UPDATE accounts SET balance = 100 WHERE id = 2;", catalog)
    both_locked = threading.Barrier(2)
    outcome = {}

    def transfer(actor, source, target):
        execute("BEGIN TRANSACTION;", catalog)
        execute(f"UPDATE accounts SET balance = 90 WHERE id = {source};", catalog)
        log(actor, f"lock X sobre cuenta {source}")
        both_locked.wait(timeout=5)
        try:
            log(actor, f"pide lock X sobre cuenta {target}")
            execute(f"UPDATE accounts SET balance = 110 WHERE id = {target};", catalog)
            execute("END TRANSACTION;", catalog)
            log(actor, "COMMIT")
            outcome[actor] = "commit"
        except DeadlockError:
            log(actor, "DEADLOCK detectado: víctima, ABORT y rollback")
            outcome[actor] = "abort"

    _run_threads(lambda: transfer("T1", 1, 2), lambda: transfer("T2", 2, 1))
    return outcome, (_balance(catalog, 1), _balance(catalog, 2))


def run_demo(echo=True):
    say = print if echo else (lambda *args: None)
    results = {}
    with tempfile.TemporaryDirectory() as base_dir:
        catalog = _open_database(base_dir)
        execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);", catalog)
        execute("INSERT INTO accounts VALUES (1, 100), (2, 100);", catalog)

        say("\n=== 1a. Lost update SIN transacciones (autocommit) ===")
        results["lost_update_autocommit"] = lost_update(catalog, Timeline(echo), use_transactions=False)
        say(f"  Saldo final: {results['lost_update_autocommit']} (esperado 180) -> se perdió un depósito")

        say("\n=== 1b. Lost update CON BEGIN/END TRANSACTION (Strict 2PL) ===")
        results["lost_update_transactions"] = lost_update(catalog, Timeline(echo), use_transactions=True)
        say(f"  Saldo final: {results['lost_update_transactions']} (esperado 180) -> ningún depósito perdido")

        say("\n=== 2. Dirty read: lector bloqueado por un escritor que aborta ===")
        results["dirty_read"] = dirty_read(catalog, Timeline(echo))
        say(f"  T2 leyó {results['dirty_read']} (el valor commiteado, nunca el 0 abortado)")

        say("\n=== 3. Deadlock entre dos transferencias ===")
        outcome, balances = deadlock(catalog, Timeline(echo))
        results["deadlock"] = outcome
        results["deadlock_balances"] = balances
        say(f"  Resultado: {outcome}; saldos finales cuentas 1 y 2: {balances}")
    return results


if __name__ == "__main__":
    run_demo()
