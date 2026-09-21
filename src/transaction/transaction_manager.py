"""Transaction coordinator: WAL, Strict 2PL and per-thread context."""

import itertools
import threading

from .lock_manager import LockManager, LockMode
from .log_manager import LogManager
from .recovery import RecoveryManager


class TransactionState:
    ACTIVE = "ACTIVE"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class TransactionManager:
    def __init__(self, log_path="data/transactions.wal", lock_manager=None):
        self.lock_manager = lock_manager or LockManager()
        self.log_manager = LogManager(log_path)
        self.recovery_manager = RecoveryManager(self.log_manager)
        self._ids = itertools.count(1)
        self._local = threading.local()
        self._mutex = threading.Lock()
        self._states = {}
        self._buffer_managers = []

    def register_buffer_manager(self, buffer_manager):
        """Register a buffer manager whose dirty pages can be flushed."""
        if buffer_manager not in self._buffer_managers:
            self._buffer_managers.append(buffer_manager)

    def flush(self, buffer_managers=()):
        """Persist the WAL before flushing registered dirty pages.

        A FLUSH marker makes the durability boundary visible in the WAL. The
        marker is durable before any data page is written, preserving WAL.
        """
        txn_id = self.current() or 0
        self.log_manager.append(txn_id, "FLUSH")
        self.log_manager.flush()

        managers = list(self._buffer_managers)
        for buffer_manager in buffer_managers:
            if buffer_manager not in managers:
                managers.append(buffer_manager)
        for buffer_manager in managers:
            buffer_manager.flush_all()

    def begin(self) -> int:
        if getattr(self._local, "txn_id", None) is not None:
            raise RuntimeError("transaction already active on this thread")
        with self._mutex:
            txn_id = next(self._ids)
            self._states[txn_id] = TransactionState.ACTIVE
        self._local.txn_id = txn_id
        self._local.undo = []
        self.log_manager.begin(txn_id)
        return txn_id

    def current(self) -> int | None:
        return getattr(self._local, "txn_id", None)

    def require(self) -> int:
        txn_id = self.current()
        if txn_id is None:
            raise RuntimeError("no active transaction on this thread")
        return txn_id

    def lock(self, resource: str, mode=LockMode.EXCLUSIVE):
        txn_id = self.require()
        self.lock_manager.acquire(txn_id, resource, mode)

    def log_update(self, resource, old_value, new_value):
        txn_id = self.require()
        return self.log_manager.update(txn_id, resource, old_value, new_value)

    def log_data_change(self, operation, table, rid, before, after):
        return self.log_manager.data_change(self.require(), operation, table, rid, before, after)

    def add_undo(self, callback):
        self.require()
        self._local.undo.append(callback)

    def commit(self):
        txn_id = self.require()
        self.log_manager.commit(txn_id)
        self.log_manager.flush()
        with self._mutex:
            self._states[txn_id] = TransactionState.COMMITTED
        self.lock_manager.release_all(txn_id)
        self._clear()

    def abort(self):
        txn_id = self.require()
        for callback in reversed(getattr(self._local, "undo", [])):
            callback()
        self.log_manager.abort(txn_id)
        with self._mutex:
            self._states[txn_id] = TransactionState.ABORTED
        self.lock_manager.release_all(txn_id)
        self._clear()

    def _clear(self):
        self._local.txn_id = None
        self._local.undo = []
