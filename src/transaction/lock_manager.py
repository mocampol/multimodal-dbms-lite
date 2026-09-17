"""Strict two-phase lock manager with S/X locks and deadlock detection."""

from collections import defaultdict
from enum import Enum
import threading
import time


class LockMode(Enum):
    SHARED = "S"
    EXCLUSIVE = "X"


class DeadlockError(RuntimeError):
    pass


class LockTimeoutError(TimeoutError):
    pass


class LockManager:
    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._holders = defaultdict(dict)
        self._waiting = defaultdict(set)
        self._held_by_txn = defaultdict(set)

    def acquire(self, txn_id: int, resource: str, mode: LockMode, timeout=None):
        with self._condition:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                blockers = self._blockers(txn_id, resource, mode)
                if not blockers:
                    self._holders[resource][txn_id] = mode
                    self._held_by_txn[txn_id].add(resource)
                    self._waiting[txn_id].discard(resource)
                    return
                self._waiting[txn_id].add(resource)
                if self._has_cycle(txn_id, set()):
                    self._waiting[txn_id].discard(resource)
                    raise DeadlockError(f"deadlock detected for transaction {txn_id}")
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LockTimeoutError(resource)
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def release_all(self, txn_id: int):
        with self._condition:
            for resource in list(self._held_by_txn.pop(txn_id, set())):
                self._holders[resource].pop(txn_id, None)
                if not self._holders[resource]:
                    del self._holders[resource]
            self._waiting.pop(txn_id, None)
            self._condition.notify_all()

    def held_resources(self, txn_id: int):
        with self._condition:
            return frozenset(self._held_by_txn.get(txn_id, set()))

    def _blockers(self, txn_id, resource, mode):
        holders = self._holders.get(resource, {})
        return {
            other for other, held_mode in holders.items()
            if other != txn_id and (
                mode == LockMode.EXCLUSIVE or held_mode == LockMode.EXCLUSIVE
            )
        }

    def _has_cycle(self, start, visited):
        if start in visited:
            return True
        visited.add(start)
        for resource in self._waiting.get(start, set()):
            for blocker in self._blockers(start, resource, LockMode.EXCLUSIVE):
                if self._has_cycle(blocker, visited.copy()):
                    return True
        return False
