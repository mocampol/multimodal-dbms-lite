"""
Demonstration of crash recovery: a transaction commits and its change
survives; a second transaction is left mid-way (simulating a crash
before COMMIT/ABORT) and RecoveryManager undoes it on the next startup.

Mirrors demo.py's style for the concurrency requirement — this is the
equivalent mandatory demo for Recovery: show the WAL, the "crash", and
the engine automatically fixing the data on the next boot.
"""

import shutil
from pathlib import Path

from common.schema import Schema, Column
from common.value import DataType, Value
from common.record import Record
from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile

from transaction.log_manager import LogManager
from transaction.transaction_manager import TransactionManager
from transaction.recovery import RecoveryManager


DEMO_DIR = Path("data/demo_recovery")
WAL_PATH = DEMO_DIR / "transactions.wal"
TABLE_PATH = DEMO_DIR / "cuentas.tbl"


def _fresh_demo_dir():
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)


def _open_heap_file() -> HeapFile:
    schema = Schema("cuentas", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("saldo", DataType.INTEGER),
    ])
    file_manager = FileManager(str(TABLE_PATH))
    buffer_manager = BufferManager(file_manager, pool_size=8)
    return HeapFile(schema, buffer_manager)


def _record(id_value: int, saldo_value: int) -> Record:
    return Record([Value(DataType.INTEGER, id_value), Value(DataType.INTEGER, saldo_value)])


def phase_1_simulate_crash():
    print("=== Fase 1: simulando actividad antes de un crash ===")
    heap = _open_heap_file()
    txn_manager = TransactionManager(str(WAL_PATH))

    
    txn_a = txn_manager.begin()
    rid_a = heap.insert(_record(1, 100))
    txn_manager.log_data_change("INSERT", "cuentas", rid_a, before=None, after=_record(1, 100))
    print(f"txn={txn_a} INSERT id=1 saldo=100 rid={rid_a}")
    txn_manager.commit()
    print(f"txn={txn_a} COMMIT")

    
    txn_b = txn_manager.begin()
    rid_b = heap.insert(_record(2, 999))
    txn_manager.log_data_change("INSERT", "cuentas", rid_b, before=None, after=_record(2, 999))
    print(f"txn={txn_b} INSERT id=2 saldo=999 rid={rid_b}  <-- proceso 'crashea' aquí, sin COMMIT")

    heap.bm.flush_all()  # persiste las páginas, igual que haría un flush periódico real
    print("(fin de la Fase 1 — el proceso se interrumpe sin cerrar la transacción B)")


def phase_2_recover_and_verify():
    """
    Session 2: simulates restarting the engine. Runs RecoveryManager
    against the same WAL file, then verifies id=1 is present (REDO/
    already committed) and id=2 is gone (UNDO applied).
    """
    print("\n=== Fase 2: reinicio del motor, corriendo recovery automático ===")
    heap = _open_heap_file()

    class _FakeCatalog:
        """Minimal stand-in so RecoveryManager can resolve table_name -> storage
        without needing a full Catalog for this isolated demo."""
        tables = {"cuentas": None}

        def get_storage(self, table_name):
            return heap

    log_manager = LogManager(str(WAL_PATH))
    recovery_manager = RecoveryManager(log_manager)
    result = recovery_manager.run_startup_recovery(_FakeCatalog())

    print(f"Transacciones commiteadas (REDO aplicado): {result['committed']}")
    print(f"Transacciones deshechas (UNDO aplicado):    {result['undone']}")

    records = list(heap.scan())
    ids_presentes = sorted(r[0].data for r in records)
    print(f"Registros presentes tras recovery: {ids_presentes}")

    assert 1 in ids_presentes, "FALLO: el registro commiteado (id=1) debería sobrevivir"
    assert 2 not in ids_presentes, "FALLO: el registro no commiteado (id=2) debería haberse deshecho"
    print("OK: recovery automático dejó los datos consistentes.")


def run_demo():
    _fresh_demo_dir()
    phase_1_simulate_crash()
    phase_2_recover_and_verify()


if __name__ == "__main__":
    run_demo()