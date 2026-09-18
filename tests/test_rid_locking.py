import pytest
import threading
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from unittest.mock import patch
from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile
from storage.sequential.sequential_file import SequentialFile
from query.parser.parser import Parser
from query.parser.visitor import SemanticVisitor
from query.query_engine import execute
from transaction.transaction_manager import TransactionManager
from transaction.lock_manager import LockTimeoutError

def make_sequential_factory(base_dir: str, pool_size: int = 64):
    def factory(schema):
        file_manager = FileManager(f"{base_dir}/{schema.table_name}.tbl")
        buffer_manager = BufferManager(file_manager, pool_size=pool_size)
        overflow_file_manager = FileManager(f"{base_dir}/{schema.table_name}.overflow.tbl")
        overflow_buffer_manager = BufferManager(overflow_file_manager, pool_size=pool_size)
        return SequentialFile(
            schema,
            key_column=schema.primary_key().name,
            buffer_manager=buffer_manager,
            overflow_buffer_manager=overflow_buffer_manager,
        )
    return factory

def make_heap_factory(base_dir: str, pool_size: int = 64):
    def factory(schema):
        file_manager = FileManager(f"{base_dir}/{schema.table_name}.tbl")
        buffer_manager = BufferManager(file_manager, pool_size=pool_size)
        return HeapFile(schema, buffer_manager)
    return factory

def make_index_buffer_factory(base_dir: str, pool_size: int = 64):
    def factory(table_name, column_name, index_id):
        file_manager = FileManager(f"{base_dir}/{table_name}.{column_name}.{index_id}.idx")
        return BufferManager(file_manager, pool_size=pool_size)
    return factory

from query.parser.scanner import Scanner

def test_rid_locking_sequential_file(tmp_path):
    base_dir = str(tmp_path)
    heap_factory = make_heap_factory(base_dir)
    catalog = Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(base_dir),
        },
        index_buffer_factory=make_index_buffer_factory(base_dir),
    )
    tx_manager = TransactionManager(log_path=f"{base_dir}/transactions.wal")
    catalog._transaction_manager = tx_manager

    def run_sql(sql):
        return execute(sql, catalog)

    # Usamos SequentialFile definiendo un PRIMARY KEY y USING SEQUENTIAL
    run_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(50)) USING SEQUENTIAL;")
    run_sql("INSERT INTO users VALUES (1, 'Alice');")
    run_sql("INSERT INTO users VALUES (2, 'Bob');")
    
    t1_done = threading.Event()
    t1_can_commit = threading.Event()
    t1_exception = None
    t1_txn_id = None
    
    def run_t1():
        nonlocal t1_exception, t1_txn_id
        try:
            t1_txn_id = tx_manager.begin()
            run_sql("UPDATE users SET name = 'Alice_Mod' WHERE id = 1;")
            t1_done.set()
            t1_can_commit.wait()
            tx_manager.commit()
        except Exception as e:
            t1_exception = e
            t1_done.set()
            
    thread1 = threading.Thread(target=run_t1)
    thread1.start()
    
    t1_done.wait()
    assert t1_exception is None, f"T1 falló con: {t1_exception}"
    
    held = tx_manager.lock_manager.held_resources(t1_txn_id)
    rid_lock_t1 = [r for r in held if r.startswith("rid:users:")]
    assert len(rid_lock_t1) >= 1, f"T1 debe tener al menos 1 candado de fila (RID), pero tiene {len(rid_lock_t1)}: {held}"
    
    t2_success_row2 = False
    t2_timeout_row1 = False
    t2_txn_id = None
    
    def run_t2():
        nonlocal t2_success_row2, t2_timeout_row1, t2_txn_id
        try:
            t2_txn_id = tx_manager.begin()
            run_sql("UPDATE users SET name = 'Bob_Mod' WHERE id = 2;")
            t2_success_row2 = True
            
            original_lock = tx_manager.lock
            def patched_lock(resource, mode):
                print(f"T2 attempting to lock {resource}")
                if resource.startswith("rid:users:"):
                    tx_manager.lock_manager.acquire(
                        tx_manager.current(), resource, mode, timeout=0.1
                    )
                else:
                    original_lock(resource, mode)
                    
            with patch.object(tx_manager, 'lock', side_effect=patched_lock):
                try:
                    run_sql("UPDATE users SET name = 'Hack' WHERE id = 1;")
                except LockTimeoutError:
                    t2_timeout_row1 = True
            
            tx_manager.commit()
        except Exception as e:
            print(f"T2 Error: {e}")
            try:
                tx_manager.abort()
            except RuntimeError:
                pass

    thread2 = threading.Thread(target=run_t2)
    thread2.start()
    thread2.join()
    
    t1_can_commit.set()
    thread1.join()
    
    assert t2_success_row2 is True, "Row-Level Locking falló: T2 no pudo actualizar la fila 2 inicialmente"
    assert t2_timeout_row1 is True, "Aislamiento falló: T2 logró modificar la fila 1"
    
    records = run_sql("SELECT id, name FROM users;")
    records.sort(key=lambda r: r[0].data)
    assert records[0][1].data == 'Alice_Mod'
    # T2 fue abortado debido al timeout, por lo que su actualización a la fila 2 se revirtió
    assert records[1][1].data == 'Bob'
