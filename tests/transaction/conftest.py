import pytest

from catalog.catalog import Catalog
from catalog.table_metadata import StorageType
from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.heap.heap_file import HeapFile
from storage.sequential.sequential_file import SequentialFile
from transaction.transaction_manager import TransactionManager


def open_catalog(base_dir):
    """Opens (or reopens, simulating an engine restart) the database in base_dir."""
    base_dir = str(base_dir)

    def heap_factory(schema):
        bm = BufferManager(FileManager(f"{base_dir}/{schema.table_name}.tbl"), pool_size=64)
        return HeapFile(schema, bm)

    def sequential_factory(schema):
        return SequentialFile(
            schema,
            key_column=schema.primary_key().name,
            buffer_manager=BufferManager(FileManager(f"{base_dir}/{schema.table_name}.tbl"), pool_size=64),
            overflow_buffer_manager=BufferManager(
                FileManager(f"{base_dir}/{schema.table_name}.overflow.tbl"), pool_size=64
            ),
        )

    def index_buffer_factory(table_name, column_name, index_id):
        return BufferManager(
            FileManager(f"{base_dir}/{table_name}.{column_name}.{index_id}.idx"), pool_size=64
        )

    catalog = Catalog(
        heap_factory=heap_factory,
        storage_factories={StorageType.HEAP: heap_factory, StorageType.SEQUENTIAL: sequential_factory},
        index_buffer_factory=index_buffer_factory,
    )
    catalog._transaction_manager = TransactionManager(log_path=f"{base_dir}/transactions.wal")
    return catalog


@pytest.fixture
def catalog(tmp_path):
    return open_catalog(tmp_path)


@pytest.fixture
def reopen():
    return open_catalog
