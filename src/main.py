from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile
from storage.sequential.sequential_file import SequentialFile
from catalog.catalog import Catalog
from catalog.table_metadata import StorageType

def make_heap_factory(base_dir: str, pool_size: int = 64):
    def factory(schema):
        file_manager = FileManager(f"{base_dir}/{schema.table_name}.tbl")
        buffer_manager = BufferManager(file_manager, pool_size=pool_size)
        return HeapFile(schema, buffer_manager)
    return factory

def make_sequential_factory(base_dir: str, pool_size: int = 64):
    def factory(schema):
        file_manager = FileManager(f"{base_dir}/{schema.table_name}.tbl")
        buffer_manager = BufferManager(file_manager, pool_size=pool_size)
        return SequentialFile(schema, buffer_manager)
    return factory

heap_factory = make_heap_factory("data")
catalog = Catalog(
    heap_factory=heap_factory,
    storage_factories={
        StorageType.HEAP: heap_factory,
        StorageType.SEQUENTIAL: make_sequential_factory("data"),  # necesitas escribir este helper
    },
)