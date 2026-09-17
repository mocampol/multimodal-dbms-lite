import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from main import catalog


def flush_all_buffers() -> None:
    for table_name in list(catalog.tables):
        storage = catalog.get_storage(table_name)
        storage.bm.flush_all()
        if hasattr(storage, "overflow"):
            storage.overflow.bm.flush_all()

    for entries in catalog.indexes.values():
        for entry in entries:
            index = catalog._physical_indexes.get(entry["index_id"])
            if index is None:
                continue
            bm = getattr(index, "bm", None) or getattr(index, "buffer_manager", None)
            if bm is not None:
                bm.flush_all()
