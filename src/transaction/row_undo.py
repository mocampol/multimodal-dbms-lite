"""
Runtime undo of row changes, run by TransactionManager.abort().

Every undo step is itself logged as a DATA record of the aborting
transaction (a compensation record). If the process crashes mid-abort,
recovery walks the log backwards and cancels these compensations before
undoing the original changes, so nothing is restored twice.
"""

from storage.latch import table_latch


def undo_insert(manager, catalog, table_name, storage, rid):
    with table_latch(catalog, table_name):
        record = storage.get(rid)
        if record is None:
            return
        manager.log_data_change("DELETE", table_name, rid, record, None)
        catalog.unregister_delete(table_name, record, rid)
        storage.delete(rid)


def undo_delete(manager, catalog, table_name, storage, record):
    with table_latch(catalog, table_name):
        rid = storage.insert(record)
        manager.log_data_change("INSERT", table_name, rid, None, record)
        catalog.register_insert(table_name, record, rid)
        catalog.register_insert_uniques(table_name, record)


def undo_update(manager, catalog, table_name, storage, new_rid, current, previous):
    with table_latch(catalog, table_name):
        restored_rid = storage.update(new_rid, previous)
        manager.log_data_change(
            "UPDATE", table_name, new_rid, current, previous, new_rid=restored_rid,
        )
        catalog.register_update(table_name, new_rid, restored_rid, current, previous)
