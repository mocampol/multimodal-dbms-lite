import csv
from pathlib import Path

from common.record import Record
from common.schema import Schema

from transaction import TransactionManager, LockMode

from .csv_reader import infer_schema as _infer_schema
from .type_inference import convert_value
from .exceptions import UnsupportedTypeError, InconsistentRowError


def prepare_import(csv_path: str, table_name: str) -> Schema:
    return _infer_schema(table_name, Path(csv_path))


def load_csv(
    catalog,
    csv_path: str,
    schema: Schema,
    on_error: str = "stop",
    batch_size: int = 1000,
) -> dict:
    if on_error not in ("stop", "ignore"):
        raise ValueError(f"on_error debe ser 'stop' o 'ignore', no {on_error!r}")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que 0")

    csv_path = Path(csv_path)
    table_name = schema.table_name

    catalog.create_table(schema)

    manager = _transaction_manager(catalog)
    inserted = 0
    skipped = []

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)

            batch = []
            for line_number, row in enumerate(reader, start=2):
                batch.append((line_number, row))
                if len(batch) >= batch_size:
                    inserted += _run_batch(catalog, manager, table_name, schema, batch, on_error, skipped)
                    batch = []

            if batch:
                inserted += _run_batch(catalog, manager, table_name, schema, batch, on_error, skipped)
    except BaseException:
        catalog.drop_table(table_name)
        raise

    return {"inserted": inserted, "skipped": skipped}


def _run_batch(catalog, manager, table_name, schema, batch, on_error, skipped) -> int:
    manager.begin()
    count = 0
    try:
        for line_number, row in batch:
            try:
                record = _row_to_record(row, schema, line_number, table_name)
                _insert_one_record(catalog, manager, table_name, record)
                count += 1
            except (UnsupportedTypeError, InconsistentRowError, ValueError) as exc:
                if on_error == "stop":
                    raise
                skipped.append({"row": line_number, "reason": str(exc)})
        manager.commit()
        return count
    except Exception:
        manager.abort()
        raise


def _row_to_record(row: list, schema: Schema, line_number: int, table_name: str) -> Record:
    if len(row) != len(schema.columns):
        raise InconsistentRowError(
            f"Fila {line_number}: {len(row)} columnas, la tabla '{table_name}' define {len(schema.columns)}"
        )

    try:
        values = [convert_value(raw, column.data_type) for raw, column in zip(row, schema.columns)]
    except UnsupportedTypeError as exc:
        raise UnsupportedTypeError(f"Fila {line_number}: {exc}") from exc

    record = Record(values)
    if not record.validate(schema):
        raise UnsupportedTypeError(
            f"Fila {line_number}: el registro no es válido para el esquema de "
            f"'{table_name}' (tipo, tamaño de VARCHAR, o NOT NULL)"
        )
    return record


def _insert_one_record(catalog, manager, table_name: str, record: Record) -> None:
    violated = catalog.check_insert_uniques(table_name, record)
    if violated:
        raise ValueError(f"Valor duplicado en columna UNIQUE '{violated}' de '{table_name}'")

    storage = catalog.get_storage(table_name)
    manager.log_data_change("INSERT", table_name, None, None, _record_data(record))
    rid = storage.insert(record)
    manager.add_undo(lambda: _undo_insert(catalog, table_name, storage, rid))
    manager.lock(f"rid:{table_name}:{rid}", LockMode.EXCLUSIVE)
    catalog.register_insert(table_name, record, rid)
    catalog.register_insert_uniques(table_name, record)


def _record_data(record: Record) -> list:
    return [{"type": value.data_type.value, "data": value.data} for value in record.values]


def _undo_insert(catalog, table_name: str, storage, rid) -> None:
    """Mismo undo que query_engine._undo_insert: borra la fila y la des-registra."""
    record = storage.get(rid)
    if record is not None:
        catalog.unregister_delete(table_name, record, rid)
        storage.delete(rid)


def _transaction_manager(catalog) -> TransactionManager:
    """
    Same lazily-created, Catalog-attached TransactionManager singleton
    query_engine.py uses (see its _transaction_manager()) — reusing it
    here means a CSV import shares the same lock table and WAL as any
    concurrent SQL session against this catalog, instead of each
    keeping its own, inconsistent view of what's locked.
    """
    manager = getattr(catalog, "_transaction_manager", None)
    if manager is None:
        manager = TransactionManager()
        catalog._transaction_manager = manager
    return manager