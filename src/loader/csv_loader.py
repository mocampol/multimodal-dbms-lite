import csv
from pathlib import Path

from common.record import Record
from common.schema import Schema

from .type_inference import convert_value
from .exceptions import UnsupportedTypeError


def prepare_import(csv_path: str, table_name: str) -> Schema:
    """
    First step: infers a Schema from csv_path (one streaming pass,
    see csv_reader.infer_schema) with no overrides, for the frontend
    to show as an editable preview. Creates nothing.
    """
    from .csv_reader import infer_schema
    return infer_schema(table_name, Path(csv_path))


def load_csv(catalog, csv_path: str, schema: Schema) -> int:
    """
    Second step: takes a Schema the caller has already finalized
    (typically prepare_import()'s result, with the user's overrides
    folded in by the caller before this is invoked) and performs the
    actual import: create_table(), then a second streaming pass over
    the file that converts each row using schema.columns' types
    exactly as given, with NO re-inference of any kind.

    schema.table_name determines the table's real name — the caller
    must set it to whatever the user chose (it may differ from
    whatever placeholder name prepare_import() was called with).
    """
    csv_path = Path(csv_path)

    catalog.create_table(schema)
    storage = catalog.get_storage(schema.table_name)

    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # descarta la cabecera: el Schema ya viene decidido

        for line_number, row in enumerate(reader, start=2):
            try:
                values = [
                    convert_value(raw, column.data_type)
                    for raw, column in zip(row, schema.columns)
                ]
            except UnsupportedTypeError as exc:
                raise UnsupportedTypeError(
                    f"Fila {line_number} de '{csv_path}': {exc}"
                ) from exc

            storage.insert(Record(values))
            count += 1

    return count