import csv
from pathlib import Path

from common.schema import Schema, Column
from common.value import DataType

from .exceptions import EmptyCSVError, InconsistentRowError, DuplicateColumnNameError
from .type_inference import narrow_type, narrow_varchar_size, resolve_varchar_size


def infer_schema(table_name: str, csv_path: Path, overrides: dict[str, DataType] = None) -> Schema:
    """
    Infers a full Schema for table_name from csv_path's header and
    values, reading the file exactly once, row by row.

    `overrides`, if given, maps column_name -> DataType for any column
    whose type the user picked explicitly in the frontend instead of
    accepting the automatic guess. Overridden columns are skipped
    during inference entirely (no candidate type or max_len is tracked
    for them) — their size (only relevant for CHAR/VARCHAR) falls back
    to a fixed default rather than one computed from the data, since
    the override may pick a type inference never considers on its own
    (e.g. NUMERIC or DATE).
    """
    overrides = overrides or {}
    csv_path = Path(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise EmptyCSVError(f"'{csv_path}' está vacío: no tiene ni siquiera una cabecera")

        seen_names = set()
        for name in header:
            if name in seen_names:
                raise DuplicateColumnNameError(
                    f"La columna '{name}' aparece más de una vez en la cabecera de '{csv_path}'"
                )
            seen_names.add(name)

        n_cols = len(header)
        candidates: list[DataType | None] = [None] * n_cols
        max_lens: list[int] = [0] * n_cols
        skip_column = [name in overrides for name in header]
        saw_any_row = False

        for line_number, row in enumerate(reader, start=2):
            if len(row) != n_cols:
                raise InconsistentRowError(
                    f"Fila {line_number} de '{csv_path}' tiene {len(row)} columnas, "
                    f"pero la cabecera define {n_cols}"
                )
            saw_any_row = True
            for i, raw in enumerate(row):
                if skip_column[i]:
                    continue
                candidates[i] = narrow_type(candidates[i], raw)
                max_lens[i] = narrow_varchar_size(max_lens[i], raw)

    if not saw_any_row:
        raise EmptyCSVError(f"'{csv_path}' no tiene ninguna fila de datos, solo cabecera")

    columns = []
    for i, name in enumerate(header):
        if name in overrides:
            data_type = overrides[name]
            size = 64 if data_type in (DataType.CHAR, DataType.VARCHAR) else None
        else:
            data_type = candidates[i] if candidates[i] is not None else DataType.VARCHAR
            size = resolve_varchar_size(max_lens[i]) if data_type == DataType.VARCHAR else None

        columns.append(Column(name, data_type, size=size, nullable=True))

    return Schema(table_name, columns)