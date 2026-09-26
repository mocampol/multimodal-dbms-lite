import csv
from pathlib import Path

from common.schema import Schema, Column
from common.value import DataType

from .exceptions import EmptyCSVError, InconsistentRowError, DuplicateColumnNameError
from .type_inference import narrow_type, narrow_varchar_size, resolve_varchar_size


def infer_schema(table_name: str, csv_path: Path) -> Schema:
    """
    Infers a full Schema for table_name from csv_path's header and
    values, reading the file exactly once, row by row. table_name may
    be a placeholder if called before the user has chosen a final
    name (e.g. during a preview step) — the caller can rename the
    resulting Schema before passing it to load_csv().
    """
    csv_path = Path(csv_path)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
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
        candidates: list = [None] * n_cols
        max_lens: list = [0] * n_cols
        saw_any_row = False

        for line_number, row in enumerate(reader, start=2):
            if len(row) != n_cols:
                raise InconsistentRowError(
                    f"Fila {line_number} de '{csv_path}' tiene {len(row)} columnas, "
                    f"pero la cabecera define {n_cols}"
                )
            saw_any_row = True
            for i, raw in enumerate(row):
                candidates[i] = narrow_type(candidates[i], raw)
                max_lens[i] = narrow_varchar_size(max_lens[i], raw)

    if not saw_any_row:
        raise EmptyCSVError(f"'{csv_path}' no tiene ninguna fila de datos, solo cabecera")

    columns = []
    for i, name in enumerate(header):
        data_type = candidates[i] if candidates[i] is not None else DataType.VARCHAR
        size = resolve_varchar_size(max_lens[i]) if data_type == DataType.VARCHAR else None
        columns.append(Column(name, data_type, size=size, nullable=True))

    return Schema(table_name, columns)