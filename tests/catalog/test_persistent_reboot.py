"""
tests/catalog/test_persistent_reboot.py

Issue: "Persistent reboot tests to verify unique constraints after a reboot."

Qué se está probando:
    Catalog reconstruye self._unique_values escaneando el storage real de
    cada tabla en Catalog._load() (ver catalog.py líneas ~123-128). Eso
    significa que el conocimiento de "qué valores UNIQUE/PK ya están
    usados" NO debería depender de que el proceso de Python siga vivo:
    debe poder reconstruirse desde cero, leyendo solo lo que hay en disco.

    Este archivo simula un reboot real: crea tablas e inserta filas con
    una primera generación de objetos (Catalog, BufferManager, HeapFile),
    fuerza un flush a disco, DESCARTA esos objetos por completo, y crea
    una segunda generación desde cero apuntando a los mismos archivos.
    Si el constraint sigue vigente ahí, es porque de verdad persiste —
    no porque quedó "cacheado" en memoria dentro del mismo proceso.

Nota importante (ver conversación previa): a día de hoy NO existe un nodo
InsertStm en query/executor/ que llame a
catalog.check_insert_uniques()/register_insert_uniques() automáticamente.
Esos métodos existen en Catalog con ese contrato documentado en sus
docstrings, pero nadie los invoca todavía. Por eso este archivo define
_insert_row() como un helper de PRUEBA que reproduce exactamente esa
secuencia (documentada en catalog.py) contra código de producción real
(Catalog, HeapFile, BufferManager, FileManager) sin mockear nada. El día
que exista el nodo Insert real, basta reemplazar _insert_row() por una
llamada a ese nodo para que este mismo test siga siendo válido end-to-end.
"""

import os

import pytest

from common.value import DataType, Value
from common.schema import Schema, Column
from common.record import Record

from storage.file_manager import FileManager
from storage.buffer_manager import BufferManager
from storage.heap.heap_file import HeapFile

from catalog.catalog import Catalog
from catalog.exceptions import UniqueConstraintError


# =============================================================================
# Helpers de arranque / apagado del motor — usados solo en estos tests
# =============================================================================

def _build_heap_factory(base_dir: str, pool_size: int = 8):
    """
    Crea un heap_factory(schema) -> HeapFile que persiste cada tabla en su
    propio archivo bajo base_dir, y devuelve también el diccionario de
    BufferManager creados (uno por tabla, incluyendo sys_tables/sys_columns/
    sys_indexes) para poder flushearlos todos antes de un "reboot".
    """
    buffer_managers: dict[str, BufferManager] = {}

    def heap_factory(schema: Schema) -> HeapFile:
        file_path = os.path.join(base_dir, f"{schema.table_name}.tbl")
        file_manager = FileManager(file_path)
        buffer_manager = BufferManager(file_manager, pool_size=pool_size)
        buffer_managers[schema.table_name] = buffer_manager
        return HeapFile(schema, buffer_manager)

    return heap_factory, buffer_managers


def _boot_engine(base_dir: str):
    """
    "Enciende" el motor: crea un Catalog nuevo (que internamente hace
    Catalog._load(), reconstruyendo self.tables y self._unique_values
    desde lo que exista en base_dir). No se pasa index_buffer_factory
    porque este test no usa índices — la unicidad se resuelve a nivel de
    Catalog escaneando la tabla, no vía índice.
    """
    heap_factory, buffer_managers = _build_heap_factory(base_dir)
    catalog = Catalog(heap_factory)
    return catalog, buffer_managers


def _shutdown_engine(buffer_managers: dict[str, BufferManager]):
    """
    "Apaga" el motor de forma ordenada: fuerza a disco cualquier página
    sucia que siga solo en el Buffer Pool. Sin esto, un insert que nunca
    se desalojó del pool se perdería al descartar los objetos en memoria
    y el test de reboot no probaría nada real.
    """
    for buffer_manager in buffer_managers.values():
        buffer_manager.flush_all()


def _insert_row(catalog: Catalog, table_name: str, values: tuple):
    """
    Reproduce, contra Catalog real, la secuencia que catalog.py documenta
    para un INSERT (ver docstrings de check_insert_uniques/
    register_insert_uniques/register_insert en catalog.py):

        1. Validar unicidad ANTES de tocar storage.
        2. Insertar en el storage físico.
        3. Registrar el insert en índices secundarios (si los hay).
        4. Registrar los valores únicos usados (si el insert fue exitoso).

    Lanza UniqueConstraintError si se violaría un UNIQUE/PK — sin llegar
    a tocar storage, tal como debería comportarse el futuro nodo Insert.
    """
    schema = catalog.get_schema(table_name)
    record = Record([
        Value(column.data_type, value)
        for column, value in zip(schema.columns, values)
    ])

    violated_column = catalog.check_insert_uniques(table_name, record)
    if violated_column is not None:
        raise UniqueConstraintError(
            f"Valor duplicado para la columna única '{violated_column}' "
            f"en la tabla '{table_name}': {values}"
        )

    storage = catalog.get_storage(table_name)
    rid = storage.insert(record)
    catalog.register_insert(table_name, record, rid)
    catalog.register_insert_uniques(table_name, record)
    return rid


def _usuarios_schema() -> Schema:
    return Schema("usuarios", [
        Column("id", DataType.INTEGER, is_primary_key=True),
        Column("email", DataType.VARCHAR, size=100, is_unique=True),
        Column("ciudad", DataType.VARCHAR, size=50),  # NO única, control negativo
    ])


# =============================================================================
# Tests
# =============================================================================

def test_unique_column_rejects_duplicate_after_reboot(tmp_path):
    """
    Caso principal del issue: un valor UNIQUE insertado antes del reboot
    debe seguir bloqueando duplicados después del reboot.
    """
    base_dir = str(tmp_path)

    # --- Generación 1: crea la tabla e inserta una fila ---
    catalog, buffer_managers = _boot_engine(base_dir)
    catalog.create_table(_usuarios_schema())
    _insert_row(catalog, "usuarios", (1, "ana@example.com", "Lima"))

    # --- Apagado ordenado + se descartan TODOS los objetos en memoria ---
    _shutdown_engine(buffer_managers)
    del catalog, buffer_managers

    # --- Generación 2 ("reboot"): objetos nuevos, mismos archivos en disco ---
    catalog, buffer_managers = _boot_engine(base_dir)

    # El email duplicado debe rechazarse SIN que el proceso original exista ya
    with pytest.raises(UniqueConstraintError):
        _insert_row(catalog, "usuarios", (2, "ana@example.com", "Cusco"))

    _shutdown_engine(buffer_managers)


def test_primary_key_rejects_duplicate_after_reboot(tmp_path):
    """
    El PK es UNIQUE por definición (Column marca is_unique=True cuando
    is_primary_key=True, ver common/schema.py). Debe comportarse igual
    que cualquier otra columna UNIQUE frente a un reboot.
    """
    base_dir = str(tmp_path)

    catalog, buffer_managers = _boot_engine(base_dir)
    catalog.create_table(_usuarios_schema())
    _insert_row(catalog, "usuarios", (1, "ana@example.com", "Lima"))
    _shutdown_engine(buffer_managers)
    del catalog, buffer_managers

    catalog, buffer_managers = _boot_engine(base_dir)
    with pytest.raises(UniqueConstraintError):
        _insert_row(catalog, "usuarios", (1, "otro@example.com", "Arequipa"))
    _shutdown_engine(buffer_managers)


def test_new_unique_value_still_accepted_after_reboot(tmp_path):
    """
    Control negativo: el reboot no debe sobre-restringir. Un valor
    genuinamente nuevo debe seguir aceptándose con total normalidad
    después de reconstruir el catálogo desde disco.
    """
    base_dir = str(tmp_path)

    catalog, buffer_managers = _boot_engine(base_dir)
    catalog.create_table(_usuarios_schema())
    _insert_row(catalog, "usuarios", (1, "ana@example.com", "Lima"))
    _shutdown_engine(buffer_managers)
    del catalog, buffer_managers

    catalog, buffer_managers = _boot_engine(base_dir)
    rid = _insert_row(catalog, "usuarios", (2, "carlos@example.com", "Cusco"))
    assert rid is not None

    # Y esa fila nueva también debe quedar protegida ante un SEGUNDO reboot
    _shutdown_engine(buffer_managers)
    del catalog, buffer_managers

    catalog, buffer_managers = _boot_engine(base_dir)
    with pytest.raises(UniqueConstraintError):
        _insert_row(catalog, "usuarios", (3, "carlos@example.com", "Trujillo"))
    _shutdown_engine(buffer_managers)


def test_non_unique_column_unaffected_by_reboot(tmp_path):
    """
    Control negativo: 'ciudad' no es UNIQUE, así que valores repetidos ahí
    deben seguir siendo válidos tanto antes como después del reboot — el
    mecanismo de unicidad no debe aplicarse de más.
    """
    base_dir = str(tmp_path)

    catalog, buffer_managers = _boot_engine(base_dir)
    catalog.create_table(_usuarios_schema())
    _insert_row(catalog, "usuarios", (1, "ana@example.com", "Lima"))
    _insert_row(catalog, "usuarios", (2, "carlos@example.com", "Lima"))  # misma ciudad, OK
    _shutdown_engine(buffer_managers)
    del catalog, buffer_managers

    catalog, buffer_managers = _boot_engine(base_dir)
    # tercera fila, misma ciudad otra vez, ya después del reboot: debe seguir OK
    rid = _insert_row(catalog, "usuarios", (3, "diana@example.com", "Lima"))
    assert rid is not None
    _shutdown_engine(buffer_managers)


def test_unique_values_rebuilt_from_rows_written_in_prior_session(tmp_path):
    """
    Verifica explícitamente el mecanismo interno que hace posible todo lo
    anterior: Catalog._load() reconstruye self._unique_values escaneando
    storage, no leyendo ningún estado en memoria heredado. Se comprueba
    indirectamente (Catalog no expone _unique_values como API pública)
    insertando en una tabla que la segunda generación de Catalog nunca
    vio crearse — la única forma de que sepa que 'ana@example.com' ya
    existe es haberlo leído de vuelta del .tbl en disco.
    """
    base_dir = str(tmp_path)

    catalog, buffer_managers = _boot_engine(base_dir)
    catalog.create_table(_usuarios_schema())
    _insert_row(catalog, "usuarios", (1, "ana@example.com", "Lima"))
    _shutdown_engine(buffer_managers)
    del catalog, buffer_managers  # esta 'catalog' nunca vuelve a usarse

    catalog, buffer_managers = _boot_engine(base_dir)  # objeto 100% nuevo

    # No se llamó create_table() en esta generación: si esto pasa, es
    # porque _load() reconstruyó tables/schema correctamente desde
    # sys_tables/sys_columns en disco.
    assert catalog.table_exists("usuarios")
    assert catalog.get_schema("usuarios").get_column("email").is_unique

    with pytest.raises(UniqueConstraintError):
        _insert_row(catalog, "usuarios", (99, "ana@example.com", "Piura"))

    _shutdown_engine(buffer_managers)