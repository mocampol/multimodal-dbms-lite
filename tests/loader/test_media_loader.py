"""
Tests de la carga de carpetas multimodales (load_media_directory): una fila
por archivo y, al terminar, el índice IVF, HNSW o SPIMI pedido.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf

from catalog.catalog import Catalog
from catalog.exceptions import TableAlreadyExistsError
from catalog.table_metadata import StorageType
from common.value import DataType
from loader import EmptyMediaDirectoryError, MediaDirectoryNotFoundError, load_media_directory
from main import make_heap_factory, make_index_buffer_factory, make_sequential_factory
from query.query_engine import execute


DOCUMENTS = {
    "bases.txt": "Las bases de datos guardan tablas e índices en disco.",
    "redes.txt": "Las redes conectan computadoras por medio de protocolos.",
    "cocina.txt": "La receta lleva arroz, pollo y culantro.",
}


def _make_catalog(base_dir):
    heap_factory = make_heap_factory(str(base_dir))
    return Catalog(
        heap_factory=heap_factory,
        storage_factories={
            StorageType.HEAP: heap_factory,
            StorageType.SEQUENTIAL: make_sequential_factory(str(base_dir)),
        },
        index_buffer_factory=make_index_buffer_factory(str(base_dir)),
    )


def _create_pattern_image(path, seed):
    """Imagen con figuras aleatorias: cada semilla produce descriptores distintos."""
    rng = np.random.default_rng(seed)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    for _ in range(30):
        x, y = rng.integers(0, 180, 2).tolist()
        size = int(rng.integers(5, 25))
        color = rng.integers(60, 255, 3).tolist()
        if rng.random() < 0.5:
            cv2.circle(image, (x, y), size, color, -1)
        else:
            cv2.rectangle(image, (x, y), (x + size, y + size), color, -1)
    cv2.imwrite(str(path), image)


def _create_tone(path, frequency, sr=22050):
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), sr)


def _folder(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    return folder


@pytest.fixture
def catalog(tmp_path):
    return _make_catalog(_folder(tmp_path, "db"))


@pytest.fixture
def image_dir(tmp_path):
    folder = _folder(tmp_path, "fotos")
    for i in range(1, 5):
        _create_pattern_image(folder / f"foto{i}.png", seed=i)
    return folder


@pytest.fixture
def audio_dir(tmp_path):
    folder = _folder(tmp_path, "audios")
    for i, frequency in enumerate((220, 440, 1000, 3000), start=1):
        _create_tone(folder / f"tono{i}.wav", frequency)
    return folder


@pytest.fixture
def text_dir(tmp_path):
    folder = _folder(tmp_path, "docs")
    for name, text in DOCUMENTS.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def _ids(rows):
    return [row[0].data for row in rows]


class TestLoadImages:
    def test_creates_one_row_per_image_with_its_absolute_path(self, catalog, image_dir):
        result = load_media_directory(catalog, str(image_dir), "fotos", "image")

        assert result["inserted"] == 4
        assert result["skipped"] == []
        assert result["directory"] == str(image_dir.resolve())
        assert [c.name for c in catalog.get_schema("fotos").columns] == ["id", "nombre", "ruta"]

        rows = execute("SELECT id, nombre, ruta FROM fotos ORDER BY id;", catalog)
        assert _ids(rows) == [1, 2, 3, 4]
        assert [row[1].data for row in rows] == [f"foto{i}.png" for i in range(1, 5)]
        assert all(Path(row[2].data) == image_dir.resolve() / row[1].data for row in rows)

    def test_without_index_only_loads_the_rows(self, catalog, image_dir):
        result = load_media_directory(catalog, str(image_dir), "fotos", "image", None)

        assert result["index"] is None
        assert catalog.get_indexes("fotos") == []

    def test_hnsw_index_answers_knn_queries(self, catalog, image_dir):
        result = load_media_directory(catalog, str(image_dir), "fotos", "image", "HNSW")

        assert result["index"]["index_type"] == "hnsw"
        assert result["index"]["column_name"] == "ruta"
        assert catalog.get_indexes("fotos") == [result["index"]]

        query = image_dir / "foto3.png"
        rows = execute(
            f"SELECT id FROM fotos WHERE SIMILAR_TO(ruta, '{query}', k=2) USING HNSW;", catalog
        )
        assert len(rows) == 2
        assert _ids(rows)[0] == 3

    def test_ignores_other_extensions_and_reports_empty_files(self, catalog, image_dir):
        shutil.copy(image_dir / "foto1.png", image_dir / "MAYUS.PNG")
        (image_dir / "notas.txt").write_text("no es una imagen", encoding="utf-8")
        (image_dir / "vacia.png").write_bytes(b"")

        result = load_media_directory(catalog, str(image_dir), "fotos", "image")

        assert result["inserted"] == 5
        assert result["skipped"] == [{"file": "vacia.png", "reason": "El archivo está vacío"}]


class TestLoadAudio:
    def test_ivf_index_answers_knn_queries(self, catalog, audio_dir):
        result = load_media_directory(catalog, str(audio_dir), "audios", "audio", "ivf")

        assert result["inserted"] == 4
        assert result["index"]["index_type"] == "ivf"

        query = audio_dir / "tono4.wav"
        rows = execute(
            f"SELECT id FROM audios WHERE SIMILAR_TO(ruta, '{query}', k=2) USING IVF;", catalog
        )
        assert _ids(rows)[0] == 4


class TestLoadDocuments:
    def test_spimi_index_answers_bm25_queries(self, catalog, text_dir):
        result = load_media_directory(catalog, str(text_dir), "docs", "text", "SPIMI")

        assert result["inserted"] == 3
        assert result["index"]["index_type"] == "spimi"
        assert result["index"]["column_name"] == "contenido"
        columns = catalog.get_schema("docs").columns
        assert [(c.name, c.data_type) for c in columns] == [
            ("id", DataType.INTEGER), ("titulo", DataType.VARCHAR), ("contenido", DataType.TEXT),
        ]

        sql = (
            "SELECT titulo, SCORE() AS r FROM docs WHERE MATCH(contenido, 'bases datos') "
            "USING BM25 ORDER BY r DESC;"
        )
        assert "TextIndexScan" in str(execute("EXPLAIN " + sql, catalog).description)
        rows = execute(sql, catalog)
        assert rows[0][0].data == "bases"
        assert rows[0][1].data > 0

    def test_reads_latin1_when_the_file_is_not_utf8(self, catalog, tmp_path):
        folder = _folder(tmp_path, "latin")
        (folder / "cafe.txt").write_bytes("Café con leche".encode("latin-1"))

        load_media_directory(catalog, str(folder), "docs", "text")

        rows = execute("SELECT contenido FROM docs;", catalog)
        assert rows[0][0].data == "Café con leche"

    def test_skips_documents_that_do_not_fit_in_a_row(self, catalog, tmp_path):
        folder = _folder(tmp_path, "largos")
        (folder / "corto.txt").write_text("texto corto", encoding="utf-8")
        (folder / "largo.txt").write_text("palabra " * 1000, encoding="utf-8")

        result = load_media_directory(catalog, str(folder), "docs", "text")

        assert result["inserted"] == 1
        assert [entry["file"] for entry in result["skipped"]] == ["largo.txt"]
        assert "no cabe" in result["skipped"][0]["reason"]


class TestInvalidInput:
    def test_missing_directory(self, catalog, tmp_path):
        with pytest.raises(MediaDirectoryNotFoundError, match="no existe"):
            load_media_directory(catalog, str(tmp_path / "no_existe"), "fotos", "image")

    @pytest.mark.parametrize("directory", ["", "   "])
    def test_empty_directory_path(self, catalog, directory):
        with pytest.raises(ValueError, match="Indica la carpeta"):
            load_media_directory(catalog, directory, "fotos", "image")

    def test_folder_without_files_of_that_kind(self, catalog, text_dir):
        with pytest.raises(EmptyMediaDirectoryError, match="no tiene archivos válidos"):
            load_media_directory(catalog, str(text_dir), "fotos", "image")
        assert not catalog.table_exists("fotos")

    def test_folder_with_only_empty_files(self, catalog, tmp_path):
        folder = _folder(tmp_path, "vacias")
        (folder / "vacia.png").write_bytes(b"")

        with pytest.raises(EmptyMediaDirectoryError, match="vacia.png"):
            load_media_directory(catalog, str(folder), "fotos", "image")
        assert not catalog.table_exists("fotos")

    def test_existing_table(self, catalog, image_dir):
        execute("CREATE TABLE fotos (id INTEGER PRIMARY KEY);", catalog)

        with pytest.raises(TableAlreadyExistsError, match="ya existe"):
            load_media_directory(catalog, str(image_dir), "fotos", "image")

    @pytest.mark.parametrize("kind, index", [("image", "SPIMI"), ("audio", "BTREE"), ("text", "HNSW")])
    def test_index_that_does_not_match_the_kind(self, catalog, tmp_path, kind, index):
        with pytest.raises(ValueError, match="no sirve"):
            load_media_directory(catalog, str(tmp_path), "tabla", kind, index)

    @pytest.mark.parametrize("table_name", ["", "1fotos", "mis fotos", "fotos;"])
    def test_invalid_table_name(self, catalog, image_dir, table_name):
        with pytest.raises(ValueError, match="Nombre de tabla inválido"):
            load_media_directory(catalog, str(image_dir), table_name, "image")

    def test_unknown_kind(self, catalog, image_dir):
        with pytest.raises(ValueError, match="Tipo de archivo desconocido"):
            load_media_directory(catalog, str(image_dir), "videos", "video")

    def test_failed_index_drops_the_table(self, catalog, tmp_path):
        folder = _folder(tmp_path, "rotas")
        (folder / "rota.png").write_bytes(b"esto no es una imagen")

        with pytest.raises(ValueError, match="No se pudo crear el índice HNSW"):
            load_media_directory(catalog, str(folder), "fotos", "image", "HNSW")
        assert not catalog.table_exists("fotos")
