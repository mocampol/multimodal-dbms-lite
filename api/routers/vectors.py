import base64
import binascii
import time

import numpy as np
from fastapi import APIRouter

from common.record import Record
from common.value import DataType, Value
from engine import catalog
from index.multimedia import decode_raw_vector, encode_raw_vector
from index.vector import HNSWIndex, IVFIndex, VectorMetric
from index.vector.metric import compute_distances_batch, distances_to_scores_batch
from loader.csv_loader import _insert_one_record, _transaction_manager
from loader.type_inference import convert_value
from schemas import CreateIndexRequest, TableRowsInsertRequest, VectorSearchRequest
from serialize import serialize_value
from storage.latch import table_latch

router = APIRouter()


@router.post("/tables/{table_name}/rows")
def insert_table_rows(table_name: str, payload: TableRowsInsertRequest):
    schema = catalog.get_schema(table_name)
    records = [_dict_to_record(row_dict, schema) for row_dict in payload.rows]

    manager = _transaction_manager(catalog)
    manager.begin()
    manager.add_flush(("table", table_name), lambda: catalog.flush_table(table_name))
    rids = []
    try:
        for record in records:
            rid = _insert_one_record(catalog, manager, table_name, record)
            rids.append([rid.page_id, rid.slot])
        manager.commit()
    except BaseException:
        manager.abort()
        raise

    return {
        "table_name": table_name,
        "inserted": len(rids),
        "rids": rids,
    }


@router.post("/tables/{table_name}/indexes")
def create_table_index(table_name: str, payload: CreateIndexRequest):
    entry = catalog.create_index(
        table_name,
        payload.column_name,
        payload.index_type,
        index_name=payload.index_name,
        dim=payload.dim,
        index_params=payload.index_params,
    )
    physical = catalog.get_physical_index(
        table_name, payload.column_name, index_type=payload.index_type
    )
    response = {
        **entry,
        "is_spatial": entry["index_type"] == "rtree",
    }
    if hasattr(physical, "dim"):
        response["dim"] = physical.dim
    return response


@router.post("/tables/{table_name}/vector-search")
def vector_search(table_name: str, payload: VectorSearchRequest):
    t0 = time.perf_counter()
    schema = catalog.get_schema(table_name)
    storage = catalog.get_storage(table_name)
    query_vec = _decode_query_vector(payload)
    column_name = _resolve_vector_column(table_name, schema, payload.column, payload.index_type)
    column_idx = schema.column_index(column_name)
    column_type = schema.columns[column_idx].data_type
    metric = VectorMetric.from_str(payload.metric)

    index = catalog.get_physical_index(table_name, column_name, index_type=payload.index_type)
    has_vector_index = isinstance(index, (IVFIndex, HNSWIndex))
    used_index = False
    matches = []

    if has_vector_index:
        if query_vec.size != index.dim:
            raise ValueError(
                f"Dimensión del vector de consulta ({query_vec.size}) no coincide con el índice ({index.dim})"
            )
        used_index = True
        if len(index) > 0:
            with table_latch(catalog, table_name):
                raw_results = _search_with_index(index, query_vec, payload, metric)
                for res in raw_results:
                    rec = storage.get(res.rid)
                    if rec is not None:
                        matches.append((res.rid, res.score, res.distance, rec))
        resolved_dim = index.dim
    else:
        if column_type != DataType.BYTEA:
            raise ValueError(
                f"No hay índice {payload.index_type.upper()} sobre '{table_name}.{column_name}' "
                "y el fallback vectorial directo requiere una columna BYTEA"
            )
        resolved_dim = int(query_vec.size)
        with table_latch(catalog, table_name):
            rids = []
            vecs = []
            records = []
            for rid, rec in storage.scan_with_rid():
                raw_val = rec[column_idx].data
                if raw_val is None:
                    continue
                vecs.append(decode_raw_vector(raw_val, expected_dim=resolved_dim))
                rids.append(rid)
                records.append(rec)

        if vecs:
            matrix = np.vstack(vecs).astype(np.float32, copy=False)
            dists = compute_distances_batch(query_vec, matrix, metric)
            scores = distances_to_scores_batch(dists, metric)
            top_k = min(payload.k, len(rids))
            order = np.argsort(dists, kind="stable")[:top_k]
            for idx in order:
                i = int(idx)
                matches.append((rids[i], float(scores[i]), float(dists[i]), records[i]))

    search_ms = (time.perf_counter() - t0) * 1000.0
    col_names = [col.name for col in schema.columns]
    serialized = [
        {
            "rid": [rid.page_id, rid.slot],
            "score": float(score),
            "distance": float(distance),
            "row": {
                name: serialize_value(val)
                for name, val in zip(col_names, rec.values)
            },
        }
        for rid, score, distance, rec in matches
    ]

    return {
        "table_name": table_name,
        "column": column_name,
        "index_type": payload.index_type,
        "used_index": used_index,
        "dim": resolved_dim,
        "k": payload.k,
        "metric": metric.value,
        "search_ms": search_ms,
        "results": serialized,
    }


def _search_with_index(index, query_vec: np.ndarray, payload: VectorSearchRequest, metric: VectorMetric):
    if isinstance(index, HNSWIndex) and payload.ef_search is not None:
        prev_ef = index.ef_search
        try:
            index.set_ef_search(payload.ef_search)
            return index.search(query_vec, k=payload.k, metric=metric)
        finally:
            index.set_ef_search(prev_ef)
    if isinstance(index, IVFIndex) and payload.n_probe is not None:
        prev_probe = index.n_probe
        try:
            index.n_probe = payload.n_probe
            return index.search(query_vec, k=payload.k, metric=metric)
        finally:
            index.n_probe = prev_probe
    return index.search(query_vec, k=payload.k, metric=metric)


def _decode_query_vector(payload: VectorSearchRequest) -> np.ndarray:
    if payload.vector is not None and payload.vector_b64 is not None:
        raise ValueError("Envía 'vector' o 'vector_b64', no ambos")
    if payload.vector is not None:
        return decode_raw_vector(payload.vector)
    if payload.vector_b64 is not None:
        try:
            raw = base64.b64decode(payload.vector_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"vector_b64 inválido: {exc}") from exc
        return decode_raw_vector(raw)
    raise ValueError("Debes enviar 'vector' o 'vector_b64' para buscar")


def _resolve_vector_column(table_name: str, schema, requested: str | None, index_type: str) -> str:
    if requested is not None:
        if schema.get_column(requested) is None:
            raise ValueError(f"La columna '{requested}' no existe en '{table_name}'")
        return requested
    for col in schema.columns:
        idx = catalog.get_physical_index(table_name, col.name, index_type=index_type)
        if isinstance(idx, (IVFIndex, HNSWIndex)):
            return col.name
    for col in schema.columns:
        if col.data_type == DataType.BYTEA:
            return col.name
    raise ValueError(f"No se encontró columna vectorial en '{table_name}'")


def _dict_to_record(row_dict: dict, schema) -> Record:
    unknown = set(row_dict) - {col.name for col in schema.columns}
    if unknown:
        raise ValueError(f"Columnas desconocidas para '{schema.table_name}': {sorted(unknown)}")

    values = []
    for col in schema.columns:
        raw = row_dict.get(col.name)
        if raw is None:
            values.append(Value(col.data_type, None))
            continue
        if col.data_type == DataType.BYTEA:
            values.append(Value(DataType.BYTEA, _coerce_bytea(raw)))
        elif col.data_type == DataType.BOOLEAN and isinstance(raw, bool):
            values.append(Value(DataType.BOOLEAN, raw))
        elif col.data_type in (DataType.SMALLINT, DataType.INTEGER, DataType.BIGINT) and isinstance(raw, int) and not isinstance(raw, bool):
            values.append(Value(col.data_type, raw))
        elif col.data_type in (DataType.REAL, DataType.DOUBLE_PRECISION) and isinstance(raw, (int, float)) and not isinstance(raw, bool):
            values.append(Value(col.data_type, float(raw)))
        elif col.data_type in (DataType.CHAR, DataType.VARCHAR, DataType.TEXT) and isinstance(raw, str):
            values.append(Value(col.data_type, raw))
        else:
            values.append(convert_value(str(raw), col.data_type))

    record = Record(values)
    if not record.validate(schema):
        raise ValueError(f"El registro no cumple el esquema de '{schema.table_name}'")
    return record


def _coerce_bytea(raw) -> bytes:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, list):
        return encode_raw_vector(raw)
    if isinstance(raw, str):
        try:
            return base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Valor BYTEA debe ser base64 válido o lista de floats: {exc}") from exc
    raise ValueError(f"Tipo inválido para columna BYTEA: {type(raw)}")
