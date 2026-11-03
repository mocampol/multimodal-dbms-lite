from __future__ import annotations

from typing import Optional
import numpy as np

from common.record import Record
from common.value import DataType, Value
from index.multimedia import ExtractFeatures
from index.vector import VectorMetric, IVFIndex, HNSWIndex, compute_distance, distance_to_score
from query.executor.plan_node import PlanNode
from query.parser.ast_nodes import SimilarToExp, SimilarityScoreSpec, VectorIndexKind, VectorMetricKind


class VectorSearch(PlanNode):
    """
    Nodo de procesamiento para búsqueda por similitud multimedia (SIMILAR_TO / SIMILARITY_SCORE).
    Acelera la búsqueda mediante índices IVF / HNSW si existen, o ejecuta escaneo secuencial.
    """

    METRIC_MAP = {
        VectorMetricKind.EUCLIDEAN: VectorMetric.EUCLIDEAN,
        VectorMetricKind.COSINE: VectorMetric.COSINE,
        VectorMetricKind.DOT_PRODUCT: VectorMetric.DOT_PRODUCT,
    }

    def __init__(
        self,
        child: PlanNode,
        schema,
        output_schema,
        catalog,
        table_name: str,
        similar_to: Optional[SimilarToExp] = None,
        score: Optional[SimilarityScoreSpec] = None,
    ) -> None:
        if similar_to is None and score is None:
            raise ValueError("VectorSearch requiere similar_to, score o ambos")

        self.child = child
        self.schema = schema
        self.output_schema = output_schema
        self.catalog = catalog
        self.table_name = table_name
        self.similar_to = similar_to
        self.score = score
        self._output: list[Record] = []
        self._cursor: int = 0

    def open(self) -> None:
        self._output = []
        self._cursor = 0

        target_obj = self.similar_to.target if self.similar_to else self.score.target
        k_val = self.similar_to.k if self.similar_to else (self.score.k if self.score else 10)
        metric_kind = self.similar_to.metric if self.similar_to else self.score.metric
        metric = self.METRIC_MAP.get(metric_kind, VectorMetric.COSINE)

        extractor = ExtractFeatures(k=64)
        query_fv = extractor.extract(target_obj)

        column_name = None
        if self.similar_to and self.similar_to.column:
            column_name = self.similar_to.column.value
            if "." in column_name:
                column_name = column_name.rsplit(".", 1)[-1]
        elif self.score and self.score.column:
            column_name = self.score.column
            if "." in column_name:
                column_name = column_name.rsplit(".", 1)[-1]

        if column_name is None:
            # Buscar columna con índice vectorial en el catálogo
            for col in self.schema.columns:
                idx = self.catalog.get_physical_index(self.table_name, col.name)
                if idx is not None and isinstance(idx, (IVFIndex, HNSWIndex)):
                    column_name = col.name
                    break

        if column_name is None:
            # Seleccionar la primera columna de tipo texto o la primera de la tabla
            for col in self.schema.columns:
                if col.data_type in (DataType.VARCHAR, DataType.TEXT):
                    column_name = col.name
                    break

        if column_name is None and self.schema.columns:
            column_name = self.schema.columns[0].name

        # Verificar si hay índice vectorial aplicable
        index = self.catalog.get_physical_index(self.table_name, column_name) if column_name else None
        used_index = False

        if index is not None and isinstance(index, (IVFIndex, HNSWIndex)) and len(index) > 0:
            try:
                results = index.search(query_fv.data, k=k_val, metric=metric)
                storage = self.catalog.get_storage(self.table_name)
                for res in results:
                    rec = storage.get(res.rid)
                    if rec is not None:
                        if self.score is not None:
                            val_score = Value(DataType.DOUBLE_PRECISION, float(res.score))
                            rec = Record(list(rec.values) + [val_score])
                        self._output.append(rec)
                used_index = True
            except Exception:
                used_index = False

        if not used_index:
            self.child.open()
            rows: list[Record] = []
            while True:
                rec = self.child.next()
                if rec is None:
                    break
                rows.append(rec)
            self.child.close()

            col_idx = self.schema.column_index(column_name) if column_name else 0
            scored_rows: list[tuple[float, float, Record]] = []

            for rec in rows:
                val = rec[col_idx].data
                if val is None:
                    continue
                try:
                    fv = extractor.extract(val)
                    dist = compute_distance(query_fv.data, fv.data, metric)
                    score_val = distance_to_score(dist, metric)
                    scored_rows.append((dist, score_val, rec))
                except Exception:
                    continue

            scored_rows.sort(key=lambda item: item[0])
            top_k = scored_rows[:k_val]

            for dist, score_val, rec in top_k:
                if self.score is not None:
                    val_score = Value(DataType.DOUBLE_PRECISION, float(score_val))
                    rec = Record(list(rec.values) + [val_score])
                self._output.append(rec)

    def next(self) -> Optional[Record]:
        if self._cursor < len(self._output):
            rec = self._output[self._cursor]
            self._cursor += 1
            return rec
        return None

    def close(self) -> None:
        self.child.close()
        self._output = []
        self._cursor = 0
