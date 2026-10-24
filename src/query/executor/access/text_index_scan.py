"""
Full-text access node backed by the catalog's inverted index (SPIMI/GIN).

With MATCH, the index yields the candidates (documents containing at least
one query term) already ranked by InvertedIndex.score_bm25, and each one is
fetched from the heap by its RID. With SCORE() and no MATCH, every row is
read and gets the index score of its document (0.0 when it is not a
candidate).

Corpus statistics (N, df, avgdl) are the index's own, so a SQL score is
exactly the one the index reports; no parallel BM25 structure is built.
"""

from typing import Optional

from common.record import Record
from common.value import DataType, Value
from index.text import doc_id_to_rid, rid_to_doc_id
from query.executor.plan_node import PlanNode
from storage.heap.rid import RID
from storage.latch import NO_LATCH, latched_scan


class TextIndexScan(PlanNode):
    """
    Args:
        index: InvertedIndex over `column_name`.
        storage: HeapFile of the table (doc_id <-> RID is only defined there).
        processor: TextProcessor the index was built with, so the query is
            tokenized exactly like the documents.
        query: full-text query string.
        match: True filters to the index candidates; False keeps every row.
        score: True appends the BM25 score as a DOUBLE_PRECISION last column.
    """

    def __init__(
        self,
        index,
        storage,
        processor,
        query: str,
        match: bool = True,
        score: bool = False,
        lock_rid=None,
        latch=NO_LATCH,
        table_name: str = "",
        column_name: str = "",
    ):
        if not match and not score:
            raise ValueError("TextIndexScan requiere match, score o ambos")
        self.index = index
        self.storage = storage
        self.processor = processor
        self.query = query
        self.match = match
        self.score = score
        self.lock_rid = lock_rid
        self.latch = latch
        self.table_name = table_name
        self.column_name = column_name
        self._records = []
        self._cursor = 0

    def open(self) -> None:
        with self.latch:
            ranked = self.index.score_bm25(self.processor.process(self.query), top_k=None)

        self._records = []
        if self.match:
            for doc_id, score in ranked:
                record = self._fetch(RID(*doc_id_to_rid(doc_id)))
                if record is not None:
                    self._records.append(self._output(record, score))
        else:
            scores = dict(ranked)
            for rid, record in latched_scan(self.storage.scan_with_rid(), self.latch):
                if self.lock_rid is not None:
                    record = self._fetch(rid)
                    if record is None:
                        continue
                doc_id = rid_to_doc_id(rid.page_id, rid.slot)
                self._records.append(self._output(record, scores.get(doc_id, 0.0)))
        self._cursor = 0

    def _fetch(self, rid) -> Optional[Record]:
        # lock first, latch only for the read: never wait for a lock under a latch
        if self.lock_rid is not None:
            from transaction.lock_manager import LockMode
            self.lock_rid(rid, LockMode.SHARED)
        with self.latch:
            return self.storage.get(rid)

    def _output(self, record: Record, score: float) -> Record:
        if not self.score:
            return record
        return Record(list(record.values) + [Value(DataType.DOUBLE_PRECISION, float(score))])

    def next(self) -> Optional[Record]:
        if self._cursor >= len(self._records):
            return None
        record = self._records[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self._records = []
        self._cursor = 0

    def describe_self(self) -> dict:
        return {
            "node": "TextIndexScan",
            "access": "text_index_scan",
            "table": self.table_name,
            "column": self.column_name,
            "query": self.query,
            "ranking": "bm25",
            "mode": "match" if self.match else "score",
        }
