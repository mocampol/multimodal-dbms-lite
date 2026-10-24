"""
Full-text search node for columns without an inverted index: filters
documents (MATCH) and appends a relevance column (SCORE). It is an ordinary
Volcano node, so Sort/Limit/Projection compose on top of it and EXPLAIN /
EXPLAIN ANALYZE instrument it like any other node. Columns with a GIN/SPIMI
index are answered by TextIndexScan instead.

Ranking needs corpus-wide statistics (document frequency, average
length), so open() drains the child once and builds a transient
InvertedIndex over it, exactly like the catalog builds a persistent one
(same TextProcessor, NULL values are not documents). Scores therefore come
from InvertedIndex.score_bm25, the only BM25 implementation, and match what
the same rows would get with an index. Statistics cover everything the
child produces, *before* the MATCH filter, so IDF reflects the whole
collection and not only the hits.
"""

from typing import Optional

from common.record import Record
from common.value import DataType, Value
from index.text import InvertedIndex, default_text_processor
from query.executor.plan_node import PlanNode


class TextSearch(PlanNode):
    """
    Args:
        child: source of documents (normally a SeqScan).
        schema: schema of the child's records.
        output_schema: `schema` plus the score column when `score` is set
            (otherwise identical to `schema`).
        match: (column_name, query) or None. A document passes when it
            contains at least one query term (OR semantics).
        score: (column_name, query, alias) or None. Appends a
            DOUBLE_PRECISION BM25 value as the last column of each record.
        processor: TextProcessor for documents and queries; defaults to the
            one the catalog indexes with.
    """

    def __init__(self, child: PlanNode, schema, output_schema, match=None, score=None, processor=None):
        if match is None and score is None:
            raise ValueError("TextSearch requiere match, score o ambos")
        self.child = child
        self.schema = schema
        self.output_schema = output_schema
        self.match = match
        self.score = score
        self.processor = processor or default_text_processor()
        self._match_index = schema.column_index(match[0]) if match else None
        self._score_index = schema.column_index(score[0]) if score else None
        self._output = []
        self._cursor = 0

    def open(self) -> None:
        self.child.open()
        rows = []
        while True:
            record = self.child.next()
            if record is None:
                break
            rows.append(record)

        matches = self._rank(rows, self._match_index, self.match[1]) if self.match else None
        scores = None
        if self.score is not None:
            same_search = self.match is not None and tuple(self.score[:2]) == tuple(self.match)
            scores = matches if same_search else self._rank(rows, self._score_index, self.score[1])

        self._output = []
        for position, record in enumerate(rows):
            # BM25 candidates are exactly the documents sharing a query term
            if matches is not None and position not in matches:
                continue
            if scores is None:
                self._output.append(record)
            else:
                self._output.append(Record(
                    list(record.values) + [Value(DataType.DOUBLE_PRECISION, float(scores.get(position, 0.0)))]
                ))
        self._cursor = 0

    def _rank(self, rows, column_index, query) -> dict:
        """{row position: BM25 score} for the candidates of `query`, from a
        transient InvertedIndex over `rows` keyed by row position."""
        index = InvertedIndex()
        for position, record in enumerate(rows):
            text = record[column_index].data
            if text is not None:
                index.add_document(position, self.processor.process(str(text)))
        return dict(index.score_bm25(self.processor.process(query), top_k=None))

    def next(self) -> Optional[Record]:
        if self._cursor >= len(self._output):
            return None
        record = self._output[self._cursor]
        self._cursor += 1
        return record

    def close(self) -> None:
        self.child.close()
        self._output = []
        self._cursor = 0

    def children(self) -> list:
        return [self.child]

    def replace_children(self, new_children: list) -> None:
        (self.child,) = new_children

    def describe_self(self) -> dict:
        info = {"node": "TextSearch", "ranking": "bm25"}
        if self.match:
            info["match"] = {"column": self.match[0], "query": self.match[1]}
        if self.score:
            info["score"] = {
                "column": self.score[0], "query": self.score[1], "alias": self.score[2],
            }
        return info
