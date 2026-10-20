"""
Full-text search node: filters documents (MATCH) and appends a relevance
column (SCORE). It is an ordinary Volcano node, so Sort/Limit/Projection
compose on top of it and EXPLAIN / EXPLAIN ANALYZE instrument it like any
other node.

Ranking needs corpus-wide statistics (document frequency, average
length), so open() drains the child once; next() then streams the result.
Statistics are computed over everything the child produces, *before* the
MATCH filter, so IDF reflects the whole collection and not only the hits.
"""

from typing import Optional

from common.record import Record
from common.value import DataType, Value
from query.executor.plan_node import PlanNode
from search.scoring import bm25_scores
from search.tokenizer import tokenize


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
    """

    def __init__(self, child: PlanNode, schema, output_schema, match=None, score=None):
        if match is None and score is None:
            raise ValueError("TextSearch requiere match, score o ambos")
        self.child = child
        self.schema = schema
        self.output_schema = output_schema
        self.match = match
        self.score = score
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

        scores = None
        if self.score is not None:
            documents = [tokenize(r[self._score_index].data) for r in rows]
            scores = bm25_scores(documents, tokenize(self.score[1]))

        match_terms = set(tokenize(self.match[1])) if self.match else None
        self._output = []
        for position, record in enumerate(rows):
            if match_terms is not None:
                doc_terms = set(tokenize(record[self._match_index].data))
                if not (match_terms & doc_terms):
                    continue
            if scores is None:
                self._output.append(record)
            else:
                self._output.append(Record(
                    list(record.values) + [Value(DataType.DOUBLE_PRECISION, float(scores[position]))]
                ))
        self._cursor = 0

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