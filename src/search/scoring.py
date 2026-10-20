"""Okapi BM25 ranking over an in-memory collection of tokenized documents."""

import math
from collections import Counter


def bm25_scores(
    documents: list[list[str]],
    query_terms: list[str],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Returns one BM25 score (float >= 0) per document, in input order.

    Statistics (N, document frequency, average length) come from
    `documents` itself, so the caller decides which collection defines
    the corpus. Repeated query terms count once. IDF uses the
    non-negative variant ln(1 + (N - df + 0.5) / (df + 0.5)).
    """
    terms = list(dict.fromkeys(query_terms))
    total = len(documents)
    if total == 0 or not terms:
        return [0.0] * total

    frequencies = [Counter(doc) for doc in documents]
    average_length = sum(len(doc) for doc in documents) / total or 1.0
    idf = {}
    for term in terms:
        df = sum(1 for counts in frequencies if term in counts)
        idf[term] = math.log(1 + (total - df + 0.5) / (df + 0.5))

    scores = []
    for doc, counts in zip(documents, frequencies):
        norm = k1 * (1 - b + b * len(doc) / average_length)
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf:
                score += idf[term] * tf * (k1 + 1) / (tf + norm)
        scores.append(score)
    return scores