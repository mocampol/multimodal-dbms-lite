"""
Inverted Index representation for full-text search.

Stores postings in the format: { term: [(doc_id, term_frequency), ...] }
Provides lookup for term frequency, document frequency, and BM25 ranking.
"""

from __future__ import annotations

import math
import pickle
from typing import Sequence


def rid_to_doc_id(page_id: int, slot: int) -> int:
    """Bijective O(1) mapping from RID(page_id, slot) to positive 32-bit doc_id."""
    return (page_id << 16) | (slot & 0xFFFF)


def doc_id_to_rid(doc_id: int) -> tuple[int, int]:
    """Inverse O(1) mapping from doc_id back to (page_id, slot)."""
    return (doc_id >> 16, doc_id & 0xFFFF)


class InvertedIndex:
    """
    Inverted index storing term postings (doc_id, tf) and corpus metadata
    for scoring and retrieval.
    """

    def __init__(
        self,
        index: dict[str, list[tuple[int, int]]] | None = None,
        doc_lengths: dict[int, int] | None = None,
        total_docs: int = 0,
        file_path: str | None = None,
        manager: object | None = None,
    ):
        # term -> list of (doc_id, tf) sorted by doc_id
        self.index: dict[str, list[tuple[int, int]]] = index or {}
        # doc_id -> number of tokens in the document
        self.doc_lengths: dict[int, int] = doc_lengths or {}
        # Total number of indexed documents
        self.total_docs: int = total_docs
        # Cached document Euclidean norms: doc_id -> ||V(D)||
        self._doc_norms: dict[int, float] = {}
        # Physical storage binding for Catalog integration
        self.file_path: str | None = file_path
        self.bm: object | None = manager

    @property
    def avg_doc_length(self) -> float:
        """Average document length in the corpus."""
        if not self.total_docs or not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / self.total_docs

    def get_terms(self) -> list[str]:
        """Returns sorted list of all unique vocabulary terms in the index."""
        return sorted(self.index.keys())

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """Returns the postings list [(doc_id, tf), ...] for a given term."""
        return self.index.get(term, [])

    def get_document_frequency(self, term: str) -> int:
        """Returns df(term): the number of documents containing the term."""
        return len(self.index.get(term, []))

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Returns tf(term, doc_id): frequency of term in the specified document."""
        postings = self.index.get(term, [])
        for d_id, tf in postings:
            if d_id == doc_id:
                return tf
            if d_id > doc_id:
                break
        return 0

    def get_doc_length(self, doc_id: int) -> int:
        """Returns length (in tokens) of the given document."""
        return self.doc_lengths.get(doc_id, 0)

    def idf(self, term: str) -> float:
        """
        Calculates non-negative Robertson-Spärck Jones IDF (used in Lucene/Okapi BM25):
        IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
        """
        df = self.get_document_frequency(term)
        if df == 0 or self.total_docs == 0:
            return 0.0
        return math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))

    def idf_smooth(self, term: str) -> float:
        """
        Calculates standard smooth IDF for Vector Space Model (TF-IDF):
        idf(t) = ln(1 + N / df(t))
        Returns 0.0 if df(t) == 0 or total_docs == 0.
        """
        df = self.get_document_frequency(term)
        if df == 0 or self.total_docs == 0:
            return 0.0
        return math.log(1.0 + (self.total_docs / df))

    def compute_doc_norms(self) -> dict[int, float]:
        """
        Computes and caches Euclidean norm ||V(D)|| for each document in the corpus:
        ||V(D)|| = sqrt(sum_{t in D} (tf(t, D) * idf_smooth(t))^2)
        """
        sum_sq: dict[int, float] = {}
        for term, postings in self.index.items():
            term_idf = self.idf_smooth(term)
            if term_idf <= 0.0:
                continue
            for doc_id, tf in postings:
                w = tf * term_idf
                sum_sq[doc_id] = sum_sq.get(doc_id, 0.0) + (w * w)

        self._doc_norms = {
            doc_id: math.sqrt(val) for doc_id, val in sum_sq.items()
        }
        return self._doc_norms

    def get_doc_norm(self, doc_id: int) -> float:
        """Returns the Euclidean norm ||V(D)|| for a given document."""
        if not self._doc_norms and self.index and self.total_docs > 0:
            self.compute_doc_norms()
        return self._doc_norms.get(doc_id, 0.0)

    def score_bm25(
        self,
        query_tokens: Sequence[str],
        k1: float = 1.2,
        b: float = 0.75,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Computes Okapi BM25 scores for candidates matching at least one query term.
        Returns top_k tuples of (doc_id, score) sorted descending by score.
        """
        if not query_tokens or self.total_docs == 0:
            return []

        avgdl = self.avg_doc_length or 1.0
        scores: dict[int, float] = {}

        # Disjunctive scoring across query terms
        for term in query_tokens:
            postings = self.get_postings(term)
            if not postings:
                continue

            term_idf = self.idf(term)
            for doc_id, tf in postings:
                doc_len = self.get_doc_length(doc_id)
                # BM25 tf normalization factor
                tf_norm = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + (term_idf * tf_norm)

        # Sort descending by score, tie-break by doc_id ascending
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def score_tfidf_cosine(
        self,
        query_terms: Sequence[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Computes Cosine Similarity scores using TF-IDF over candidate documents:
        sim(Q, D) = (V(Q) · V(D)) / (||V(Q)|| * ||V(D)||)

        Handles non-indexable terms, empty queries, and zero-norm documents safely.
        Results are sorted descending by cosine score with deterministic tie-breaking
        by doc_id ascending: key=(-score, doc_id).

        Returns top_k tuples of (doc_id, score) or all candidate documents if top_k is None.
        """
        if not query_terms or self.total_docs == 0:
            return []

        # Count term frequencies in query tf(t, Q)
        query_tf: dict[str, int] = {}
        for term in query_terms:
            query_tf[term] = query_tf.get(term, 0) + 1

        # Compute query vector coordinates and query Euclidean norm ||V(Q)||
        query_weights: dict[str, float] = {}
        query_norm_sq = 0.0
        for term, q_tf in query_tf.items():
            term_idf = self.idf_smooth(term)
            if term_idf > 0.0:
                w_q = q_tf * term_idf
                query_weights[term] = w_q
                query_norm_sq += w_q * w_q

        # If query has no indexable or known terms, norm is 0 -> return [] safely
        if query_norm_sq == 0.0:
            return []

        query_norm = math.sqrt(query_norm_sq)

        # Accumulate dot product V(Q) · V(D) across candidate documents
        dot_products: dict[int, float] = {}
        for term, w_q in query_weights.items():
            postings = self.get_postings(term)
            term_idf = self.idf_smooth(term)
            for doc_id, d_tf in postings:
                w_d = d_tf * term_idf
                dot_products[doc_id] = dot_products.get(doc_id, 0.0) + (w_q * w_d)

        if not dot_products:
            return []

        # Ensure doc norms are populated
        if not self._doc_norms:
            self.compute_doc_norms()

        # Compute cosine similarity: dot_product / (||V(Q)|| * ||V(D)||)
        scores: dict[int, float] = {}
        for doc_id, dot_prod in dot_products.items():
            doc_norm = self.get_doc_norm(doc_id)
            if doc_norm <= 0.0:
                continue
            sim = dot_prod / (query_norm * doc_norm)
            # Bound numerically to [0.0, 1.0] against precision overflow
            sim = min(1.0, max(0.0, sim))
            scores[doc_id] = sim

        # Deterministic ranking: score descending, then doc_id ascending
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            return ranked[:top_k]
        return ranked

    def bm25_search(
        self,
        query_tokens: Sequence[str],
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[tuple[int, float]]:
        """Convenience alias for score_bm25."""
        return self.score_bm25(query_tokens, top_k=top_k, k1=k1, b=b)

    def cosine_similarity_search(
        self,
        query_tokens: Sequence[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Convenience alias for score_tfidf_cosine."""
        return self.score_tfidf_cosine(query_tokens, top_k=top_k)

    def save(self, file_path: str | None = None) -> None:
        """Persists the inverted index to disk."""
        target_path = file_path or self.file_path
        if not target_path:
            raise ValueError("No file_path provided for saving InvertedIndex")
        state = {
            "index": self.index,
            "doc_lengths": self.doc_lengths,
            "total_docs": self.total_docs,
            "doc_norms": self._doc_norms,
        }
        with open(target_path, "wb") as f:
            pickle.dump(state, f, protocol=5)

    @classmethod
    def load(cls, file_path: str, manager: object | None = None) -> InvertedIndex:
        """Loads a persisted InvertedIndex from disk."""
        import os
        if not os.path.exists(file_path):
            return cls(file_path=file_path, manager=manager)
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        idx = cls(
            index=state["index"],
            doc_lengths=state["doc_lengths"],
            total_docs=state["total_docs"],
            file_path=file_path,
            manager=manager,
        )
        idx._doc_norms = state.get("doc_norms", {})
        return idx

    def add_document(self, doc_id: int, tokens: Sequence[str]) -> None:
        """Indexes a single document dynamically (for Catalog DML register_insert)."""
        if not tokens:
            self.doc_lengths[doc_id] = 0
            self.total_docs += 1
            self._doc_norms.clear()
            return

        term_counts: dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        for term, tf in term_counts.items():
            postings = self.index.setdefault(term, [])
            postings.append((doc_id, tf))
            postings.sort(key=lambda x: x[0])

        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        self._doc_norms.clear()

    def remove_document(self, doc_id: int) -> None:
        """Removes a document from postings (for Catalog DML unregister_delete)."""
        if doc_id not in self.doc_lengths:
            return
        del self.doc_lengths[doc_id]
        self.total_docs = max(0, self.total_docs - 1)
        self._doc_norms.clear()

        # Remove from postings
        empty_terms = []
        for term, postings in self.index.items():
            new_postings = [p for p in postings if p[0] != doc_id]
            if not new_postings:
                empty_terms.append(term)
            else:
                self.index[term] = new_postings
        for term in empty_terms:
            del self.index[term]
