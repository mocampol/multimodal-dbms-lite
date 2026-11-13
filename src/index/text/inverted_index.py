"""
Inverted Index representation for full-text search.

Stores postings in the format: { term: [(doc_id, term_frequency), ...] }
Provides lookup for term frequency, document frequency, and BM25 ranking.
"""

from __future__ import annotations

import math
import os
import pickle
import struct
from collections.abc import MutableMapping
from typing import Iterator, Sequence

from storage.buffer_manager import BufferManager
from storage.file_manager import FileManager
from storage.page import PAGE_SIZE

BM25_K1 = 1.2
BM25_B = 0.75
_INDEX_MAGIC = b"FTI1"
_PAGE_MAGIC = b"FTR1"
_HEADER = struct.Struct(">4sII")
_PAGE_HEADER = struct.Struct(">4sIH")
_END_OF_CHAIN = 0xFFFFFFFF


class _PagedPostings(MutableMapping):
    """Term-to-postings mapping that reads each serialized list through the BM."""

    def __init__(
        self,
        index: InvertedIndex,
        term_pages: dict[str, tuple[int, int, int]],
    ):
        self._index = index
        self._term_pages = term_pages
        self._cache: dict[str, list[tuple[int, int]]] = {}
        self._dirty: set[str] = set()

    def __getitem__(self, term: str) -> list[tuple[int, int]]:
        if term in self._cache:
            return self._cache[term]
        descriptor = self._term_pages.get(term)
        if descriptor is None:
            raise KeyError(term)
        page_id, length, _ = descriptor
        postings = pickle.loads(self._index._read_chain(page_id, length))
        self._cache[term] = postings
        return postings

    def __setitem__(self, term: str, postings: list[tuple[int, int]]) -> None:
        self._cache[term] = postings
        self._dirty.add(term)

    def __delitem__(self, term: str) -> None:
        if term not in self._term_pages and term not in self._cache:
            raise KeyError(term)
        self._cache.pop(term, None)
        self._term_pages.pop(term, None)
        self._dirty.discard(term)

    def __iter__(self) -> Iterator[str]:
        return iter(dict.fromkeys((*self._term_pages, *self._cache)))

    def __len__(self) -> int:
        return len(set(self._term_pages) | set(self._cache))

    def document_frequency(self, term: str) -> int:
        postings = self._cache.get(term)
        if postings is not None:
            return len(postings)
        descriptor = self._term_pages.get(term)
        if descriptor is not None:
            return descriptor[2]
        return 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self._dirty.clear()


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
        manager: BufferManager | None = None,
    ):
        # term -> list of (doc_id, tf) sorted by doc_id
        self.index: dict[str, list[tuple[int, int]]] | _PagedPostings = index or {}
        # doc_id -> number of tokens in the document
        self.doc_lengths: dict[int, int] = doc_lengths or {}
        # Total number of indexed documents
        self.total_docs: int = total_docs
        # Cached document Euclidean norms: doc_id -> ||V(D)||
        self._doc_norms: dict[int, float] = {}
        # Physical storage binding for Catalog integration
        self.file_path: str | None = file_path
        self.bm: BufferManager | None = manager
        self._term_pages: dict[str, tuple[int, int, int]] = {}
        self._paged_format = False

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
        if isinstance(self.index, _PagedPostings):
            return self.index.document_frequency(term)
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
        k1: float = BM25_K1,
        b: float = BM25_B,
        top_k: int | None = 10,
    ) -> list[tuple[int, float]]:
        """
        Computes Okapi BM25 scores for candidates matching at least one query term.
        The sum runs over the distinct query terms (t in Q), so a repeated term
        counts once. Returns top_k tuples of (doc_id, score) sorted descending by
        score, or every candidate when top_k is None.
        """
        if not query_tokens or self.total_docs == 0:
            return []

        avgdl = self.avg_doc_length or 1.0
        scores: dict[int, float] = {}

        # Disjunctive scoring across distinct query terms
        for term in dict.fromkeys(query_tokens):
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
        if top_k is not None:
            return ranked[:top_k]
        return ranked

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

    def rank(
        self,
        query_tokens: Sequence[str],
        ranking: str = "bm25",
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Ranks the candidates of the query with the named algorithm:
        'bm25' (Okapi BM25) or 'tf_idf' (TF-IDF with cosine similarity).
        Both return (doc_id, score) sorted by (-score, doc_id).
        """
        if ranking == "bm25":
            return self.score_bm25(query_tokens, top_k=top_k)
        if ranking == "tf_idf":
            return self.score_tfidf_cosine(query_tokens, top_k=top_k)
        raise ValueError(f"Algoritmo de ranking no soportado: {ranking}")

    def bm25_search(
        self,
        query_tokens: Sequence[str],
        top_k: int | None = 10,
        k1: float = BM25_K1,
        b: float = BM25_B,
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
        """Persists metadata and postings in pages managed by the Buffer Manager."""
        target_path = file_path or self.file_path
        if not target_path:
            raise ValueError("No file_path provided for saving InvertedIndex")

        # TF-IDF needs norms for every document. Persist them so a query only
        # has to read postings for its terms.
        if not self._doc_norms and self.total_docs:
            self.compute_doc_norms()

        manager = self.bm
        if manager is not None and manager.file_manager.file_path != target_path:
            if isinstance(self.index, _PagedPostings):
                self.index = {term: list(self.index[term]) for term in self.index}
            self._term_pages = {}
            self._paged_format = False
            manager = None
        if manager is None:
            manager = BufferManager(FileManager(target_path))
            self.bm = manager

        if not self._paged_format:
            has_paged_file = False
            if manager.file_manager.page_count():
                with open(target_path, "rb") as handle:
                    has_paged_file = handle.read(len(_INDEX_MAGIC)) == _INDEX_MAGIC
            if not has_paged_file:
                manager.reset()
                if manager.allocate_page() != 0:
                    raise RuntimeError("El encabezado del índice debe estar en la página 0")
            self._term_pages = {}
            self._paged_format = True
        dirty_terms = (
            list(self.index._dirty)
            if isinstance(self.index, _PagedPostings)
            else list(self.index)
        )

        if isinstance(self.index, _PagedPostings):
            for term in dirty_terms:
                if term not in self.index:
                    continue
                postings = self.index[term]
                payload = pickle.dumps(postings, protocol=5)
                first_page = self._write_chain(payload)
                self._term_pages[term] = (first_page, len(payload), len(postings))
        else:
            for term in dirty_terms:
                postings = self.index[term]
                payload = pickle.dumps(postings, protocol=5)
                first_page = self._write_chain(payload)
                self._term_pages[term] = (first_page, len(payload), len(postings))

        metadata = pickle.dumps({
            "version": 1,
            "term_pages": self._term_pages,
            "doc_lengths": self.doc_lengths,
            "total_docs": self.total_docs,
            "doc_norms": self._doc_norms,
        }, protocol=5)
        metadata_page = self._write_chain(metadata)
        manager.flush_all()

        page = manager.fetch_page(0)
        try:
            page.clear()
            page.write_bytes(0, _HEADER.pack(_INDEX_MAGIC, metadata_page, len(metadata)))
        finally:
            manager.unpin_page(0, is_dirty=True)
        manager.flush_page(0)

        self.file_path = target_path
        if isinstance(self.index, _PagedPostings):
            self.index.clear_cache()
        else:
            self.index = _PagedPostings(self, self._term_pages)

    @classmethod
    def load(cls, file_path: str, manager: BufferManager | None = None) -> InvertedIndex:
        """Loads paged indexes lazily, or upgrades a legacy pickle on next save."""
        if manager is None:
            manager = BufferManager(FileManager(file_path))
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return cls(file_path=file_path, manager=manager)

        with open(file_path, "rb") as handle:
            magic = handle.read(len(_INDEX_MAGIC))
        if magic != _INDEX_MAGIC:
            with open(file_path, "rb") as handle:
                state = pickle.load(handle)
            idx = cls(
                index=state["index"],
                doc_lengths=state["doc_lengths"],
                total_docs=state["total_docs"],
                file_path=file_path,
                manager=manager,
            )
            idx._doc_norms = state.get("doc_norms", {})
            return idx

        idx = cls(file_path=file_path, manager=manager)
        header_page = manager.fetch_page(0)
        try:
            magic, metadata_page, metadata_length = _HEADER.unpack(
                header_page.read_bytes(0, _HEADER.size)
            )
        finally:
            manager.unpin_page(0)
        if magic != _INDEX_MAGIC:
            raise ValueError(f"Formato de índice invertido inválido: {file_path}")
        metadata = pickle.loads(idx._read_chain(metadata_page, metadata_length))
        if metadata.get("version") != 1:
            raise ValueError(f"Versión de índice invertido no soportada: {metadata.get('version')}")
        idx._term_pages = metadata["term_pages"]
        idx.doc_lengths = metadata["doc_lengths"]
        idx.total_docs = metadata["total_docs"]
        idx._doc_norms = metadata.get("doc_norms", {})
        idx.index = _PagedPostings(idx, idx._term_pages)
        idx._paged_format = True
        return idx

    def _write_chain(self, payload: bytes) -> int:
        manager = self.bm
        if manager is None:
            raise RuntimeError("No hay Buffer Manager para escribir el índice")
        page_size = manager.file_manager.page_size
        chunk_size = page_size - _PAGE_HEADER.size
        page_count = max(1, (len(payload) + chunk_size - 1) // chunk_size)
        page_ids = [manager.allocate_page() for _ in range(page_count)]
        for offset, page_id in enumerate(page_ids):
            chunk = payload[offset * chunk_size:(offset + 1) * chunk_size]
            next_page = page_ids[offset + 1] if offset + 1 < page_count else _END_OF_CHAIN
            page = manager.fetch_page(page_id)
            try:
                page.clear()
                page.write_bytes(
                    0,
                    _PAGE_HEADER.pack(_PAGE_MAGIC, next_page, len(chunk)) + chunk,
                )
            finally:
                manager.unpin_page(page_id, is_dirty=True)
        return page_ids[0]

    def _read_chain(self, first_page: int, length: int) -> bytes:
        if self.bm is None:
            raise RuntimeError("No hay Buffer Manager para leer el índice")
        result = bytearray()
        page_id = first_page
        visited: set[int] = set()
        while len(result) < length:
            if page_id in visited:
                raise ValueError("La cadena de páginas de índice invertido contiene un ciclo")
            visited.add(page_id)
            page = self.bm.fetch_page(page_id)
            try:
                magic, next_page, chunk_length = _PAGE_HEADER.unpack(
                    page.read_bytes(0, _PAGE_HEADER.size)
                )
                if magic != _PAGE_MAGIC or chunk_length > page.size - _PAGE_HEADER.size:
                    raise ValueError(f"Página de índice invertido inválida: {page_id}")
                result.extend(page.read_bytes(_PAGE_HEADER.size, chunk_length))
            finally:
                self.bm.unpin_page(page_id)
            if len(result) < length:
                if next_page == _END_OF_CHAIN or next_page == page_id:
                    raise ValueError("Cadena de páginas de índice invertido truncada")
                page_id = next_page
        if len(result) != length:
            raise ValueError("Cadena de páginas de índice invertido excede su longitud")
        return bytes(result)

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
            postings = list(self.index.get(term, []))
            postings.append((doc_id, tf))
            postings.sort(key=lambda x: x[0])
            self.index[term] = postings

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
        for term in list(self.index):
            postings = self.index[term]
            new_postings = [p for p in postings if p[0] != doc_id]
            if not new_postings:
                empty_terms.append(term)
            else:
                self.index[term] = new_postings
        for term in empty_terms:
            del self.index[term]
