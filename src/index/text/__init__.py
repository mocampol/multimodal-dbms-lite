"""Text search index module."""

from .text_processor import (
    ENGLISH_STOPWORDS,
    SPANISH_STOPWORDS,
    TextProcessor,
    load_stopwords,
)
from .inverted_index import InvertedIndex, doc_id_to_rid, rid_to_doc_id
from .spimi import SPIMIIndexer, SPIMIStats

__all__ = [
    "TextProcessor",
    "SPANISH_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "load_stopwords",
    "InvertedIndex",
    "SPIMIIndexer",
    "SPIMIStats",
    "rid_to_doc_id",
    "doc_id_to_rid",
]
