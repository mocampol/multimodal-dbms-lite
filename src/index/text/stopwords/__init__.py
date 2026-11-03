"""
Official PostgreSQL Full Text Search stopwords.

Exposes:
- load_stopwords: function to load and cache official .stop dictionaries.
- SPANISH_STOPWORDS: frozenset containing normalized terms from spanish.stop.
- ENGLISH_STOPWORDS: frozenset containing normalized terms from english.stop.
"""

from .loader import (
    ENGLISH_STOPWORDS,
    SPANISH_STOPWORDS,
    load_stopwords,
)

__all__ = [
    "load_stopwords",
    "SPANISH_STOPWORDS",
    "ENGLISH_STOPWORDS",
]
