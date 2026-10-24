"""
Shared text processing and normalization component.

Guarantees identical token generation rules for both document indexing
and query execution according to the text search specification contract.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable

from .stopwords import (
    ENGLISH_STOPWORDS,
    SPANISH_STOPWORDS,
    load_stopwords,
)


class TextProcessor:
    """
    Component for text normalization, tokenization, stopword filtering,
    and optional stemming.
    """

    def __init__(
        self,
        strip_accents: bool = True,
        lowercase: bool = True,
        stopwords: Iterable[str] | str | None = None,
        stemmer: Callable[[str], str] | None = None,
    ):
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.stemmer = stemmer
        # Regex to match alphanumeric tokens without underscores (unicode-aware)
        self._token_pattern = re.compile(r"[^\W_]+", re.UNICODE)

        # Normalize stopwords with the exact same rules if provided
        if isinstance(stopwords, str):
            self.stopwords: set[str] = set(load_stopwords(stopwords))
        elif stopwords is not None:
            self.stopwords: set[str] = {self.normalize(w) for w in stopwords if w}
        else:
            self.stopwords = set()

    def normalize(self, text: str | None) -> str:
        """
        Normalizes Unicode, removes diacritics/accents, and converts to lowercase.
        Handles None or empty strings gracefully.
        """
        if not text:
            return ""

        # 1. Canonical compatibility decomposition
        norm_text = unicodedata.normalize("NFKD", text)

        # 2. Strip diacritical marks if requested
        if self.strip_accents:
            norm_text = "".join(
                c for c in norm_text if not unicodedata.combining(c)
            )

        # 3. Lowercase
        if self.lowercase:
            norm_text = norm_text.lower()

        return norm_text

    def tokenize(self, text: str | None) -> list[str]:
        """
        Extracts alphanumeric tokens from text, treating punctuation and symbols
        as delimiters.
        """
        if not text:
            return []
        return self._token_pattern.findall(text)

    def process(self, text: str | None) -> list[str]:
        """
        Full text processing pipeline:
        1. Normalization (Unicode NFKD, accents, case folding)
        2. Tokenization
        3. Stopwords filtering
        4. Optional stemming
        
        Guaranteed to produce identical results for documents and queries.
        """
        if not text:
            return []

        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)

        # Filter stopwords
        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Apply optional stemmer
        if self.stemmer is not None:
            tokens = [self.stemmer(t) for t in tokens]

        return tokens
