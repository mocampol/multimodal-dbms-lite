"""
Stopwords loader for PostgreSQL Full Text Search dictionary files.

Sources:
- PostgreSQL official repository: src/backend/snowball/stopwords/spanish.stop
  https://raw.githubusercontent.com/postgres/postgres/master/src/backend/snowball/stopwords/spanish.stop
- PostgreSQL official repository: src/backend/snowball/stopwords/english.stop
  https://raw.githubusercontent.com/postgres/postgres/master/src/backend/snowball/stopwords/english.stop
"""

from __future__ import annotations

import functools
from pathlib import Path
import unicodedata

_STOPWORDS_DIR = Path(__file__).resolve().parent

LANGUAGE_FILE_MAP: dict[str, str] = {
    "spanish": "spanish.stop",
    "es": "spanish.stop",
    "espanol": "spanish.stop",
    "español": "spanish.stop",
    "english": "english.stop",
    "en": "english.stop",
    "ingles": "english.stop",
    "inglés": "english.stop",
}


def _normalize_token(token: str) -> str:
    """Normalizes token to lowercase NFKD with combining diacritics stripped."""
    nfkd = unicodedata.normalize("NFKD", token.strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@functools.lru_cache(maxsize=16)
def load_stopwords(language: str = "spanish") -> frozenset[str]:
    """
    Loads official PostgreSQL stopword dictionary for the specified language.
    Results are normalized (NFKD, accent-stripped, lowercase) and cached in memory.

    :param language: Language name or code ('spanish', 'english', 'es', 'en').
    :return: Immutable frozenset of normalized stopwords.
    :raises ValueError: If the requested language is unsupported.
    :raises FileNotFoundError: If the .stop file is missing from the package.
    """
    key = language.strip().lower()
    filename = LANGUAGE_FILE_MAP.get(key)
    if not filename:
        valid_options = ", ".join(sorted(set(LANGUAGE_FILE_MAP.values())))
        raise ValueError(
            f"Unsupported language '{language}' for stopwords. Available dictionaries: {valid_options}"
        )

    filepath = _STOPWORDS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"PostgreSQL stopword file not found: {filepath}. Ensure '{filename}' is present in { _STOPWORDS_DIR}."
        )

    words: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments
            if not line or line.startswith("#") or line.startswith("|"):
                continue
            cleaned = _normalize_token(line)
            if cleaned:
                words.add(cleaned)

    return frozenset(words)


# Pre-cached module-level constants for direct import
SPANISH_STOPWORDS: frozenset[str] = load_stopwords("spanish")
ENGLISH_STOPWORDS: frozenset[str] = load_stopwords("english")
