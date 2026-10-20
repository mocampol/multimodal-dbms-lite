"""Text normalization shared by MATCH filtering and SCORE() ranking."""

import re
import unicodedata

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text) -> list[str]:
    """Lowercases, strips accents and splits into alphanumeric terms.

    None (SQL NULL) yields no terms, so a NULL document never matches and
    scores 0.0. 'Introducción' and 'introduccion' produce the same term.
    """
    if text is None:
        return []
    decomposed = unicodedata.normalize("NFKD", str(text).lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WORD.findall(stripped)