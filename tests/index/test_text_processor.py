"""
Unit tests for the shared TextProcessor component (Issue 1).

Validates:
- Handling of Unicode, punctuation, empty text, and accented words.
- Edge cases (None, only punctuation, whitespace, numbers, symbols).
- Query-document parity invariant (same text yields same terms).
- Stopwords filtering and optional stemming.
"""

import pytest
from index.text.text_processor import (
    TextProcessor,
    SPANISH_STOPWORDS,
    ENGLISH_STOPWORDS,
)


def test_empty_and_none_text():
    """Validates that empty inputs or None do not raise errors and return empty lists."""
    processor = TextProcessor()
    assert processor.process(None) == []
    assert processor.process("") == []
    assert processor.process("   \t\n  ") == []
    assert processor.normalize(None) == ""
    assert processor.normalize("") == ""
    assert processor.tokenize("") == []


def test_punctuation_and_symbols_only():
    """Punctuation, symbols, and whitespace are treated as delimiters."""
    processor = TextProcessor()
    assert processor.process("¡!¿?,.;:-_(){}[]'\"/\\@#$%^&*") == []
    assert processor.process("   ...   ---   ") == []


def test_accents_and_diacritics_spanish():
    """Accents and diacritics are removed while preserving the base characters."""
    processor = TextProcessor(strip_accents=True)
    text = "¡La canción del pingüino en Bogotá!"
    tokens = processor.process(text)
    assert tokens == ["la", "cancion", "del", "pinguino", "en", "bogota"]


def test_unicode_normalization():
    """Composite and decomposed Unicode characters produce identical tokens."""
    processor = TextProcessor()
    # Form precomposed 'é' (\u00e9) vs decomposed 'e' + combining acute (\u0065\u0301)
    precomposed = "café"
    decomposed = "cafe\u0301"
    assert processor.process(precomposed) == ["cafe"]
    assert processor.process(decomposed) == ["cafe"]
    assert processor.process(precomposed) == processor.process(decomposed)


def test_query_document_parity_invariant():
    """
    CRITICAL ACCEPTANCE CRITERION:
    The same text must produce identical terms when indexing a document
    and when searching as a query.
    """
    processor = TextProcessor()
    document_text = "Base de Datos Relacional y Multimodal 2026."
    query_text = "Base de Datos Relacional y Multimodal 2026."

    doc_tokens = processor.process(document_text)
    query_tokens = processor.process(query_text)

    assert doc_tokens == query_tokens
    assert doc_tokens == ["base", "de", "datos", "relacional", "y", "multimodal", "2026"]


def test_case_folding():
    """Upper, lower, and mixed case strings produce identical lowercased tokens."""
    processor = TextProcessor()
    assert processor.process("POSTGRESQL") == ["postgresql"]
    assert processor.process("PostgreSql") == ["postgresql"]
    assert processor.process("postgresql") == ["postgresql"]


def test_english_text():
    """English sentences with punctuation, contractions, and numbers."""
    processor = TextProcessor()
    text = "The QUICK brown fox jumps over 12 lazy dogs!"
    tokens = processor.process(text)
    assert tokens == ["the", "quick", "brown", "fox", "jumps", "over", "12", "lazy", "dogs"]


def test_stopwords_filtering_spanish():
    """Stopwords in Spanish are stripped when enabled."""
    processor = TextProcessor(stopwords=SPANISH_STOPWORDS)
    text = "La base de datos de los estudiantes con un buen promedio"
    tokens = processor.process(text)
    assert "la" not in tokens
    assert "de" not in tokens
    assert "los" not in tokens
    assert "con" not in tokens
    assert "un" not in tokens
    assert tokens == ["base", "datos", "estudiantes", "buen", "promedio"]


def test_stopwords_filtering_english():
    """Stopwords in English are stripped when enabled."""
    processor = TextProcessor(stopwords=ENGLISH_STOPWORDS)
    text = "The query processing in a distributed system"
    tokens = processor.process(text)
    assert tokens == ["query", "processing", "distributed", "system"]


def test_optional_stemmer():
    """A custom stemmer function can be attached and executed in the pipeline."""
    # Simple test suffix stemmer
    def dummy_stemmer(word: str) -> str:
        if word.endswith("ing"):
            return word[:-3]
        if word.endswith("es"):
            return word[:-2]
        return word

    processor = TextProcessor(stemmer=dummy_stemmer)
    tokens = processor.process("Testing foxes walking")
    assert tokens == ["test", "fox", "walk"]


def test_load_stopwords_postgres_official():
    """Validates loading official PostgreSQL stopwords files directly."""
    from index.text.stopwords import load_stopwords

    # Spanish dictionary (~313 words from PostgreSQL spanish.stop)
    es_stops = load_stopwords("spanish")
    assert len(es_stops) >= 300
    # Must include normalized forms of words with accents
    assert "mas" in es_stops
    assert "tambien" in es_stops
    assert "habia" in es_stops
    assert "estuvimos" in es_stops
    assert "ellos" in es_stops

    # English dictionary (~127 words from PostgreSQL english.stop)
    en_stops = load_stopwords("english")
    assert len(en_stops) >= 100
    assert "the" in en_stops
    assert "myself" in en_stops
    assert "ourselves" in en_stops

    # Unsupported language
    with pytest.raises(ValueError, match="Unsupported language"):
        load_stopwords("german")


def test_text_processor_stopwords_string_alias():
    """TextProcessor accepts string language names ('spanish', 'english')."""
    processor_es = TextProcessor(stopwords="spanish")
    tokens_es = processor_es.process("Él había corrido más que ellos")
    # 'el', 'habia', 'mas', 'que', 'ellos' are all PostgreSQL stopwords
    assert tokens_es == ["corrido"]

    processor_en = TextProcessor(stopwords="english")
    tokens_en = processor_en.process("They were doing tests by themselves")
    assert tokens_en == ["tests"]


def test_custom_stopwords_with_accents_and_case():
    """Custom stopwords with mixed case and accents are auto-normalized."""
    processor = TextProcessor(stopwords=["MÁS", "Él", "INFORMACIÓN"])
    tokens = processor.process("Más información para él y todos")
    assert "mas" not in tokens
    assert "el" not in tokens
    assert "informacion" not in tokens
    assert tokens == ["para", "y", "todos"]

