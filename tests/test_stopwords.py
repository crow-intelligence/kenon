"""Tests for kenon.stopwords."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kenon.stopwords import get_stopwords


class TestGetStopwords:
    """Unit tests for get_stopwords."""

    def test_english_default(self) -> None:
        sw = get_stopwords("english")
        assert isinstance(sw, frozenset)
        assert "the" in sw
        assert "a" in sw

    def test_extra_words(self) -> None:
        sw = get_stopwords("english", extra=["gonna", "wanna"])
        assert "gonna" in sw
        assert "wanna" in sw

    def test_nltk_only(self) -> None:
        sw = get_stopwords("english", sources=["nltk"])
        assert isinstance(sw, frozenset)
        assert "the" in sw

    def test_sklearn_only(self) -> None:
        sw = get_stopwords("english", sources=["sklearn"])
        assert isinstance(sw, frozenset)
        assert len(sw) > 0

    def test_sklearn_non_english_raises(self) -> None:
        with pytest.raises(ValueError, match="only available for English"):
            get_stopwords("german", sources=["sklearn"])

    def test_unsupported_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            get_stopwords("english", sources=["bogus"])

    def test_union_superset(self) -> None:
        nltk_only = get_stopwords("english", sources=["nltk"])
        sklearn_only = get_stopwords("english", sources=["sklearn"])
        both = get_stopwords("english", sources=["nltk", "sklearn"])
        assert both >= nltk_only
        assert both >= sklearn_only

    def test_all_lowercase(self) -> None:
        sw = get_stopwords("english")
        for word in sw:
            assert word == word.lower()


class TestStopwordsProperties:
    """Property-based tests for get_stopwords."""

    @settings(max_examples=50)
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    def test_extra_always_present(self, extra: list[str]) -> None:
        sw = get_stopwords("english", extra=extra)
        for word in extra:
            assert word.lower() in sw

    @settings(max_examples=50)
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    def test_always_frozenset(self, extra: list[str]) -> None:
        sw = get_stopwords("english", extra=extra)
        assert isinstance(sw, frozenset)

    @settings(max_examples=50)
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    def test_always_lowercase(self, extra: list[str]) -> None:
        sw = get_stopwords("english", extra=extra)
        for word in sw:
            assert word == word.lower()
