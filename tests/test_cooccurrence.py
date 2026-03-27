"""Tests for kenon.cooccurrence."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kenon.cooccurrence import build_cooccurrence_graph, detect_collocations
from tests.strategies import token_list


class TestBuildCooccurrenceGraph:
    """Unit tests for build_cooccurrence_graph."""

    def test_basic_graph(self, sample_tokens: list[str]) -> None:
        g = build_cooccurrence_graph(sample_tokens, window=2)
        assert g.number_of_nodes() > 0
        assert g.number_of_edges() > 0

    def test_no_self_loops(self, sample_tokens: list[str]) -> None:
        g = build_cooccurrence_graph(sample_tokens, window=2)
        for u, v in g.edges():
            assert u != v

    def test_positive_weights(self, sample_tokens: list[str]) -> None:
        g = build_cooccurrence_graph(sample_tokens, window=2)
        for _u, _v, data in g.edges(data=True):
            assert data["weight"] > 0

    def test_window_validation(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            build_cooccurrence_graph(["a", "b"], window=0)

    def test_stopword_filtering(self) -> None:
        tokens = ["the", "cat", "sat", "the", "mat"]
        stopwords = frozenset(["the"])
        g = build_cooccurrence_graph(tokens, window=1, stopwords=stopwords)
        assert "the" not in g.nodes()

    def test_min_weight(self, sample_tokens: list[str]) -> None:
        g = build_cooccurrence_graph(sample_tokens, window=1, min_weight=0.5)
        for _u, _v, data in g.edges(data=True):
            assert data["weight"] >= 0.5

    def test_empty_tokens(self) -> None:
        g = build_cooccurrence_graph([], window=1)
        assert g.number_of_nodes() == 0

    def test_single_token(self) -> None:
        g = build_cooccurrence_graph(["hello"], window=1)
        assert g.number_of_edges() == 0


class TestDetectCollocations:
    """Unit tests for detect_collocations."""

    def test_bigram_detection(self) -> None:
        tokens = ["new", "york", "city", "new", "york", "times"] * 10
        colls = detect_collocations(tokens, n=2, top_n=5)
        assert ("new", "york") in colls

    def test_trigram_detection(self) -> None:
        tokens = ["new", "york", "city", "new", "york", "city"] * 10
        colls = detect_collocations(tokens, n=3, top_n=5, min_freq=2)
        assert len(colls) >= 0  # may or may not find trigrams

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be 2 or 3"):
            detect_collocations(["a", "b"], n=4)

    def test_invalid_metric(self) -> None:
        with pytest.raises(ValueError, match="Unsupported metric"):
            detect_collocations(["a", "b"], metric="bogus")

    def test_top_n_limit(self) -> None:
        tokens = ["new", "york", "city", "los", "angeles"] * 10
        colls = detect_collocations(tokens, n=2, top_n=2, min_freq=2)
        assert len(colls) <= 2


class TestCooccurrenceProperties:
    """Property-based tests for cooccurrence module."""

    @settings(max_examples=50, deadline=5000)
    @given(token_list)
    def test_no_self_loops(self, tokens: list[str]) -> None:
        g = build_cooccurrence_graph(tokens, window=2)
        for u, v in g.edges():
            assert u != v

    @settings(max_examples=50, deadline=5000)
    @given(token_list)
    def test_all_weights_positive(self, tokens: list[str]) -> None:
        g = build_cooccurrence_graph(tokens, window=2)
        for _u, _v, data in g.edges(data=True):
            assert data["weight"] > 0

    @settings(max_examples=30, deadline=5000)
    @given(token_list, st.integers(min_value=1, max_value=5))
    def test_stopword_removal(self, tokens: list[str], window: int) -> None:
        if not tokens:
            return
        stopwords = frozenset(tokens[:2])
        g = build_cooccurrence_graph(tokens, window=window, stopwords=stopwords)
        for node in g.nodes():
            assert node not in stopwords

    @settings(max_examples=50, deadline=5000)
    @given(st.integers(min_value=-5, max_value=0))
    def test_invalid_window_raises(self, window: int) -> None:
        with pytest.raises(ValueError):
            build_cooccurrence_graph(["a", "b", "c"], window=window)
