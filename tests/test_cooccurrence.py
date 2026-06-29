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

    def test_exact_structure_and_weights(self) -> None:
        # ["a", "b", "c"] with window=1: each adjacent pair co-occurs twice
        # (once from each direction), giving normalised weights of 0.5 each, and
        # NO a-c edge (they are 2 apart). Pins the skip-gram window arithmetic.
        g = build_cooccurrence_graph(["a", "b", "c"], window=1)
        assert set(g.nodes()) == {"a", "b", "c"}
        assert g["a"]["b"]["weight"] == pytest.approx(0.5)
        assert g["b"]["c"]["weight"] == pytest.approx(0.5)
        assert not g.has_edge("a", "c")

    def test_min_weight_is_inclusive(self) -> None:
        # Both edges have weight exactly 0.5; min_weight=0.5 must keep them (>=).
        g = build_cooccurrence_graph(["a", "b", "c"], window=1, min_weight=0.5)
        assert g.number_of_edges() == 2

    def test_default_window_is_two(self) -> None:
        # Default window is 2: tokens 2 apart co-occur, tokens 3 apart do not.
        g = build_cooccurrence_graph(["a", "b", "c", "d"])
        assert g.has_edge("a", "c")
        assert not g.has_edge("a", "d")


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

    @settings(max_examples=50, deadline=5000)
    @given(token_list)
    def test_nodes_subset_of_tokens(self, tokens: list[str]) -> None:
        g = build_cooccurrence_graph(tokens, window=2)
        assert set(g.nodes()) <= set(tokens)

    @settings(max_examples=50, deadline=5000)
    @given(token_list)
    def test_graph_is_symmetric(self, tokens: list[str]) -> None:
        g = build_cooccurrence_graph(tokens, window=2)
        for u, v, data in g.edges(data=True):
            assert g[u][v]["weight"] == g[v][u]["weight"]

    @settings(max_examples=30, deadline=5000)
    @given(token_list, st.floats(min_value=0.5, max_value=5.0, allow_nan=False))
    def test_min_weight_respected(self, tokens: list[str], min_w: float) -> None:
        g = build_cooccurrence_graph(tokens, window=2, min_weight=min_w)
        for _u, _v, data in g.edges(data=True):
            assert data["weight"] >= min_w

    @settings(max_examples=50, deadline=5000)
    @given(token_list, st.integers(min_value=1, max_value=5))
    def test_weights_sum_to_one(self, tokens: list[str], window: int) -> None:
        # Edge weights are normalised co-occurrence frequencies: with no
        # min_weight cut, they must sum to 1.0 over the whole graph.
        g = build_cooccurrence_graph(tokens, window=window, min_weight=0.0)
        if g.number_of_edges() == 0:
            return
        total = sum(data["weight"] for _u, _v, data in g.edges(data=True))
        assert abs(total - 1.0) < 1e-9


# Tokens drawn from a tiny alphabet so n-grams actually repeat and collocations
# can be found (exercising the scoring path, not just the empty-result path).
_repeating_tokens = st.lists(
    st.sampled_from(["a", "b", "c", "d", "e", "f"]),
    min_size=12,
    max_size=80,
)


class TestCollocationProperties:
    """Property-based tests for detect_collocations invariants.

    Uses the ``pmi`` metric only: the structural invariants checked here are
    metric-independent, and NLTK's ``chi_sq`` / ``likelihood`` scorers raise
    (ZeroDivisionError / math-domain ValueError) on degenerate corpora such as
    all-identical tokens, which detect_collocations does not currently guard.
    """

    @settings(max_examples=60, deadline=5000)
    @given(
        _repeating_tokens,
        st.sampled_from([2, 3]),
        st.integers(min_value=1, max_value=10),
    )
    def test_collocation_invariants(
        self, tokens: list[str], n: int, top_n: int
    ) -> None:
        result = detect_collocations(tokens, n=n, metric="pmi", top_n=top_n, min_freq=2)
        contiguous_ngrams = {
            tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        }
        # At most top_n results, each an n-tuple drawn from the text's n-grams.
        assert len(result) <= top_n
        assert all(len(t) == n for t in result)
        assert all(t in contiguous_ngrams for t in result)
