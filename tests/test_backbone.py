"""Tests for kenon.backbone."""

import copy

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

from kenon.backbone import (
    apply_disparity_filter,
    disparity_integral,
    extract_backbone,
    get_disparity_significance,
)


def _make_weighted_graph(n_nodes: int = 6) -> nx.Graph:
    """Create a small weighted graph for testing."""
    g = nx.Graph()
    edges = [
        ("a", "b", 0.8),
        ("b", "c", 0.3),
        ("a", "c", 0.5),
        ("c", "d", 0.9),
        ("d", "e", 0.2),
        ("e", "f", 0.7),
        ("a", "d", 0.4),
        ("b", "e", 0.6),
    ]
    for u, v, w in edges[:n_nodes]:
        g.add_edge(u, v, weight=w)
    return g


class TestDisparityIntegral:
    """Unit tests for disparity_integral."""

    def test_basic_value(self) -> None:
        val = disparity_integral(0.5, 3.0)
        assert isinstance(val, float)

    def test_different_inputs(self) -> None:
        v1 = disparity_integral(0.3, 4.0)
        v2 = disparity_integral(0.7, 4.0)
        assert v1 != v2


class TestGetDisparitySignificance:
    """Unit tests for get_disparity_significance."""

    def test_returns_float(self) -> None:
        alpha = get_disparity_significance(0.5, 3.0)
        assert isinstance(alpha, float)

    def test_degree_one_returns_zero(self) -> None:
        alpha = get_disparity_significance(0.5, 1.0)
        assert alpha == 0.0

    def test_in_range(self) -> None:
        alpha = get_disparity_significance(0.5, 3.0)
        assert 0.0 <= alpha <= 1.0


class TestApplyDisparityFilter:
    """Unit tests for apply_disparity_filter."""

    def test_adds_attributes(self) -> None:
        g = _make_weighted_graph()
        alphas = apply_disparity_filter(g)
        assert len(alphas) == g.number_of_edges()
        for _u, _v, data in g.edges(data=True):
            assert "norm_weight" in data
            assert "alpha" in data
            assert "alpha_ptile" in data
        for node in g.nodes():
            assert "strength" in g.nodes[node]

    def test_empty_graph(self) -> None:
        g = nx.Graph()
        alphas = apply_disparity_filter(g)
        assert alphas == []


class TestExtractBackbone:
    """Unit tests for extract_backbone."""

    def test_does_not_mutate_original(self) -> None:
        g = _make_weighted_graph()
        original_edges = set(g.edges())
        original_nodes = set(g.nodes())
        _ = extract_backbone(g, min_alpha_ptile=0.3)
        assert set(g.edges()) == original_edges
        assert set(g.nodes()) == original_nodes

    def test_reduces_graph(self) -> None:
        g = _make_weighted_graph()
        backbone = extract_backbone(g, min_alpha_ptile=0.5)
        assert backbone.number_of_nodes() <= g.number_of_nodes()
        assert backbone.number_of_edges() <= g.number_of_edges()

    def test_empty_graph(self) -> None:
        g = nx.Graph()
        backbone = extract_backbone(g)
        assert backbone.number_of_nodes() == 0

    def test_min_degree_respected(self) -> None:
        g = _make_weighted_graph()
        backbone = extract_backbone(g, min_alpha_ptile=0.1, min_degree=2)
        for node in backbone.nodes():
            assert backbone.degree(node) >= 2


class TestBackboneProperties:
    """Property-based tests for backbone module."""

    @settings(max_examples=30, deadline=5000)
    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_never_increases_edges(self, ptile: float) -> None:
        g = _make_weighted_graph()
        backbone = extract_backbone(g, min_alpha_ptile=ptile)
        assert backbone.number_of_edges() <= g.number_of_edges()

    @settings(max_examples=30, deadline=5000)
    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_never_increases_nodes(self, ptile: float) -> None:
        g = _make_weighted_graph()
        backbone = extract_backbone(g, min_alpha_ptile=ptile)
        assert backbone.number_of_nodes() <= g.number_of_nodes()

    @settings(max_examples=30, deadline=5000)
    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_original_not_mutated(self, ptile: float) -> None:
        g = _make_weighted_graph()
        g_copy = copy.deepcopy(g)
        _ = extract_backbone(g, min_alpha_ptile=ptile)
        assert set(g.edges()) == set(g_copy.edges())
        assert set(g.nodes()) == set(g_copy.nodes())
        for u, v, data in g.edges(data=True):
            assert data["weight"] == g_copy[u][v]["weight"]

    @settings(max_examples=20, deadline=5000)
    @given(st.integers(min_value=1, max_value=5))
    def test_min_degree_invariant(self, min_deg: int) -> None:
        g = _make_weighted_graph()
        backbone = extract_backbone(g, min_alpha_ptile=0.3, min_degree=min_deg)
        for node in backbone.nodes():
            assert backbone.degree(node) >= min_deg
