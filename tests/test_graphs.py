"""Tests for kenon.graphs."""

import tempfile

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings

from kenon.embeddings import CountVectorizerEmbedder, TfidfEmbedder
from kenon.graphs import (
    build_semantic_graph,
    cosine_similarity_matrix,
    load_graph,
    save_graph,
)
from tests.strategies import similarity_threshold, small_corpus


class TestBuildSemanticGraph:
    """Unit tests for build_semantic_graph."""

    def test_basic_graph(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        g = build_semantic_graph(emb, sample_corpus, similarity_threshold=0.1)
        assert isinstance(g, nx.Graph)

    def test_invalid_threshold(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        with pytest.raises(ValueError, match="similarity_threshold"):
            build_semantic_graph(emb, sample_corpus, similarity_threshold=1.5)
        with pytest.raises(ValueError, match="similarity_threshold"):
            build_semantic_graph(emb, sample_corpus, similarity_threshold=-0.1)

    def test_threshold_one_no_edges(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        g = build_semantic_graph(emb, sample_corpus, similarity_threshold=1.0)
        assert g.number_of_edges() == 0

    def test_no_self_loops(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        g = build_semantic_graph(emb, sample_corpus, similarity_threshold=0.0)
        for u, v in g.edges():
            assert u != v

    def test_edge_weights_in_range(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        g = build_semantic_graph(emb, sample_corpus, similarity_threshold=0.1)
        for _u, _v, data in g.edges(data=True):
            assert 0.0 <= data["weight"] <= 1.0

    def test_stopword_filtering(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        stopwords = frozenset(["the", "on", "in"])
        g = build_semantic_graph(
            emb, sample_corpus, similarity_threshold=0.1, stopwords=stopwords
        )
        for node in g.nodes():
            assert node not in stopwords

    def test_with_k_neighbors(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        g = build_semantic_graph(
            emb, sample_corpus, similarity_threshold=0.05, k_neighbors=3
        )
        assert isinstance(g, nx.Graph)


class TestCosineSimilarityMatrix:
    """Unit tests for cosine_similarity_matrix."""

    def test_shape(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        sim, vocab = cosine_similarity_matrix(emb, sample_corpus)
        assert sim.shape[0] == sim.shape[1] == len(vocab)

    def test_symmetric(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        sim, _ = cosine_similarity_matrix(emb, sample_corpus)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)


class TestSaveLoadGraph:
    """Unit tests for save_graph and load_graph."""

    def test_graphml_roundtrip(self) -> None:
        g = nx.Graph()
        g.add_edge("a", "b", weight=0.5)
        g.add_edge("b", "c", weight=0.3)
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            save_graph(g, f.name, fmt="graphml")
            loaded = load_graph(f.name, fmt="graphml")
        assert set(loaded.nodes()) == {"a", "b", "c"}
        assert abs(loaded["a"]["b"]["weight"] - 0.5) < 1e-10

    def test_pickle_roundtrip(self) -> None:
        g = nx.Graph()
        g.add_edge("x", "y", weight=0.9)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            save_graph(g, f.name, fmt="pickle")
            loaded = load_graph(f.name, fmt="pickle")
        assert loaded["x"]["y"]["weight"] == 0.9

    def test_gml_roundtrip(self) -> None:
        g = nx.Graph()
        g.add_edge("a", "b", weight=0.7)
        with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as f:
            save_graph(g, f.name, fmt="gml")
            loaded = load_graph(f.name, fmt="gml")
        assert abs(loaded["a"]["b"]["weight"] - 0.7) < 1e-10

    def test_unsupported_format(self) -> None:
        g = nx.Graph()
        with pytest.raises(ValueError, match="Unsupported format"):
            save_graph(g, "/tmp/test.xyz", fmt="xyz")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_graph("/tmp/test.xyz", fmt="xyz")


class TestGraphProperties:
    """Property-based tests for graphs module."""

    @settings(max_examples=15, deadline=10000)
    @given(small_corpus, similarity_threshold)
    def test_no_self_loops(self, corpus: list[str], threshold: float) -> None:
        emb = CountVectorizerEmbedder()
        g = build_semantic_graph(emb, corpus, similarity_threshold=threshold)
        for u, v in g.edges():
            assert u != v

    @settings(max_examples=15, deadline=10000)
    @given(small_corpus, similarity_threshold)
    def test_weights_in_range(self, corpus: list[str], threshold: float) -> None:
        emb = CountVectorizerEmbedder()
        g = build_semantic_graph(emb, corpus, similarity_threshold=threshold)
        for _u, _v, data in g.edges(data=True):
            assert -0.01 <= data["weight"] <= 1.01  # small tolerance

    @settings(max_examples=10, deadline=10000)
    @given(small_corpus)
    def test_threshold_one_no_edges(self, corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        g = build_semantic_graph(emb, corpus, similarity_threshold=1.0)
        assert g.number_of_edges() == 0
