"""Tests for kenon.embeddings."""

import numpy as np
import pytest
from hypothesis import given, settings

from kenon.embeddings import CountVectorizerEmbedder, PMIEmbedder, TfidfEmbedder
from tests.strategies import small_corpus


class TestCountVectorizerEmbedder:
    """Unit tests for CountVectorizerEmbedder."""

    def test_fit_transform_shape(self, sample_corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        mat = emb.fit_transform(sample_corpus)
        assert mat.shape[0] == len(sample_corpus)
        assert mat.ndim == 2

    def test_vocabulary_after_fit(self, sample_corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        emb.fit(sample_corpus)
        vocab = emb.vocabulary
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

    def test_vocabulary_before_fit_raises(self) -> None:
        emb = CountVectorizerEmbedder()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = emb.vocabulary

    def test_dtype_float(self, sample_corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        mat = emb.fit_transform(sample_corpus)
        assert mat.dtype == np.float64

    def test_transform_after_fit(self, sample_corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        emb.fit(sample_corpus)
        mat = emb.transform(sample_corpus)
        assert mat.shape[0] == len(sample_corpus)


class TestTfidfEmbedder:
    """Unit tests for TfidfEmbedder."""

    def test_fit_transform_shape(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        mat = emb.fit_transform(sample_corpus)
        assert mat.shape[0] == len(sample_corpus)

    def test_vocabulary_after_fit(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        emb.fit(sample_corpus)
        vocab = emb.vocabulary
        assert isinstance(vocab, dict)

    def test_vocabulary_before_fit_raises(self) -> None:
        emb = TfidfEmbedder()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = emb.vocabulary

    def test_dtype_float(self, sample_corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        mat = emb.fit_transform(sample_corpus)
        assert mat.dtype == np.float64


class TestPMIEmbedder:
    """Unit tests for PMIEmbedder."""

    def test_fit_transform(self) -> None:
        corpus = ["the cat sat on the mat"] * 20 + ["the dog ran on the road"] * 20
        emb = PMIEmbedder(n_components=2, window=3)
        mat = emb.fit_transform(corpus)
        assert mat.ndim == 2
        assert mat.dtype == np.float64

    def test_vocabulary_after_fit(self) -> None:
        corpus = ["the cat sat on the mat"] * 20 + ["the dog ran on the road"] * 20
        emb = PMIEmbedder(n_components=2, window=3)
        emb.fit(corpus)
        vocab = emb.vocabulary
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

    def test_vocabulary_before_fit_raises(self) -> None:
        emb = PMIEmbedder(n_components=2)
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = emb.vocabulary


class TestEmbedderProperties:
    """Property-based tests for embedders."""

    @settings(max_examples=20, deadline=10000)
    @given(small_corpus)
    def test_count_shape_matches_corpus(self, corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        mat = emb.fit_transform(corpus)
        assert mat.shape[0] == len(corpus)

    @settings(max_examples=20, deadline=10000)
    @given(small_corpus)
    def test_tfidf_shape_matches_corpus(self, corpus: list[str]) -> None:
        emb = TfidfEmbedder()
        mat = emb.fit_transform(corpus)
        assert mat.shape[0] == len(corpus)

    @settings(max_examples=20, deadline=10000)
    @given(small_corpus)
    def test_count_dtype_is_float(self, corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        mat = emb.fit_transform(corpus)
        assert np.issubdtype(mat.dtype, np.floating)

    @settings(max_examples=10, deadline=10000)
    @given(small_corpus)
    def test_vocabulary_stable_after_refit(self, corpus: list[str]) -> None:
        emb = CountVectorizerEmbedder()
        emb.fit(corpus)
        vocab1 = emb.vocabulary
        emb.fit(corpus)
        vocab2 = emb.vocabulary
        assert vocab1 == vocab2
