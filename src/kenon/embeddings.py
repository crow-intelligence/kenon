"""Corpus-internal token embeddings using count, TF-IDF, and PMI methods.

All embedders derive statistics exclusively from the supplied text —
no external training corpus is involved. Three backends are available:

- ``CountVectorizerEmbedder``: raw term frequencies via sklearn.
- ``TfidfEmbedder``: TF-IDF via sklearn.
- ``PMIEmbedder``: PPMI + SVD via ``chronowords``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from kenon._types import Matrix


class EmbedderProtocol(Protocol):
    """Common interface for all corpus-internal embedders.

    Contract:
        - ``fit`` must be called before ``transform`` or ``vocabulary``.
        - ``fit_transform`` is equivalent to calling ``fit`` then ``transform``.
        - ``vocabulary`` raises ``RuntimeError`` if accessed before ``fit``.
    """

    def fit(self, corpus: list[str]) -> None:
        """Fit the embedder on a corpus of strings."""

    def transform(self, corpus: list[str]) -> Matrix:
        """Return embedding matrix (n_docs x n_features)."""

    def fit_transform(self, corpus: list[str]) -> Matrix:
        """Fit and transform in one step."""

    @property
    def vocabulary(self) -> dict[str, int]:
        """Mapping from token to column index."""


class CountVectorizerEmbedder:
    """Corpus-internal count-based token embeddings via sklearn CountVectorizer.

    Args:
        stopwords: Stopword set to exclude. Pass ``None`` to keep all tokens.
        min_df: Minimum document frequency (int or float).
        max_df: Maximum document frequency (int or float).
        ngram_range: Tuple ``(min_n, max_n)`` for n-gram extraction.

    Contract:
        - ``vocabulary`` raises ``RuntimeError`` if accessed before ``fit``.
        - Output matrix dtype is always float64.

    Example:
        >>> emb = CountVectorizerEmbedder()
        >>> mat = emb.fit_transform(["the cat sat", "the dog ran"])
        >>> mat.shape[0]
        2
    """

    def __init__(
        self,
        stopwords: frozenset[str] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        ngram_range: tuple[int, int] = (1, 1),
    ) -> None:
        stop = list(stopwords) if stopwords else None
        self._vectorizer = CountVectorizer(
            stop_words=stop,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Fit the embedder on a corpus of strings.

        Args:
            corpus: List of document strings.
        """
        self._vectorizer.fit(corpus)
        self._fitted = True

    def transform(self, corpus: list[str]) -> Matrix:
        """Return embedding matrix (n_docs x n_features).

        Args:
            corpus: List of document strings.

        Returns:
            A 2-D float64 numpy array.
        """
        return self._vectorizer.transform(corpus).toarray().astype(np.float64)

    def fit_transform(self, corpus: list[str]) -> Matrix:
        """Fit and transform in one step.

        Args:
            corpus: List of document strings.

        Returns:
            A 2-D float64 numpy array.
        """
        result = self._vectorizer.fit_transform(corpus).toarray().astype(np.float64)
        self._fitted = True
        return result

    @property
    def vocabulary(self) -> dict[str, int]:
        """Mapping from token to column index.

        Raises:
            RuntimeError: If accessed before ``fit``.
        """
        if not self._fitted:
            msg = "Embedder has not been fitted yet. Call fit() first."
            raise RuntimeError(msg)
        return dict(self._vectorizer.vocabulary_)


class TfidfEmbedder:
    """Corpus-internal TF-IDF token embeddings via sklearn TfidfVectorizer.

    Args:
        stopwords: Stopword set to exclude.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.
        sublinear_tf: Apply sublinear TF scaling (``1 + log(tf)``).

    Contract:
        - ``vocabulary`` raises ``RuntimeError`` if accessed before ``fit``.
        - Output matrix dtype is always float64.

    Example:
        >>> emb = TfidfEmbedder()
        >>> mat = emb.fit_transform(["the cat sat", "the dog ran"])
        >>> mat.shape[0]
        2
    """

    def __init__(
        self,
        stopwords: frozenset[str] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = False,
    ) -> None:
        stop = list(stopwords) if stopwords else None
        self._vectorizer = TfidfVectorizer(
            stop_words=stop,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Fit the embedder on a corpus of strings.

        Args:
            corpus: List of document strings.
        """
        self._vectorizer.fit(corpus)
        self._fitted = True

    def transform(self, corpus: list[str]) -> Matrix:
        """Return embedding matrix (n_docs x n_features).

        Args:
            corpus: List of document strings.

        Returns:
            A 2-D float64 numpy array.
        """
        return self._vectorizer.transform(corpus).toarray().astype(np.float64)

    def fit_transform(self, corpus: list[str]) -> Matrix:
        """Fit and transform in one step.

        Args:
            corpus: List of document strings.

        Returns:
            A 2-D float64 numpy array.
        """
        result = self._vectorizer.fit_transform(corpus).toarray().astype(np.float64)
        self._fitted = True
        return result

    @property
    def vocabulary(self) -> dict[str, int]:
        """Mapping from token to column index.

        Raises:
            RuntimeError: If accessed before ``fit``.
        """
        if not self._fitted:
            msg = "Embedder has not been fitted yet. Call fit() first."
            raise RuntimeError(msg)
        return dict(self._vectorizer.vocabulary_)


class PMIEmbedder:
    """Corpus-internal PPMI embeddings via ``chronowords.algebra.SVDAlgebra``.

    Builds a Positive Pointwise Mutual Information matrix from the supplied
    corpus using chronowords' Cython-optimised kernel, then applies SVD to
    produce dense word vectors. All statistics are derived exclusively from
    the supplied text — no external training corpus is involved.

    Note:
        ``SVDAlgebra.train()`` accepts a generator of text lines and tokenises
        internally by whitespace splitting. It filters words shorter than
        ``min_word_length`` (default 3). The ``transform()`` method on this
        class returns word-level embeddings (vocab x n_components), not
        document-level embeddings.

    Args:
        n_components: Number of SVD dimensions (i.e. embedding size).
        window: Context window size in tokens passed to ``SVDAlgebra``.
        min_word_length: Minimum word length for vocabulary inclusion.

    Raises:
        ImportError: If ``chronowords`` is not installed.
            Install with: ``uv add chronowords``

    Contract:
        - ``vocabulary`` raises ``RuntimeError`` if accessed before ``fit``.
        - ``transform()`` returns a 2-D float64 array of shape
          ``(len(vocabulary), n_components)``.
        - The embedder is serialisable via ``pickle``.

    Example:
        >>> emb = PMIEmbedder(n_components=50, window=3)
        >>> corpus = ["the cat sat on the mat", "the dog ran on the road"]
        >>> mat = emb.fit_transform(corpus)
        >>> mat.ndim
        2
    """

    def __init__(
        self,
        n_components: int = 100,
        window: int = 5,
        min_word_length: int = 3,
    ) -> None:
        try:
            from chronowords.algebra.svd import SVDAlgebra
        except ImportError as exc:
            msg = (
                "chronowords is required for PMIEmbedder. "
                "Install with: uv add chronowords"
            )
            raise ImportError(msg) from exc

        self._model = SVDAlgebra(
            n_components=n_components,
            window_size=window,
            min_word_length=min_word_length,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Fit the embedder on a corpus of strings.

        Args:
            corpus: List of document strings. Each string is treated as a
                line of text; chronowords tokenises by whitespace internally.
        """
        self._model.train(line for line in corpus)
        self._fitted = True

    def transform(self, corpus: list[str]) -> Matrix:
        """Return word embedding matrix (n_vocab x n_components).

        Unlike the sklearn-based embedders which return document-level
        matrices, this returns word-level embeddings since PMI operates
        at the word level.

        Args:
            corpus: Ignored if already fitted. Provided for protocol
                compatibility.

        Returns:
            A 2-D float64 numpy array of shape ``(len(vocabulary), n_components)``.

        Raises:
            RuntimeError: If the embedder has not been fitted.
        """
        if not self._fitted or self._model.embeddings is None:
            msg = "Embedder has not been fitted yet. Call fit() first."
            raise RuntimeError(msg)
        return self._model.embeddings.astype(np.float64)

    def fit_transform(self, corpus: list[str]) -> Matrix:
        """Fit and transform in one step.

        Args:
            corpus: List of document strings.

        Returns:
            A 2-D float64 numpy array.
        """
        self.fit(corpus)
        return self.transform(corpus)

    @property
    def vocabulary(self) -> dict[str, int]:
        """Mapping from token to column index.

        Raises:
            RuntimeError: If accessed before ``fit``.
        """
        if not self._fitted:
            msg = "Embedder has not been fitted yet. Call fit() first."
            raise RuntimeError(msg)
        return dict(self._model._vocab_index)
